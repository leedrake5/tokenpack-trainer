import torch
from transformers import Seq2SeqTrainer
from torch.nn.utils.rnn import pad_sequence
import time
from transformers.utils import logging
logger = logging.get_logger(__name__)

class TokenPackTrainer(Seq2SeqTrainer):
    """
    Trainer that:
      - (optionally) truncates encoder/decoder sequences to hard caps,
      - splits each batch into token-budgeted microbatches (enc_len + dec_len),
      - tracks/logs max lengths seen so far.

    If use_cpu_microbatch = True:
      - keep full HF batch on CPU for planning,
      - move only microbatches to GPU via _to_device.

    If use_cpu_microbatch = False:
      - use HF/Accelerate's normal device placement,
      - microbatch on whatever device inputs are already on (typically GPU).
    """
    
        # --- Compatibility shim: silence tokenizer deprecation, use processing_class internally ---
    @property
    def tokenizer(self):
        # HF now uses "processing_class" in lieu of "tokenizer"
        return getattr(self, "processing_class", None)

    @tokenizer.setter
    def tokenizer(self, value):
        # HF now uses "processing_class" in lieu of "tokenizer"
        self.processing_class = value

    def __init__(
        self,
        *args,
        max_tokens_per_microbatch: int = 400,
        max_eval_tokens_per_microbatch: int | None = None,
        max_encoder_len: int | None = None,
        max_decoder_len: int | None = None,
        max_tokens_per_batch: int | None = None,
        max_examples_per_microbatch: int | None = None,
        length_column_name: str = "input_length",
        log_longest_every: int = 0,   # 0 = disable logging, >0 = log every N steps
        use_cpu_microbatch: bool = True,
        eval_mode: str | None = None,  #use classic hf or switch to "token_aware_metrics" to use token packing
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_tokens_per_microbatch = int(max_tokens_per_microbatch)
        self.max_encoder_len = int(max_encoder_len) if max_encoder_len is not None else None
        self.max_decoder_len = int(max_decoder_len) if max_decoder_len is not None else None
        self.max_tokens_per_batch = max_tokens_per_batch
        self.length_column_name = length_column_name
        self.log_longest_every = int(log_longest_every)
        self.use_cpu_microbatch = bool(use_cpu_microbatch)
        self.max_examples_per_microbatch = max_examples_per_microbatch
        self.max_eval_tokens_per_microbatch = (
            int(max_eval_tokens_per_microbatch/4)
            if max_eval_tokens_per_microbatch is not None
            else self.max_tokens_per_microbatch
        )
        self.eval_mode = eval_mode or "token_aware_metrics"
        self.debug = debug

        # running maxima
        self._max_seen_enc_len = 0
        self._max_seen_dec_len = 0
        self._max_seen_total_len = 0
        self._num_trunc_hits = 0
        
        if getattr(self, "processing_class", None) is None:
            # fallback, just in case
            if hasattr(self, "_tokenizer") and self._tokenizer is not None:
                self.processing_class = self._tokenizer

        if self.args.gradient_accumulation_steps != 1:
            print(
                "[HierarchicalTokenTrainer] Warning: gradient_accumulation_steps != 1.\n"
                "You now have two layers of accumulation (HF + microbatch). "
                "Make sure this is intentional."
            )

    @staticmethod
    def _pad_and_concat(arrays, pad_value: int):
        """
        Given a list of 2D arrays [N_i, T_i], pad each to max_T along dim 1
        and stack into a single [sum_i N_i, max_T] array.
        """
        if not arrays:
            return np.zeros((0, 0), dtype=np.int64)

        # All arrays must be 2D
        max_T = max(a.shape[1] for a in arrays)
        total_N = sum(a.shape[0] for a in arrays)
        dtype = arrays[0].dtype

        out = np.full((total_N, max_T), pad_value, dtype=dtype)

        offset = 0
        for a in arrays:
            n, t = a.shape
            out[offset : offset + n, :t] = a
            offset += n

        return out
        
        
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if self.max_tokens_per_batch is None:
            return super().get_train_dataloader()

        ds = self.train_dataset

        # 🔹 Drop raw text / unused columns *here*
        if hasattr(ds, "column_names"):
            keep_cols = {
                "input_ids",
                "attention_mask",
                "labels",
                "decoder_input_ids",
                "decoder_attention_mask",
                self.length_column_name,   # "input_length"
            }
            to_remove = [c for c in ds.column_names if c not in keep_cols]
            if to_remove:
                ds = ds.remove_columns(to_remove)

        # Now safe to use for lengths + collator
        if hasattr(ds, "column_names"):
            lengths = ds[self.length_column_name]
        else:
            lengths = [ds[i][self.length_column_name] for i in range(len(ds))]

        batch_sampler = LengthBucketedBatchSampler(
            lengths=lengths,
            max_tokens_per_batch=self.max_tokens_per_batch,
            bucket_size=16,
            shuffle=True,
            drop_last=False,
        )

        return DataLoader(
            ds,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=getattr(self.args, "dataloader_prefetch_factor", 2),
        )

    # ----------------------------------------------------------------------
    # Truncation helper
    # ----------------------------------------------------------------------

    def _truncate_batch(self, inputs: dict) -> dict:
        """
        In-place truncate encoder/decoder sequences to hard caps, if set.

        - Encoder side: input_ids, attention_mask
        - Decoder side: labels, decoder_input_ids, decoder_attention_mask

        Works for:
          - encoder-only (no labels -> only encoder is truncated),
          - seq2seq (2D labels).
        """
        # Encoder truncation
        if self.max_encoder_len is not None:
            for k in ("input_ids", "attention_mask"):
                if k in inputs and isinstance(inputs[k], torch.Tensor):
                    t = inputs[k]
                    if t.ndim == 2 and t.size(1) > self.max_encoder_len:
                        inputs[k] = t[:, : self.max_encoder_len]
                        self._num_trunc_hits += 1

        # Decoder truncation (only if 2D sequence labels are present)
        if self.max_decoder_len is not None and "labels" in inputs:
            labels = inputs["labels"]
            if isinstance(labels, torch.Tensor) and labels.ndim == 2:
                if labels.size(1) > self.max_decoder_len:
                    inputs["labels"] = labels[:, : self.max_decoder_len]
                    self._num_trunc_hits += 1

                for k in ("decoder_input_ids", "decoder_attention_mask"):
                    if k in inputs and isinstance(inputs[k], torch.Tensor):
                        t = inputs[k]
                        if t.ndim == 2 and t.size(1) > self.max_decoder_len:
                            inputs[k] = t[:, : self.max_decoder_len]
                            self._num_trunc_hits += 1

        return inputs

    # ----------------------------------------------------------------------
    # Length helpers (enc+dec), strict cap check
    # ----------------------------------------------------------------------

    def _compute_lengths_enc_dec(self, inputs):
        """
        Compute per-example (enc_len, dec_len, total_len) from inputs.

        - enc_len: sum over attention_mask
        - dec_len: # of non -100 labels (if 2D); 0 for pretraining/encoder-only.
        """
        am = inputs.get("attention_mask", None)
        if am is None:
            raise ValueError(
                "attention_mask is required to compute lengths for HierarchicalTokenTrainer. "
                "Make sure your collator provides it."
            )
        if not isinstance(am, torch.Tensor):
            am = torch.as_tensor(am, device="cpu")

        enc_len = am.sum(dim=-1)  # (B,)

        labels = inputs.get("labels", None)
        if labels is None:
            dec_len = torch.zeros_like(enc_len)
        else:
            if not isinstance(labels, torch.Tensor):
                labels = torch.as_tensor(labels)
            if labels.ndim == 2:
                dec_len = (labels != -100).sum(dim=-1)
            else:
                # 1D class labels or something non-seq -> treat as encoder-only
                dec_len = torch.zeros_like(enc_len)

        total_len = enc_len + dec_len

        # update maxima
        enc_max = int(enc_len.max().item())
        dec_max = int(dec_len.max().item())
        tot_max = int(total_len.max().item())
        self._max_seen_enc_len = max(self._max_seen_enc_len, enc_max)
        self._max_seen_dec_len = max(self._max_seen_dec_len, dec_max)
        self._max_seen_total_len = max(self._max_seen_total_len, tot_max)

        # OPTIONAL: soft check instead of hard error
        if self.max_tokens_per_microbatch is not None:
            over = total_len > self.max_tokens_per_microbatch
            if over.any():
                # e.g., just log once per step or so; do NOT raise
                idx = int(torch.nonzero(over, as_tuple=False)[0].item())
                offending_total = int(total_len[idx].item())
                offending_enc = int(enc_len[idx].item())
                offending_dec = int(dec_len[idx].item())
                if self.control.should_log:
                    self.log(
                        {
                            "warn_example_over_cap_enc_len": float(offending_enc),
                            "warn_example_over_cap_dec_len": float(offending_dec),
                            "warn_example_over_cap_total":   float(offending_total),
                        }
                    )
                # No raise – just report
        return enc_len, dec_len, total_len

    @staticmethod
    def _slice_inputs(self_or_none, inputs, indices):
        """
        Slice a dict of tensors along the batch dimension for the given indices.
        Non-tensor values or tensors with mismatched batch dims are passed through.

        This function does NOT *change* device of the batch; it just matches the
        device of the index tensor to whatever the batch tensors are already on.
        """
        if not indices:
            return inputs

        # Find an example tensor to infer device
        example_tensor = None
        for v in inputs.values():
            if isinstance(v, torch.Tensor):
                example_tensor = v
                break

        device = example_tensor.device if example_tensor is not None else torch.device("cpu")
        idx = torch.as_tensor(indices, dtype=torch.long, device=device)

        out = {}
        batch_dim = None
        if "input_ids" in inputs and isinstance(inputs["input_ids"], torch.Tensor):
            batch_dim = inputs["input_ids"].size(0)

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if batch_dim is not None and v.size(0) == batch_dim:
                    out[k] = v.index_select(0, idx)
                else:
                    out[k] = v
            else:
                out[k] = v

        return out

    # ------------------------------------------------------------------
    # Disable Trainer's automatic device placement
    # ------------------------------------------------------------------
    def _prepare_input(self, data):
        """
        If we are doing CPU-based microbatching, do nothing here.
        Otherwise, fall back to Seq2SeqTrainer's normal behavior.
        """
        if self.use_cpu_microbatch:
            return data
        return super()._prepare_input(data)

    def _prepare_inputs(self, inputs):
        """
        If we are doing CPU-based microbatching, do nothing here.
        Otherwise, fall back to Seq2SeqTrainer's normal behavior.
        """
        if self.use_cpu_microbatch:
            return inputs
        return super()._prepare_inputs(inputs)

    def _to_device(self, inputs: dict) -> dict:
        """
        Explicitly move a microbatch dict to self.args.device.
        Used only when use_cpu_microbatch=True.
        """
        device = self.args.device
        out = {}

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                kwargs = {"device": device}
                if self.is_deepspeed_enabled and (torch.is_floating_point(v) or torch.is_complex(v)):
                    kwargs["dtype"] = self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()
                out[k] = v.to(**kwargs)
            else:
                out[k] = v

        return out

    def _move_to_cpu(self, inputs: dict) -> dict:
        cpu_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                cpu_inputs[k] = v.detach().to("cpu")
            else:
                cpu_inputs[k] = v
        return cpu_inputs

    def _make_microbatches(self, inputs, max_tokens_per_microbatch: int | None = None):
        # Compute lengths
        enc_len, dec_len, _ = self._compute_lengths_enc_dec(inputs)

        alpha = 2.0  # decoder weight
        effective_len = enc_len + alpha * dec_len
        lengths = [int(L) for L in effective_len.tolist()]
        N = len(lengths)

        budget = int(max_tokens_per_microbatch or self.max_tokens_per_microbatch)
        max_B = self.max_examples_per_microbatch  # may be None

        microbatches = []
        cur_indices: list[int] = []
        cur_tokens = 0

        sorted_indices = sorted(range(N), key=lambda i: lengths[i])

        for i in sorted_indices:
            L = lengths[i]

            too_many_tokens = cur_indices and (cur_tokens + L) > budget
            too_many_examples = max_B is not None and len(cur_indices) >= max_B

            if too_many_tokens or too_many_examples:
                microbatches.append(cur_indices)
                cur_indices = [i]
                cur_tokens = L
            else:
                cur_indices.append(i)
                cur_tokens += L

        if cur_indices:
            microbatches.append(cur_indices)

        return [self._slice_inputs(self, inputs, mb_idx) for mb_idx in microbatches]

    def _plan_microbatches(self, inputs):
        """
        Given a batch on CPU, compute total lengths and return
        a list of index lists for each microbatch.

        No tensors are moved to GPU here; we only work with CPU tensors.
        """
        # compute_lengths_enc_dec works fine on CPU
        _, _, total_len = self._compute_lengths_enc_dec(inputs)
        lengths = [int(L) for L in total_len.tolist()]
        N = len(lengths)

        # Sort indices by length ascending (shortest first)
        sorted_indices = sorted(range(N), key=lambda i: lengths[i])

        microbatches = []
        cur_indices = []
        cur_tokens = 0

        for i in sorted_indices:
            L = lengths[i]
            if self.max_tokens_per_microbatch is not None and L > self.max_tokens_per_microbatch:
                # single example exceeds cap; we already logged this in _compute_lengths_enc_dec
                # You can choose to raise here if you want:
                # raise RuntimeError(f"Example {i} has {L} tokens > max_tokens_per_microbatch={self.max_tokens_per_microbatch}")
                pass

            if (
                self.max_tokens_per_microbatch is not None
                and cur_indices
                and (cur_tokens + L) > self.max_tokens_per_microbatch
            ):
                microbatches.append(cur_indices)
                cur_indices = [i]
                cur_tokens = L
            else:
                cur_indices.append(i)
                cur_tokens += L

        if cur_indices:
            microbatches.append(cur_indices)

        return microbatches

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool = False,
        ignore_keys: Optional[List[str]] = None,
    ):
    
        if self.length_column_name in inputs:
            inputs = {k: v for k, v in inputs.items() if k != self.length_column_name}

        # If we need logits/labels, or no microbatching, just use the base implementation
        if (not prediction_loss_only) or (self.max_tokens_per_microbatch is None):
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        
        model.eval()

        # -----------------------------
        # CPU-microbatching path
        # -----------------------------
        if self.use_cpu_microbatch:
            inputs_cpu = self._move_to_cpu(inputs)
            inputs_cpu = self._truncate_batch(inputs_cpu)

            microbatches = self._make_microbatches(
                inputs_cpu,
                max_tokens_per_microbatch=self.max_eval_tokens_per_microbatch,
            )

            if not microbatches:
                return (None, None, None)

            total_loss = 0.0
            total_examples = 0

            for mb in microbatches:
                bsz = mb["input_ids"].size(0)
                total_examples += bsz

                mb = self._to_device(mb)

                # 🔹 drop length column
                if self.length_column_name in mb:
                    mb = {k: v for k, v in mb.items() if k != self.length_column_name}

                with torch.no_grad():
                    with self.compute_loss_context_manager():
                        loss_mb = self.compute_loss(model, mb, return_outputs=False)

                if isinstance(loss_mb, tuple):
                    loss_mb = loss_mb[0]

                loss_mb = loss_mb.detach()
                total_loss += loss_mb * bsz
                print(
                    "mb shapes:",
                    mb["input_ids"].shape,
                    mb["attention_mask"].shape,
                    "labels" in mb and mb["labels"].shape,
                )


            if total_examples == 0:
                return (None, None, None)

            avg_loss = total_loss / total_examples
            return (avg_loss, None, None)

        # -----------------------------
        # GPU-native microbatching path
        # (mirrors training_step)
        # -----------------------------
        inputs = self._prepare_inputs(inputs)  # let HF/Accelerate do device placement
        inputs = self._truncate_batch(inputs)

        microbatches = self._make_microbatches(
            inputs,
            max_tokens_per_microbatch=self.max_eval_tokens_per_microbatch,
        )

        if not microbatches:
            return (None, None, None)

        total_loss = 0.0
        total_examples = 0

        for mb in microbatches:
            bsz = mb["input_ids"].size(0)
            total_examples += bsz

            # 🔹 drop length column
            if self.length_column_name in mb:
                mb = {k: v for k, v in mb.items() if k != self.length_column_name}

            # mb is already on correct device; do NOT call _to_device here
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    loss_mb = self.compute_loss(model, mb, return_outputs=False)

            if isinstance(loss_mb, tuple):
                loss_mb = loss_mb[0]

            loss_mb = loss_mb.detach()
            total_loss += loss_mb * bsz

        if total_examples == 0:
            return (None, None, None)

        avg_loss = total_loss / total_examples
        return (avg_loss, None, None)

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If no token budget or we explicitly want HF-style eval, use default
        if self.max_tokens_per_batch is None or self.eval_mode == "hf":
            return super().get_eval_dataloader(eval_dataset)

        # Otherwise: token-aware eval DataLoader (length-bucketed)
        ds = eval_dataset
        if hasattr(ds, "column_names"):
            lengths = ds[self.length_column_name]
        else:
            lengths = [ds[i][self.length_column_name] for i in range(len(ds))]

        batch_sampler = LengthBucketedBatchSampler(
            lengths=lengths,
            max_tokens_per_batch=self.max_tokens_per_batch,
            bucket_size=16,
            shuffle=False,
            drop_last=False,
        )

        return DataLoader(
            ds,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=getattr(self.args, "dataloader_prefetch_factor", 2),
        )
        
        
    # --------------------------------------------------------------
    # Token-aware evaluation with metrics
    # --------------------------------------------------------------

    def _token_aware_evaluate(
        self,
        eval_dataset=None,
        max_eval_tokens_per_microbatch: int | None = None,
        desc: str = "Eval (token-aware)",
    ):
        import time
        from tqdm.auto import tqdm

        self.model.eval()

        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Need an eval_dataset for token-aware evaluation.")

        # 1) Basic dataloader (no token-budgeting here; we control size via per_device_eval_batch_size)
        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=getattr(self.args, "dataloader_prefetch_factor", 2),
        )

        # 2) Build generation kwargs like HF does
        gen_kwargs: dict[str, Any] = {}

        if self.args.generation_max_length is not None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        if self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams

        if getattr(self.args, "do_sample", False):
            gen_kwargs["do_sample"] = True
            if self.args.top_k is not None:
                gen_kwargs["top_k"] = self.args.top_k
            if self.args.top_p is not None:
                gen_kwargs["top_p"] = self.args.top_p

        # Merge with any internally-prepared _gen_kwargs
        if hasattr(self, "_gen_kwargs") and isinstance(self._gen_kwargs, dict):
            tmp = self._gen_kwargs.copy()
            tmp.update(gen_kwargs)
            gen_kwargs = tmp

        # 3) Loop with tqdm + collect preds/labels
        all_preds = []
        all_labels = []

        num_steps = 0
        num_examples = 0

        total_eval_loss = 0.0
        total_eval_tokens = 0

        start_time = time.time()

        for batch in tqdm(dataloader, desc=desc, leave=False):
            num_steps += 1

            # 1) Loss with our microbatch-aware prediction_step
            with torch.no_grad():
                loss, _, _ = self.prediction_step(
                    self.model,
                    batch,
                    prediction_loss_only=True,
                    ignore_keys=None,
                )

            labels = batch.get("labels", None)
            if labels is None:
                raise ValueError("Eval dataset must have labels for metric computation.")

            if loss is not None:
                # Move labels to CPU just for counting tokens
                labels_cpu = labels.detach().to("cpu")
                num_tokens = int((labels_cpu != -100).sum().item())
                total_eval_loss += float(loss.item()) * num_tokens
                total_eval_tokens += num_tokens

            batch_size = labels.size(0)
            num_examples += int(batch_size)

            # 2) Now prepare inputs for generation (separate from loss)
            batch = self._prepare_inputs(batch)

            # Drop labels & input_length from generation kwargs
            ignore_keys = {
                "labels",
                self.length_column_name,
                "decoder_input_ids",
                "decoder_attention_mask",
            }
            gen_inputs = {k: v for k, v in batch.items() if k not in ignore_keys}

            # 3) Fast generation with use_cache=True
            orig_use_cache = getattr(self.model.config, "use_cache", True)
            self.model.config.use_cache = True
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **gen_inputs,
                    **gen_kwargs,
                )
            self.model.config.use_cache = orig_use_cache

            if generated_tokens.ndim == 1:
                generated_tokens = generated_tokens.unsqueeze(0)

            all_preds.append(generated_tokens.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        end_time = time.time()
        runtime = end_time - start_time if num_steps > 0 else 0.0

        # 4) Concatenate + call compute_metrics
        if len(all_preds) == 0:
            # No batches? Return zeros instead of crashing.
            raw_metrics = {
                "bleu": 0.0,
                "chrf": 0.0,
                "meteor": 0.0,
                "gen_len": 0.0,
            }
        else:
            # ---- PAD PREDICTIONS TO COMMON LENGTH ----
            # all_preds: list of (B_i, L_pred_i)
            max_pred_len = max(p.shape[1] for p in all_preds)
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = 0  # very safe fallback

            padded_preds = []
            for p in all_preds:
                if p.shape[1] < max_pred_len:
                    pad_width = max_pred_len - p.shape[1]
                    p = np.pad(
                        p,
                        pad_width=((0, 0), (0, pad_width)),
                        mode="constant",
                        constant_values=pad_id,
                    )
                padded_preds.append(p)
            preds = np.concatenate(padded_preds, axis=0)

            # ---- PAD LABELS TO COMMON LENGTH ----
            # all_labels: list of (B_i, L_label_i)
            max_label_len = max(l.shape[1] for l in all_labels)
            padded_labels = []
            for l in all_labels:
                if l.shape[1] < max_label_len:
                    pad_width = max_label_len - l.shape[1]
                    l = np.pad(
                        l,
                        pad_width=((0, 0), (0, pad_width)),
                        mode="constant",
                        constant_values=-100,  # ignore index
                    )
                padded_labels.append(l)
            labels = np.concatenate(padded_labels, axis=0)

            # (N, max_pred_len), (N, max_label_len)
            raw_metrics = self.compute_metrics((preds, labels))
            # raw_metrics is assumed to contain {"bleu": ..., "chrf": ..., "meteor": ..., "gen_len": ...}

        # 5) Convert to HF-style eval_* keys + runtime stats
        eval_loss = (
            total_eval_loss / total_eval_tokens
            if total_eval_tokens > 0
            else float("nan")
        )

        metrics = {
            "eval_loss":   float(eval_loss),
            "eval_bleu":   float(raw_metrics.get("bleu", 0.0)),
            "eval_chrf":   float(raw_metrics.get("chrf", 0.0)),
            "eval_meteor": float(raw_metrics.get("meteor", 0.0)),
            "eval_gen_len": float(raw_metrics.get("gen_len", 0.0)),
        }

        if runtime > 0 and num_steps > 0:
            metrics["eval_runtime"] = float(runtime)
            metrics["eval_samples_per_second"] = float(num_examples / runtime)
            metrics["eval_steps_per_second"] = float(num_steps / runtime)

        return metrics
        
    # --------------------------------------------------------------
    # OOM handling + emergency checkpoint
    # --------------------------------------------------------------

    def _handle_oom_and_save(self, err: BaseException, reason: str):
        msg = str(err)

        is_cuda_oom = (
            isinstance(err, torch.cuda.OutOfMemoryError)
            or "CUDA out of memory" in msg
            or "CUBLAS_STATUS_ALLOC_FAILED" in msg
        )
        is_launch_timeout = (
            "cudaErrorLaunchTimeout" in msg
            or "the launch timed out and was terminated" in msg
        )
        is_accelerator_cuda_error = "CUDA error" in msg

        should_save = is_cuda_oom or is_launch_timeout or is_accelerator_cuda_error
        if not should_save:
            raise err

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        ckpt_dir = os.path.join(self.args.output_dir, "last_error_checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.save_model(ckpt_dir)

        if self.args.should_save:
            self.state.save_to_json(os.path.join(ckpt_dir, "trainer_state.json"))
            # <-- only if optimizer/scheduler actually exist
            if self.optimizer is not None:
                self._save_optimizer_and_scheduler(ckpt_dir)

        print(f"\n[HierarchicalTokenTrainer] Saved emergency checkpoint to {ckpt_dir} "
              f"after CUDA error: {msg}\n")

        raise err
        
    # --------------------------------------------------------------
    # OOM-aware wrappers around train/evaluate
    # --------------------------------------------------------------
    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            # Let _handle_oom_and_save decide whether to checkpoint
            return self._handle_oom_and_save(err, reason="train")

    logger = logging.get_logger(__name__)

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
        eval_mode: str | None = None,
    ):
        """
        eval_mode:
          - None / "hf": use standard HF evaluation_loop
          - "token_aware_metrics": use our custom generation+metrics with tqdm,
            but log/save metrics similarly to HF, without the extra
            '***** eval metrics *****' banner from log_metrics().
        """
        # Pick mode: explicit arg wins, else trainer.default
        mode = eval_mode or getattr(self, "eval_mode", None)

        # -----------------------------
        # Token-aware loop
        # -----------------------------
        if mode == "token_aware_metrics":
            if eval_dataset is None:
                eval_dataset = self.eval_dataset
            if eval_dataset is None:
                raise ValueError("Evaluation requires an eval_dataset.")

            start_time = time.time()
            raw_metrics = self._token_aware_evaluate(
                eval_dataset=eval_dataset,
                max_eval_tokens_per_microbatch=self.max_eval_tokens_per_microbatch,
                desc="eval (token-aware)",
            )
            runtime = time.time() - start_time

            num_samples = len(eval_dataset)
            batch_size = self.args.per_device_eval_batch_size
            num_steps = (num_samples + batch_size - 1) // batch_size

            # Add runtime stats (if _token_aware_evaluate didn't already)
            raw_metrics.setdefault("eval_runtime", float(runtime))
            raw_metrics.setdefault(
                "eval_samples_per_second",
                float(num_samples / runtime) if runtime > 0 else 0.0,
            )
            raw_metrics.setdefault(
                "eval_steps_per_second",
                float(num_steps / runtime) if runtime > 0 else 0.0,
            )

            # Attach epoch info (like HF)
            if self.state.epoch is not None:
                raw_metrics[f"{metric_key_prefix}_epoch"] = self.state.epoch
                raw_metrics["epoch"] = self.state.epoch

            metrics = raw_metrics

            # 1) Log to the underlying logger (goes to console + WandB/TF via HF)
            self.log(metrics)

            # 2) Append to log_history so it lands in trainer_state.json
            #    (this is normally done inside log_metrics)
            self.state.log_history.append(metrics)

            # 3) OPTIONAL: detailed print:
            if self.is_world_process_zero():
                logger.info("***** eval results (token-aware) *****")
                for k in sorted(metrics.keys()):
                    logger.info("  %s = %s", k, metrics[k])

            return metrics

        # -----------------------------
        # Fallback: standard HF behavior
        # -----------------------------
        return super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
    # --------------------------------------------------------------
    # Core override
    # --------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False):
        # Strip length column if present
        if self.length_column_name in inputs:
            inputs = {k: v for k, v in inputs.items() if k != self.length_column_name}

        outputs = model(**inputs)
        # standard HF pattern
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        If use_cpu_microbatch:
          - force full batch to CPU,
          - compute lengths + truncate + plan microbatches on CPU,
          - move only microbatches to GPU via _to_device.

        Else:
          - let HF/Accelerate handle device placement via _prepare_inputs,
          - truncate + microbatch on that device (typically GPU).
        """
        model.train()

        # --- DEBUG: see where the batch lives when we first see it ---
        #if "input_ids" in inputs and isinstance(inputs["input_ids"], torch.Tensor):
            #print(f"[HierarchicalTokenTrainer] training_step got input_ids on device: {inputs['input_ids'].device}")

        # ---------------- CPU-microbatching path ----------------
        if self.use_cpu_microbatch:
            # Make absolutely sure everything is on CPU before we do any planning
            inputs = self._move_to_cpu(inputs)

            try:
                # Compute lengths on CPU only
                enc_len, dec_len, total_len = self._compute_lengths_enc_dec(inputs)
            except Exception:
                # If this is where an error hits, you'll still see it
                raise

            inputs = self._truncate_batch(inputs)
            microbatches = self._make_microbatches(inputs)

        # ---------------- GPU-native microbatching path ----------------
        else:
            inputs = self._prepare_inputs(inputs)
            inputs = self._truncate_batch(inputs)

            enc_len, dec_len, total_len = self._compute_lengths_enc_dec(inputs)
            microbatches = self._make_microbatches(inputs)

            # --- DEBUG: log summary for this HF batch ---
            with torch.no_grad():
                mb_stats = []
                for mb in microbatches:
                    am_mb = mb["attention_mask"]
                    enc_mb_len = am_mb.sum(dim=-1)                  # (B,)
                    labels_mb = mb.get("labels", None)
                    if labels_mb is not None and labels_mb.ndim == 2:
                        dec_mb_len = (labels_mb != -100).sum(dim=-1)
                    else:
                        dec_mb_len = torch.zeros_like(enc_mb_len)

                    total_mb_len = enc_mb_len + dec_mb_len
                    mb_stats.append({
                        "B": int(enc_mb_len.size(0)),
                        "enc_tokens": int(enc_mb_len.sum().item()),
                        "dec_tokens": int(dec_mb_len.sum().item()),
                        "total_tokens": int(total_mb_len.sum().item()),
                        "max_enc": int(enc_mb_len.max().item()),
                        "max_dec": int(dec_mb_len.max().item()),
                    })
                if self.debug:
                    print("[DEBUG] HF batch has", len(microbatches), "microbatches")
                    for j, s in enumerate(mb_stats[:4]):  # print only first few
                        print(
                            f"   mb{j}: B={s['B']}, "
                            f"enc_tokens={s['enc_tokens']}, dec_tokens={s['dec_tokens']}, "
                            f"total={s['total_tokens']}, "
                            f"max_enc={s['max_enc']}, max_dec={s['max_dec']}"
                        )

        # ---------------- Common microbatch execution ----------------
        num_micro = max(len(microbatches), 1)
        total_loss = 0.0
        total_examples = sum(mb["input_ids"].size(0) for mb in microbatches)

        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()

        for mb_idx, mb in enumerate(microbatches):
            if self.use_cpu_microbatch:
                mb = self._to_device(mb)

            # 🔹 Drop length column so it never reaches the model
            if self.length_column_name in mb:
                mb = {k: v for k, v in mb.items() if k != self.length_column_name}

            bsz = mb["input_ids"].size(0)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, mb)

            if isinstance(loss, tuple):
                loss = loss[0]

            loss = loss * (bsz / total_examples)

            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            self.accelerator.backward(loss)
            total_loss += loss.detach().float()

            # --- DEBUG: memory after this microbatch ---
            if self.debug:
                after = torch.cuda.memory_allocated()
                peak = torch.cuda.max_memory_allocated()
                print(
                    f"[DEBUG] mb {mb_idx}: mem_alloc={after/1e9:.2f} GB, "
                    f"peak={peak/1e9:.2f} GB"
                )

        avg_loss = total_loss / num_micro

        if (
            self.log_longest_every > 0
            and self.state.global_step > 0
            and (self.state.global_step % self.log_longest_every == 0)
        ):
            self.log(
                {
                    "max_seen_enc_len":   float(self._max_seen_enc_len),
                    "max_seen_dec_len":   float(self._max_seen_dec_len),
                    "max_seen_total_len": float(self._max_seen_total_len),
                    "num_trunc_hits":     float(self._num_trunc_hits),
                }
            )

        return avg_loss
