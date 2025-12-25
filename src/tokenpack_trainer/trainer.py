import torch
from transformers import Seq2SeqTrainer
from torch.nn.utils.rnn import pad_sequence
import time
from typing import Any, List, Optional
from transformers.utils import logging
from torch.utils.data import DataLoader
logger = logging.get_logger(__name__)
import os
import numpy as np

from .samplers import LengthBucketedBatchSampler


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
    
        # --- Compatibility shim: silence processing_class deprecation, use processing_class internally ---
    @property
    def tokenizer(self):
        # In recent HF, Trainer stores it as .processing_class
        pc = getattr(self, "processing_class", None)
        if pc is not None:
            return pc
        # Fallback if some older version stored it differently
        return getattr(self, "_processing_class", None)

    @tokenizer.setter
    def tokenizer(self, value):
        # Route legacy assignment to the new attribute
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
        oom_max_retries: int = 3,
        oom_shrink_B: float = 0.5,           # halve B on OOM
        oom_shrink_tokens: float = 0.85,     # then shrink token budget
        oom_min_B: int = 1,
        oom_min_tokens: int = 64,
        oom_skip_batch_on_fail: bool = True, # if retries exhausted, just skip batch
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
            int(max_eval_tokens_per_microbatch)
            if max_eval_tokens_per_microbatch is not None
            else self.max_tokens_per_microbatch
        )
        self.eval_mode = eval_mode or "token_aware_metrics"
        self.debug = debug
        self.oom_max_retries = int(oom_max_retries)
        self.oom_shrink_B = float(oom_shrink_B)
        self.oom_shrink_tokens = float(oom_shrink_tokens)
        self.oom_min_B = int(oom_min_B)
        self.oom_min_tokens = int(oom_min_tokens)
        self.oom_skip_batch_on_fail = bool(oom_skip_batch_on_fail)

        self._oom_events = 0
        self._oom_skipped_batches = 0

        self._regime_limits = {}  # key -> {"B": int|None, "T": int, "stable": int}
        self._regime_bucket_size = 128   # effective-length bucket size (tokens); tune 64/128/256
        self._regime_ramp_every = 50     # successes before increasing limits
        self._regime_ramp_B = 1.10       # +10% B on ramp
        self._regime_ramp_T = 1.03       # +3% token budget on ramp
        self._regime_min_B = getattr(self, "oom_min_B", 1)
        self._regime_min_T = getattr(self, "oom_min_tokens", 64)

        # Defaults used to initialize new regimes
        self._regime_default_B = self.max_examples_per_microbatch
        self._regime_default_T = self.max_tokens_per_microbatch

        # running maxima
        self._max_seen_enc_len = 0
        self._max_seen_dec_len = 0
        self._max_seen_total_len = 0
        self._num_trunc_hits = 0
        
        if getattr(self, "processing_class", None) is None:
            # fallback, just in case
            if hasattr(self, "_processing_class") and self._processing_class is not None:
                self.processing_class = self._processing_class

        if self.args.gradient_accumulation_steps != 1:
            print(
                "[TokenPackTrainer] Warning: gradient_accumulation_steps != 1.\n"
                "You now have two layers of accumulation (HF + microbatch). "
                "Make sure this is intentional."
            )

        # Optional: if something populated _processing_class (older HF),
        # but processing_class is missing, sync it.
        if getattr(self, "processing_class", None) is None and getattr(self, "_processing_class", None) is not None:
            self.processing_class = self._processing_class


    def _regime_key_from_inputs(self, inputs_cpu: dict) -> int:
        """
        Compute a coarse "regime key" from the maximum effective length
        in this HF batch: eff = enc_len + alpha * dec_len.
        """
        enc_len, dec_len, _ = self._compute_lengths_enc_dec(inputs_cpu)
        alpha = 2.0
        eff = enc_len + alpha * dec_len
        mx = int(eff.max().item()) if eff.numel() else 0
        bs = int(self._regime_bucket_size)
        return int((mx + bs - 1) // bs) if bs > 0 else mx

    def _regime_state(self, key: int) -> dict:
        st = self._regime_limits.get(key)
        if st is None:
            st = {"B": self._regime_default_B, "T": self._regime_default_T, "stable": 0}
            # If defaults are None, pick something reasonable so we can adapt
            if st["B"] is None:
                st["B"] = 16
            if st["T"] is None:
                st["T"] = 400
            self._regime_limits[key] = st
        return st

    def _apply_regime_limits(self, key: int):
        st = self._regime_state(key)
        self.max_examples_per_microbatch = st["B"]
        self.max_tokens_per_microbatch = st["T"]
        # keep eval budget <= train budget
        if getattr(self, "max_eval_tokens_per_microbatch", None) is not None:
            self.max_eval_tokens_per_microbatch = min(self.max_eval_tokens_per_microbatch, self.max_tokens_per_microbatch)

    def _regime_on_success(self, key: int):
        st = self._regime_state(key)
        st["stable"] += 1

        if st["stable"] % int(self._regime_ramp_every) == 0:
            # ramp B up gently
            if st["B"] is None:
                st["B"] = 16
            st["B"] = max(self._regime_min_B, int(st["B"] * float(self._regime_ramp_B)) + 1)

            # ramp token budget slightly (optional)
            st["T"] = max(self._regime_min_T, int(st["T"] * float(self._regime_ramp_T)))

    def _regime_on_oom(self, key: int):
        st = self._regime_state(key)
        st["stable"] = 0

        # Shrink B first
        if st["B"] is None:
            st["B"] = 16
        new_B = max(self._regime_min_B, int(st["B"] * 0.7))
        if new_B < st["B"]:
            st["B"] = new_B
            return

        # If B can't shrink further, shrink token budget
        st["T"] = max(self._regime_min_T, int(st["T"] * 0.85))

    def _regime_key_from_batch(self, inputs_cpu) -> int:
        enc_len, dec_len, _ = self._compute_lengths_enc_dec(inputs_cpu)
        alpha = 2.0
        eff = enc_len + alpha * dec_len
        mx = int(eff.max().item())
        bucket_size = 128   # choose something coarse
        return int((mx + bucket_size - 1) // bucket_size)

    def _is_cuda_oom(self, err: BaseException) -> bool:
        msg = str(err)
        return (
            isinstance(err, torch.cuda.OutOfMemoryError)
            or "CUDA out of memory" in msg
            or "CUBLAS_STATUS_ALLOC_FAILED" in msg
            or "cudaMalloc" in msg
        )

    def _oom_cleanup(self):
        # Clear gradients that may be partially accumulated
        try:
            self.accelerator.zero_grad(set_to_none=True)
        except Exception:
            # fallback
            try:
                for p in self.model.parameters():
                    p.grad = None
            except Exception:
                pass

        # Clear CUDA allocator state
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _oom_shrink_limits(self):
        """
        Prefer shrinking max_examples_per_microbatch (B) first.
        If already minimal, shrink max_tokens_per_microbatch.
        """
        changed = False

        # Shrink B (if set)
        if self.max_examples_per_microbatch is not None and self.max_examples_per_microbatch > self.oom_min_B:
            new_B = max(self.oom_min_B, int(self.max_examples_per_microbatch * self.oom_shrink_B))
            if new_B < self.max_examples_per_microbatch:
                self.max_examples_per_microbatch = new_B
                changed = True

        # If B is None (unbounded) or already minimal, shrink token budget
        if not changed:
            if self.max_tokens_per_microbatch is not None and self.max_tokens_per_microbatch > self.oom_min_tokens:
                new_T = max(self.oom_min_tokens, int(self.max_tokens_per_microbatch * self.oom_shrink_tokens))
                if new_T < self.max_tokens_per_microbatch:
                    self.max_tokens_per_microbatch = new_T
                    # keep eval consistent-ish
                    self.max_eval_tokens_per_microbatch = min(self.max_eval_tokens_per_microbatch, self.max_tokens_per_microbatch)
                    changed = True

        return changed

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

    def _compact_microbatch(self, mb: dict) -> dict:
        # Trim encoder side to max attention_mask sum in this microbatch
        if "attention_mask" in mb and isinstance(mb["attention_mask"], torch.Tensor):
            am = mb["attention_mask"]
            if am.ndim == 2:
                enc_max = int(am.sum(dim=-1).max().item())
                enc_max = max(enc_max, 1)
                for k in ("input_ids", "attention_mask"):
                    if k in mb and isinstance(mb[k], torch.Tensor) and mb[k].ndim == 2:
                        mb[k] = mb[k][:, :enc_max]

        # Trim decoder side to max non -100 labels in this microbatch
        if "labels" in mb and isinstance(mb["labels"], torch.Tensor):
            lab = mb["labels"]
            if lab.ndim == 2:
                dec_max = int((lab != -100).sum(dim=-1).max().item())
                dec_max = max(dec_max, 1)
                for k in ("labels", "decoder_input_ids", "decoder_attention_mask"):
                    if k in mb and isinstance(mb[k], torch.Tensor) and mb[k].ndim == 2:
                        mb[k] = mb[k][:, :dec_max]

        return mb
        
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
            raw_lengths = ds[self.length_column_name]
        else:
            raw_lengths = [ds[i][self.length_column_name] for i in range(len(ds))]

        # IMPORTANT: cap lengths to what the model will actually see
        if self.max_encoder_len is not None:
            lengths_for_sampler = [min(int(L), self.max_encoder_len) for L in raw_lengths]
        else:
            lengths_for_sampler = [int(L) for L in raw_lengths]

        batch_sampler = LengthBucketedBatchSampler(
            lengths=lengths_for_sampler,
            max_tokens_per_batch=self.max_tokens_per_batch,
            bucket_size=16,
            shuffle=True,
            drop_last=False,
            long_behavior="truncate",
            max_length_in_batch=self.max_encoder_len,  # optional but recommended
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
                "attention_mask is required to compute lengths for TokenPackTrainer. "
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

    def _slice_inputs(self, inputs: dict, indices: list[int]) -> dict:
        if not indices:
            return inputs

        # Infer device from any tensor in inputs
        example_tensor = next((v for v in inputs.values() if isinstance(v, torch.Tensor)), None)
        device = example_tensor.device if example_tensor is not None else torch.device("cpu")
        idx = torch.as_tensor(indices, dtype=torch.long, device=device)

        # Infer batch dim from input_ids if present
        batch_dim = None
        if "input_ids" in inputs and isinstance(inputs["input_ids"], torch.Tensor):
            batch_dim = inputs["input_ids"].size(0)

        out = {}
        for k, v in inputs.items():
            if (
                isinstance(v, torch.Tensor)
                and batch_dim is not None
                and v.ndim >= 1
                and v.size(0) == batch_dim
            ):
                out[k] = v.index_select(0, idx)
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


    def _maybe_adapt_limits(self):
        if not torch.cuda.is_available():
            return
        total = torch.cuda.get_device_properties(self.args.device.index).total_memory
        peak = torch.cuda.max_memory_allocated(self.args.device)
        util = peak / total

        # tune these
        high = 0.92
        low  = 0.80

        # initialize state
        if not hasattr(self, "_adapt_cooldown"):
            self._adapt_cooldown = 0
            self._stable_low_steps = 0

        if self._adapt_cooldown > 0:
            self._adapt_cooldown -= 1
            return

        if util > high:
            # back off fast
            if self.max_examples_per_microbatch is None:
                self.max_examples_per_microbatch = 16  # pick a sane start
            self.max_examples_per_microbatch = max(1, int(self.max_examples_per_microbatch * 0.7))
            self._stable_low_steps = 0
            self._adapt_cooldown = 10  # wait N steps before changing again

        elif util < low:
            self._stable_low_steps += 1
            if self._stable_low_steps >= 10:
                if self.max_examples_per_microbatch is None:
                    self.max_examples_per_microbatch = 16
                self.max_examples_per_microbatch = int(self.max_examples_per_microbatch * 1.15) + 1
                self._stable_low_steps = 0
                self._adapt_cooldown = 10
        else:
            self._stable_low_steps = 0

        if self.control.should_log:
            self.log({
                "adaptive_vram_util": float(util),
                "adaptive_max_B": float(self.max_examples_per_microbatch or 0),
            })

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

        return [self._compact_microbatch(self._slice_inputs(inputs, mb_idx)) for mb_idx in microbatches]

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
        # If caller wants logits/labels, let HF handle it
        if not prediction_loss_only or self.max_tokens_per_microbatch is None:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        model.eval()

        # ------------- CPU microbatch path -------------
        if self.use_cpu_microbatch:
            inputs_cpu = self._move_to_cpu(inputs)
            inputs_cpu = self._truncate_batch(inputs_cpu)

            microbatches = self._make_microbatches(
                inputs_cpu,
                max_tokens_per_microbatch=self.max_eval_tokens_per_microbatch,
            )
            if not microbatches:
                return (None, None, None)

            total_loss_tokens = 0.0
            total_tokens = 0

            for mb in microbatches:
                labels = mb.get("labels", None)
                if not (isinstance(labels, torch.Tensor) and labels.ndim == 2):
                    continue
                ntok = int((labels != -100).sum().item())
                if ntok == 0:
                    continue

                mb = self._to_device(mb)

                if self.length_column_name in mb:
                    mb = {k: v for k, v in mb.items() if k != self.length_column_name}

                with torch.no_grad():
                    with self.compute_loss_context_manager():
                        loss_mb = self.compute_loss(model, mb, return_outputs=False)

                if isinstance(loss_mb, tuple):
                    loss_mb = loss_mb[0]

                total_loss_tokens += float(loss_mb.detach().item()) * ntok
                total_tokens += ntok

            if total_tokens == 0:
                return (None, None, None)

            avg_loss = total_loss_tokens / total_tokens
            return (torch.tensor(avg_loss, device=self.args.device), None, None)

        # ------------- GPU-native microbatch path -------------
        inputs = self._prepare_inputs(inputs)
        inputs = self._truncate_batch(inputs)

        microbatches = self._make_microbatches(
            inputs,
            max_tokens_per_microbatch=self.max_eval_tokens_per_microbatch,
        )
        if not microbatches:
            return (None, None, None)

        total_loss_tokens = 0.0
        total_tokens = 0

        for mb in microbatches:
            labels = mb.get("labels", None)
            if not (isinstance(labels, torch.Tensor) and labels.ndim == 2):
                continue
            ntok = int((labels != -100).sum().item())
            if ntok == 0:
                continue

            if self.length_column_name in mb:
                mb = {k: v for k, v in mb.items() if k != self.length_column_name}

            with torch.no_grad():
                with self.compute_loss_context_manager():
                    loss_mb = self.compute_loss(model, mb, return_outputs=False)

            if isinstance(loss_mb, tuple):
                loss_mb = loss_mb[0]

            total_loss_tokens += float(loss_mb.detach().item()) * ntok
            total_tokens += ntok

        if total_tokens == 0:
            return (None, None, None)

        avg_loss = total_loss_tokens / total_tokens
        return (torch.tensor(avg_loss, device=self.args.device), None, None)

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If no token budget or we explicitly want HF-style eval, use default
        if self.max_tokens_per_batch is None or self.eval_mode == "hf":
            return super().get_eval_dataloader(eval_dataset)

        # Otherwise: token-aware eval DataLoader (length-bucketed)
        if hasattr(ds, "column_names"):
            raw_lengths = ds[self.length_column_name]
        else:
            raw_lengths = [ds[i][self.length_column_name] for i in range(len(ds))]

        if self.max_encoder_len is not None:
            lengths_for_sampler = [min(int(L), self.max_encoder_len) for L in raw_lengths]
        else:
            lengths_for_sampler = [int(L) for L in raw_lengths]

        batch_sampler = LengthBucketedBatchSampler(
            lengths=lengths_for_sampler,
            max_tokens_per_batch=self.max_tokens_per_batch,
            bucket_size=16,
            shuffle=False,
            drop_last=False,
            long_behavior="truncate",
            max_length_in_batch=self.max_encoder_len,  # strongly recommended for eval
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

    def _eval_oom_cleanup(self):
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _shrink_eval_limits(self) -> bool:
        """
        Shrink eval limits on OOM. Returns True if we changed something.
        Prefers shrinking eval token budget; optionally shrink B as secondary.
        """
        changed = False

        # shrink eval token budget first
        if self.max_eval_tokens_per_microbatch is not None and self.max_eval_tokens_per_microbatch > self.oom_min_tokens:
            new_T = max(self.oom_min_tokens, int(self.max_eval_tokens_per_microbatch * self.oom_shrink_tokens))
            if new_T < self.max_eval_tokens_per_microbatch:
                self.max_eval_tokens_per_microbatch = new_T
                changed = True

        # optional: also shrink microbatch examples if tokens can't shrink further
        if not changed:
            if self.max_examples_per_microbatch is not None and self.max_examples_per_microbatch > self.oom_min_B:
                new_B = max(self.oom_min_B, int(self.max_examples_per_microbatch * self.oom_shrink_B))
                if new_B < self.max_examples_per_microbatch:
                    self.max_examples_per_microbatch = new_B
                    changed = True

        return changed

    def _token_aware_evaluate(
        self,
        eval_dataset=None,
        max_eval_tokens_per_microbatch: int | None = None,
        desc: str = "Eval (token-aware)",
    ):
        import time, random, numpy as np, torch
        from tqdm.auto import tqdm

        py_state = random.getstate()
        np_state = np.random.get_state()

        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            try:
                # Deterministic eval RNG (inside fork_rng)
                torch.manual_seed(0)
                np.random.seed(0)
                random.seed(0)

                if eval_dataset is None:
                    eval_dataset = self.eval_dataset
                if eval_dataset is None:
                    raise ValueError("Need an eval_dataset for token-aware evaluation.")

                dataloader = self.get_eval_dataloader(eval_dataset)

                # --- loss-only path ---
                if getattr(self, "compute_metrics", None) is None:
                    total_eval_loss = 0.0
                    total_eval_tokens = 0
                    num_steps = 0
                    num_examples = 0
                    start_time = time.time()

                    for batch in dataloader:
                        num_steps += 1
                        labels = batch.get("labels", None)
                        if labels is None:
                            raise ValueError("Eval dataset must have labels to compute eval_loss.")

                        with torch.no_grad():
                            loss, _, _ = self.prediction_step(
                                self.model, batch, prediction_loss_only=True, ignore_keys=None
                            )

                        # NOTE: best is to use tokens actually used in prediction_step,
                        # but this is OK if prediction_step doesn't change labels.
                        num_tokens = int((labels.detach().to("cpu") != -100).sum().item())
                        if loss is not None and num_tokens > 0:
                            total_eval_loss += float(loss.item()) * num_tokens
                            total_eval_tokens += num_tokens

                        num_examples += int(labels.size(0))

                    runtime = time.time() - start_time
                    eval_loss = total_eval_loss / total_eval_tokens if total_eval_tokens > 0 else float("nan")

                    metrics = {
                        "eval_loss": float(eval_loss),
                        "eval_runtime": float(runtime),
                        "eval_samples_per_second": float(num_examples / runtime) if runtime > 0 else 0.0,
                        "eval_steps_per_second": float(num_steps / runtime) if runtime > 0 else 0.0,
                    }
                    return metrics
       
            finally:
                # fork_rng restores torch + cuda automatically
                np.random.set_state(np_state)
                random.setstate(py_state)

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
                batch_gpu = self._prepare_inputs(batch)

                ignore_keys = {
                    "labels",
                    self.length_column_name,
                    "decoder_input_ids",
                    "decoder_attention_mask",
                }
                # We will slice labels separately for metrics alignment
                labels_full = batch_gpu.get("labels", None)
                if labels_full is None:
                    raise ValueError("Eval dataset must have labels for metric computation.")

                # ---- OOM-resilient generation over MICROBATCHE(S) ----
                orig_use_cache = getattr(self.model.config, "use_cache", True)
                last_err = None

                # Always plan on CPU to avoid duplicating GPU memory in a list of microbatches
                batch_cpu = self._move_to_cpu(batch_gpu)
                batch_cpu = self._truncate_batch(batch_cpu)

                for attempt in range(self.oom_max_retries + 1):
                    self.model.config.use_cache = True
                    try:
                        # Replan microbatches each attempt
                        eval_microbatches_cpu = self._make_microbatches(
                            batch_cpu,
                            max_tokens_per_microbatch=self.max_eval_tokens_per_microbatch,
                        )

                        batch_pred_chunks = []
                        batch_label_chunks = []

                        for mb_cpu in eval_microbatches_cpu:
                            mb = self._to_device(mb_cpu)

                            gen_inputs_mb = {k: v for k, v in mb.items() if k not in ignore_keys}

                            with torch.no_grad():
                                gen_out = self.model.generate(**gen_inputs_mb, **gen_kwargs)

                            if gen_out.ndim == 1:
                                gen_out = gen_out.unsqueeze(0)

                            batch_pred_chunks.append(gen_out.cpu().numpy())
                            batch_label_chunks.append(mb["labels"].cpu().numpy())

                            # aggressively release GPU memory
                            del mb, gen_out, gen_inputs_mb
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        # success → commit
                        all_preds.extend(batch_pred_chunks)
                        all_labels.extend(batch_label_chunks)
                        last_err = None
                        break

                    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                        if not self._is_cuda_oom(e):
                            raise
                        last_err = e

                        self._eval_oom_cleanup()
                        changed = self._shrink_eval_limits()

                        if self.control.should_log:
                            self.log({
                                "eval_oom_event": 1.0,
                                "eval_oom_attempt": float(attempt),
                                "eval_max_T_now": float(self.max_eval_tokens_per_microbatch or 0),
                                "eval_max_B_now": float(self.max_examples_per_microbatch or 0),
                                "eval_changed_limits": float(changed),
                            })

                        if not changed:
                            break

                    finally:
                        self.model.config.use_cache = orig_use_cache

                if last_err is not None:
                    if getattr(self, "oom_skip_batch_on_fail", True):
                        if self.control.should_log:
                            self.log({"eval_oom_skipped_batch": 1.0})
                        continue
                    raise last_err

                self.model.config.use_cache = orig_use_cache

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
                pad_id = self.processing_class.pad_token_id
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

                compute_metrics_fn = getattr(self, "compute_metrics", None)
                if compute_metrics_fn is None:
                    # No metrics function provided: return loss-only metrics
                    raw_metrics = {"bleu": 0.0, "chrf": 0.0, "meteor": 0.0, "gen_len": 0.0}
                else:
                    raw_metrics = compute_metrics_fn((preds, labels))

                # (N, max_pred_len), (N, max_label_len)
                #raw_metrics = self.compute_metrics((preds, labels))
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

        print(f"\n[TokenPackTrainer] Saved emergency checkpoint to {ckpt_dir} "
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
          - "token_aware_metrics": use our custom generation+metrics
        """
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
            metrics = self._token_aware_evaluate(
                eval_dataset=eval_dataset,
                max_eval_tokens_per_microbatch=self.max_eval_tokens_per_microbatch,
                desc="eval (token-aware)",
            )
            runtime = time.time() - start_time

            # If _token_aware_evaluate didn't add runtime stats, add them
            metrics.setdefault("eval_runtime", float(runtime))

            # Attach epoch info (like HF)
            if self.state.epoch is not None:
                metrics[f"{metric_key_prefix}_epoch"] = self.state.epoch
                metrics["epoch"] = self.state.epoch

            # Log to integrations (wandb/tensorboard) + keep trainer_state.json consistent
            self.log(metrics)
            self.state.log_history.append(metrics)

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

    def _contiguous_inputs(self, inputs: dict) -> dict:
        out = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and not v.is_contiguous():
                out[k] = v.contiguous()
            else:
                out[k] = v
        return out

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.length_column_name in inputs:
            inputs = {k: v for k, v in inputs.items() if k != self.length_column_name}

        # 🔹 critical: view() in HF loss expects contiguous
        inputs = self._contiguous_inputs(inputs)

        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()

        # turn off KV cache during training to reduce memory spikes
        if hasattr(model, "config") and getattr(model.config, "use_cache", None) is True:
            model.config.use_cache = False

        last_err = None
        regime_key = None  # <<< keep across retries for this same HF batch

        for attempt in range(self.oom_max_retries + 1):
            try:
                # ---------------- PLAN MICROBATCHE(S) ----------------
                if self.use_cpu_microbatch:
                    inputs_work = self._move_to_cpu(inputs)

                    # >>> compute regime key once per batch (from CPU tensors)
                    if regime_key is None:
                        regime_key = self._regime_key_from_inputs(inputs_work)

                    # >>> apply per-regime best limits BEFORE planning microbatches
                    self._apply_regime_limits(regime_key)

                    _ = self._compute_lengths_enc_dec(inputs_work)
                    inputs_work = self._truncate_batch(inputs_work)
                    microbatches = self._make_microbatches(inputs_work)

                else:
                    # We don't have CPU tensors by default; easiest is to still derive a key on CPU
                    # from a detached copy. This is cheap because it's just masks/labels.
                    if regime_key is None:
                        tmp_cpu = self._move_to_cpu(inputs)
                        regime_key = self._regime_key_from_inputs(tmp_cpu)

                    # >>> apply per-regime best limits BEFORE planning microbatches
                    self._apply_regime_limits(regime_key)

                    inputs_work = self._prepare_inputs(inputs)
                    inputs_work = self._truncate_batch(inputs_work)
                    _ = self._compute_lengths_enc_dec(inputs_work)
                    microbatches = self._make_microbatches(inputs_work)

                num_micro = max(len(microbatches), 1)
                total_loss = 0.0
                total_examples = sum(mb["input_ids"].size(0) for mb in microbatches)

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                # ---------------- EXECUTE MICROBATCHE(S) ----------------
                total_tokens = 0
                mb_token_counts = []
                for mb in microbatches:
                    labels = mb.get("labels", None)
                    ntok = int((labels != -100).sum().item()) if (isinstance(labels, torch.Tensor) and labels.ndim == 2) else 0
                    mb_token_counts.append(ntok)
                    total_tokens += ntok

                total_loss_weighted = 0.0

                for mb, ntok in zip(microbatches, mb_token_counts):
                    if ntok == 0:
                        continue

                    if self.use_cpu_microbatch:
                        mb = self._to_device(mb)

                    if self.length_column_name in mb:
                        mb = {k: v for k, v in mb.items() if k != self.length_column_name}

                    with self.compute_loss_context_manager():
                        loss_mb = self.compute_loss(model, mb)   # mean over tokens in this mb

                    if isinstance(loss_mb, tuple):
                        loss_mb = loss_mb[0]

                    weight = ntok / max(total_tokens, 1)
                    loss = loss_mb * weight

                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps

                    self.accelerator.backward(loss)

                    total_loss_weighted += (loss_mb.detach().float() * weight)


                # >>> on success, mark this regime as stable and possibly ramp up
                if regime_key is not None:
                    self._regime_on_success(regime_key)

                # Return a sane scalar for logging
                return total_loss_weighted

                # optional logging
                if attempt > 0 and self.control.should_log:
                    self.log({
                        "oom_recovered": 1.0,
                        "oom_recovery_attempt": float(attempt),
                        "regime_key": float(regime_key if regime_key is not None else -1),
                        "max_B_now": float(self.max_examples_per_microbatch or 0),
                        "max_tokens_now": float(self.max_tokens_per_microbatch or 0),
                    })

                return avg_loss

            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if not self._is_cuda_oom(e):
                    raise

                last_err = e
                self._oom_events += 1

                # cleanup
                self._oom_cleanup()

                # >>> shrink only THIS regime first
                if regime_key is not None:
                    self._regime_on_oom(regime_key)
                    # re-apply immediately so the retry uses the new values
                    self._apply_regime_limits(regime_key)
                    changed = True
                else:
                    # fallback to global shrink if key somehow missing
                    changed = self._oom_shrink_limits()

                if self.control.should_log:
                    self.log({
                        "oom_event": 1.0,
                        "oom_attempt": float(attempt),
                        "regime_key": float(regime_key if regime_key is not None else -1),
                        "oom_changed_limits": float(1.0 if changed else 0.0),
                        "max_B_now": float(self.max_examples_per_microbatch or 0),
                        "max_tokens_now": float(self.max_tokens_per_microbatch or 0),
                    })

                if not changed:
                    break

                continue

        # ---------------- OUT OF RETRIES ----------------
        if self.oom_skip_batch_on_fail:
            self._oom_skipped_batches += 1
            if self.control.should_log:
                self.log({
                    "oom_skipped_batch": 1.0,
                    "oom_skipped_batches_total": float(self._oom_skipped_batches),
                    "regime_key": float(regime_key if regime_key is not None else -1),
                })

            return torch.tensor(0.0, device=self.args.device)

        raise last_err
