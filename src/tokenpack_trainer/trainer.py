"""
TokenPack Trainer - Token-aware batch processing for efficient seq2seq training.

This module provides TokenPackTrainer, a drop-in replacement for HuggingFace's
Seq2SeqTrainer that uses token-based batching instead of fixed batch sizes.

Key Concepts:
-------------
1. Token Budgeting: Instead of batch_size × max_seq_length, batches are formed
   by sum(tokens_in_batch) ≤ token_budget. This eliminates padding waste for
   variable-length sequences.

2. Two-Level Budgeting:
   - Batch level: max_tokens_per_batch controls the sampler (how many examples
     are grouped together from the dataset)
   - Microbatch level: max_tokens_per_microbatch controls how examples within
     a batch are split for GPU memory management

3. Adaptive OOM Handling: The trainer learns stable microbatch sizes per "regime"
   (groups of similar sequence lengths) and automatically adjusts on OOM.

4. Gradient Equivalence: Microbatch gradients are accumulated to produce
   mathematically equivalent gradients to full-batch training.

Architecture:
-------------
    Dataset → LengthBucketedBatchSampler → DataLoader → TokenPackTrainer
                     ↓                                         ↓
              Groups by length                          Splits into microbatches
              Packs to token budget                     Handles OOM adaptively

Example:
--------
    >>> from tokenpack_trainer import TokenPackTrainer
    >>> trainer = TokenPackTrainer(
    ...     model=model,
    ...     args=training_args,
    ...     train_dataset=dataset,
    ...     max_tokens_per_batch=16384,      # Sampler budget
    ...     max_tokens_per_microbatch=4096,  # GPU memory budget
    ...     max_encoder_len=512,
    ...     max_decoder_len=128,
    ... )
"""

import os
import time
from typing import Any, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer
from transformers.utils import logging

from .samplers import LengthBucketedBatchSampler

logger = logging.get_logger(__name__)

class CUDAPrefetcher:
    """
    Wrap a DataLoader to asynchronously move each batch to GPU on a side stream.

    IMPORTANT for HF Trainer: preserve __len__ and key attributes, otherwise
    Trainer can't infer max_steps.
    """
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def __len__(self):
        # HF Trainer relies on len(dataloader) for step math
        if hasattr(self.loader, "__len__"):
            return len(self.loader)
        raise TypeError("Underlying loader has no __len__; set args.max_steps > 0")

    def __getattr__(self, name):
        # Forward common attributes (dataset, batch_sampler, etc.)
        return getattr(self.loader, name)

    def __iter__(self):
        if self.stream is None:
            yield from self.loader
            return

        it = iter(self.loader)

        def _to_device(batch):
            out = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    out[k] = v.to(self.device, non_blocking=True)
                else:
                    out[k] = v
            return out

        with torch.cuda.stream(self.stream):
            next_batch = next(it, None)
            if next_batch is not None:
                next_batch = _to_device(next_batch)

        while True:
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = next_batch
            if batch is None:
                break

            with torch.cuda.stream(self.stream):
                next_batch = next(it, None)
                if next_batch is not None:
                    next_batch = _to_device(next_batch)

            yield batch

class TokenPackTrainer(Seq2SeqTrainer):
    """
    Token-aware Seq2Seq trainer with adaptive microbatching and OOM recovery.

    This trainer extends HuggingFace's Seq2SeqTrainer to handle variable-length
    sequences efficiently by:

    1. Packing batches by token count instead of example count
    2. Splitting batches into microbatches that fit GPU memory
    3. Learning optimal microbatch sizes per sequence length regime
    4. Automatically recovering from OOM errors

    Key Parameters:
    ---------------
    max_tokens_per_batch : int
        Maximum total tokens per batch for the dataloader sampler.
        Controls how many examples are grouped together. Default: None (uses HF default).

    max_tokens_per_microbatch : int
        Maximum tokens per microbatch for GPU processing. Smaller = less memory,
        more microbatches. Start with ~4096, increase if no OOMs. Default: 400.

    max_encoder_len / max_decoder_len : int, optional
        Hard truncation limits for encoder/decoder sequences.

    use_cpu_microbatch : bool
        If True (default), keep batches on CPU and transfer microbatches to GPU.
        Uses prefetching to overlap transfer with computation.
        If False, use standard HF device placement.

    eval_mode : str, optional
        "hf" or None: Use standard HF evaluation (faster, no generation).
        "token_aware_metrics": Use token-aware eval with generation for BLEU/chrF.

    Adaptive OOM Parameters:
    ------------------------
    oom_max_retries : int
        Max retry attempts on OOM before skipping batch. Default: 3.

    oom_shrink_B : float
        Factor to shrink batch size on OOM. Default: 0.5.

    oom_shrink_tokens : float
        Factor to shrink token budget on OOM. Default: 0.85.

    Debugging:
    ----------
    Use get_regime_stats() to inspect learned adaptive limits:
        >>> stats = trainer.get_regime_stats()
        >>> print(stats['regimes'])  # Per-regime B/T limits
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
        eval_data_collator=None,
        sampler_bucket_size: int = 8,  # bucket size for length-bucketed sampling (smaller = tighter grouping)
        padding_aware_budget: bool = False,  # if True, budget by max_len * num_examples (actual memory) vs sum of lengths
        max_eval_generate_examples: int | None = None,  # max examples per generate call (None = no limit, uses token budget only)
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eval_data_collator = eval_data_collator
        self.padding_aware_budget = bool(padding_aware_budget)
        self.max_eval_generate_examples = max_eval_generate_examples  # None = use full batch when possible
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
        self.eval_mode = self._normalize_eval_mode(eval_mode)
        self.debug = debug
        self.oom_max_retries = int(oom_max_retries)
        self.oom_shrink_B = float(oom_shrink_B)
        self.oom_shrink_tokens = float(oom_shrink_tokens)
        self.oom_min_B = int(oom_min_B)
        self.oom_min_tokens = int(oom_min_tokens)
        self.oom_skip_batch_on_fail = bool(oom_skip_batch_on_fail)
        self.sampler_bucket_size = int(sampler_bucket_size)

        # --- OOM tracking ---
        self._oom_events = 0           # Total OOM events across all batches
        self._oom_skipped_batches = 0  # Batches skipped after exhausting retries

        # --- Adaptive Regime Management ---
        # Regimes group sequences by length to learn stable microbatch limits.
        # Key = bucket of max effective length, Value = {"B": batch_size, "T": token_budget, "stable": success_count}
        self._regime_limits = {}
        self._regime_bucket_size = 128   # Bucket width in tokens (groups similar lengths)
        self._regime_ramp_every = 10     # Ramp limits after N consecutive successes
        self._regime_ramp_B = 1.25       # Factor to increase B on ramp
        self._regime_ramp_T = 1.10       # Factor to increase T on ramp
        self._regime_min_B = getattr(self, "oom_min_B", 1)
        self._regime_min_T = getattr(self, "oom_min_tokens", 64)

        # Initial values for new regimes (from user config)
        self._regime_default_B = self.max_examples_per_microbatch
        self._regime_default_T = self.max_tokens_per_microbatch

        # --- Sequence length tracking for diagnostics ---
        self._max_seen_enc_len = 0
        self._max_seen_dec_len = 0
        self._max_seen_total_len = 0
        self._num_trunc_hits = 0  # Count of sequences that hit truncation limits

        # Upper bounds to prevent runaway growth during adaptation
        self._regime_max_T = int(max_tokens_per_microbatch * 4) if max_tokens_per_microbatch is not None else (1 << 62)
        self._regime_max_B = 1024



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

        # Validate parameter combinations
        self._validate_params()

    def _validate_params(self):
        """Validate parameter combinations and fail fast on invalid configs."""
        errors = []

        # Token budget sanity checks
        if self.max_tokens_per_microbatch is not None and self.max_tokens_per_microbatch <= 0:
            errors.append(f"max_tokens_per_microbatch must be positive, got {self.max_tokens_per_microbatch}")

        if self.max_tokens_per_batch is not None and self.max_tokens_per_batch <= 0:
            errors.append(f"max_tokens_per_batch must be positive, got {self.max_tokens_per_batch}")

        if self.max_eval_tokens_per_microbatch is not None and self.max_eval_tokens_per_microbatch <= 0:
            errors.append(f"max_eval_tokens_per_microbatch must be positive, got {self.max_eval_tokens_per_microbatch}")

        # Microbatch should not exceed batch budget
        if (self.max_tokens_per_batch is not None
            and self.max_tokens_per_microbatch is not None
            and self.max_tokens_per_microbatch > self.max_tokens_per_batch):
            errors.append(
                f"max_tokens_per_microbatch ({self.max_tokens_per_microbatch}) should not exceed "
                f"max_tokens_per_batch ({self.max_tokens_per_batch})"
            )

        # Length caps should be positive if set
        if self.max_encoder_len is not None and self.max_encoder_len <= 0:
            errors.append(f"max_encoder_len must be positive, got {self.max_encoder_len}")

        if self.max_decoder_len is not None and self.max_decoder_len <= 0:
            errors.append(f"max_decoder_len must be positive, got {self.max_decoder_len}")

        # Sampler bucket size
        if self.sampler_bucket_size <= 0:
            errors.append(f"sampler_bucket_size must be positive, got {self.sampler_bucket_size}")

        # OOM parameters
        if self.oom_max_retries < 0:
            errors.append(f"oom_max_retries must be non-negative, got {self.oom_max_retries}")

        if not (0 < self.oom_shrink_B <= 1):
            errors.append(f"oom_shrink_B must be in (0, 1], got {self.oom_shrink_B}")

        if not (0 < self.oom_shrink_tokens <= 1):
            errors.append(f"oom_shrink_tokens must be in (0, 1], got {self.oom_shrink_tokens}")

        if self.oom_min_B < 1:
            errors.append(f"oom_min_B must be >= 1, got {self.oom_min_B}")

        if self.oom_min_tokens < 1:
            errors.append(f"oom_min_tokens must be >= 1, got {self.oom_min_tokens}")

        # max_examples_per_microbatch if set
        if self.max_examples_per_microbatch is not None and self.max_examples_per_microbatch <= 0:
            errors.append(f"max_examples_per_microbatch must be positive, got {self.max_examples_per_microbatch}")

        if errors:
            raise ValueError(
                "Invalid TokenPackTrainer configuration:\n  - " + "\n  - ".join(errors)
            )

    def _normalize_eval_mode(self, mode: str | None) -> str | None:
        if mode is None:
            return None
        m = str(mode).lower()
        if m in ("none", "hf", "vanilla", "huggingface"):
            return None
        if m in ("token_aware", "token-aware", "token_aware_metrics"):
            return "token_aware_metrics"
        return m

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
        INT64_MAX = (1 << 63) - 1

        T = int(st["T"])
        if hasattr(self, "_regime_max_T") and self._regime_max_T is not None:
            T = min(T, int(self._regime_max_T))
        T = min(T, INT64_MAX)

        self.max_examples_per_microbatch = st["B"]
        self.max_tokens_per_microbatch = T
        # keep eval budget <= train budget
        if getattr(self, "max_eval_tokens_per_microbatch", None) is not None:
            self.max_eval_tokens_per_microbatch = min(self.max_eval_tokens_per_microbatch, self.max_tokens_per_microbatch)

    INT64_MAX = (1 << 63) - 1

    @staticmethod
    def _clamp_int(x: int, lo: int, hi: int) -> int:
        return max(int(lo), min(int(x), int(hi)))

    def _regime_on_success(self, key: int):
        st = self._regime_state(key)
        st["stable"] += 1

        if st["stable"] % int(self._regime_ramp_every) == 0:
            if st["B"] is None:
                st["B"] = 16
            st["B"] = max(self._regime_min_B, int(st["B"] * float(self._regime_ramp_B)) + 1)
            st["B"] = self._clamp_int(st["B"], self._regime_min_B, self._regime_max_B)

            st["T"] = max(self._regime_min_T, int(st["T"] * float(self._regime_ramp_T)))
            # hard clamp to practical + int64 safety
            st["T"] = self._clamp_int(st["T"], self._regime_min_T, min(self._regime_max_T, self.INT64_MAX))

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

    def get_regime_stats(self) -> dict:
        """
        Return current adaptive regime limits for debugging and monitoring.

        Returns a dict with:
            - regimes: dict mapping regime_key → {"B": int, "T": int, "stable": int}
            - current_max_B: current max_examples_per_microbatch
            - current_max_T: current max_tokens_per_microbatch
            - oom_events: total OOM events encountered
            - oom_skipped_batches: batches skipped due to unrecoverable OOM

        Example:
            >>> stats = trainer.get_regime_stats()
            >>> print(f"OOM events: {stats['oom_events']}")
            >>> for key, limits in stats['regimes'].items():
            ...     print(f"  Regime {key}: B={limits['B']}, T={limits['T']}, stable={limits['stable']}")
        """
        return {
            "regimes": dict(self._regime_limits),
            "current_max_B": self.max_examples_per_microbatch,
            "current_max_T": self.max_tokens_per_microbatch,
            "current_eval_max_T": getattr(self, "max_eval_tokens_per_microbatch", None),
            "oom_events": getattr(self, "_oom_events", 0),
            "oom_skipped_batches": getattr(self, "_oom_skipped_batches", 0),
            "regime_bucket_size": getattr(self, "_regime_bucket_size", None),
        }

    def _regime_key_from_batch(self, inputs_cpu) -> int:
        enc_len, dec_len, _ = self._compute_lengths_enc_dec(inputs_cpu)
        alpha = 2.0
        eff = enc_len + alpha * dec_len
        mx = int(eff.max().item())
        bucket_size = 128   # choose something coarse
        return int((mx + bucket_size - 1) // bucket_size)


    def _autotune_regime_from_peak(self, key: int, eff_tokens_in_step: int, safety: float = 0.90):
        if not torch.cuda.is_available() or eff_tokens_in_step <= 0:
            return

        total = torch.cuda.get_device_properties(self.args.device.index).total_memory
        peak  = torch.cuda.max_memory_allocated(self.args.device)

        # avoid divide-by-zero / nonsense
        if peak <= 0:
            return

        bytes_per_eff_token = peak / float(eff_tokens_in_step)

        # target peak = safety * total
        target_peak = safety * total
        target_eff_tokens = int(target_peak / max(bytes_per_eff_token, 1e-9))

        # clamp hard
        INT64_MAX = (1 << 63) - 1
        st = self._regime_state(key)
        st["T"] = max(self._regime_min_T, min(target_eff_tokens, INT64_MAX, getattr(self, "_regime_max_T", INT64_MAX)))

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

        # ---- build the DataLoader exactly as before ----
        if self.max_tokens_per_batch is None:
            dl = super().get_train_dataloader()
        else:
            ds = self.train_dataset

            if hasattr(ds, "column_names"):
                keep_cols = {
                    "input_ids",
                    "attention_mask",
                    "labels",
                    "decoder_input_ids",
                    "decoder_attention_mask",
                    self.length_column_name,
                }
                to_remove = [c for c in ds.column_names if c not in keep_cols]
                if to_remove:
                    ds = ds.remove_columns(to_remove)

            if hasattr(ds, "column_names"):
                raw_lengths = ds[self.length_column_name]
            else:
                raw_lengths = [ds[i][self.length_column_name] for i in range(len(ds))]

            if self.max_encoder_len is not None:
                lengths_for_sampler = [
                    min(int(L), self.max_encoder_len) for L in raw_lengths
                ]
            else:
                lengths_for_sampler = [int(L) for L in raw_lengths]

            batch_sampler = LengthBucketedBatchSampler(
                lengths=lengths_for_sampler,
                max_tokens_per_batch=self.max_tokens_per_batch,
                bucket_size=self.sampler_bucket_size,
                shuffle=True,
                drop_last=False,
                long_behavior="truncate",
                max_length_in_batch=self.max_encoder_len,
            )

            dl = DataLoader(
                ds,
                batch_sampler=batch_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                persistent_workers=self.args.dataloader_persistent_workers,
                prefetch_factor=getattr(self.args, "dataloader_prefetch_factor", 2),
            )

        # ---- OPTIONAL GPU prefetch wrapper ----
        if torch.cuda.is_available() and not self.use_cpu_microbatch:
            dl = CUDAPrefetcher(dl, self.args.device)

        return dl

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
        
        T = self.max_tokens_per_microbatch
        if T is not None and (not isinstance(T, int) or T <= 0 or T > (1 << 62)):
            raise RuntimeError(f"max_tokens_per_microbatch is insane: {T}")
            
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
        device = self.args.device
        out = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                kwargs = {"device": device, "non_blocking": True}
                if self.is_deepspeed_enabled and (torch.is_floating_point(v) or torch.is_complex(v)):
                    kwargs["dtype"] = self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()
                out[k] = v.to(**kwargs)
            else:
                out[k] = v
        return out

    def _to_device_on_stream(self, inputs: dict, stream: torch.cuda.Stream) -> dict:
        """Transfer inputs to GPU on a specific CUDA stream."""
        device = self.args.device
        out = {}
        with torch.cuda.stream(stream):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    kwargs = {"device": device, "non_blocking": True}
                    if self.is_deepspeed_enabled and (torch.is_floating_point(v) or torch.is_complex(v)):
                        kwargs["dtype"] = self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()
                    out[k] = v.to(**kwargs)
                else:
                    out[k] = v
        return out

    def _prefetch_microbatches(self, microbatches: list):
        """
        Iterator that prefetches next microbatch to GPU while current one is processed.
        Overlaps CPU→GPU transfer with computation for ~10-20% speedup.
        """
        if not microbatches:
            return

        # If not using CPU microbatch mode or CUDA unavailable, just yield directly
        if not self.use_cpu_microbatch or not torch.cuda.is_available():
            for mb in microbatches:
                yield mb
            return

        # Create a dedicated stream for data transfer
        transfer_stream = torch.cuda.Stream()

        # Start transfer of first microbatch
        next_mb = self._to_device_on_stream(microbatches[0], transfer_stream)

        for i in range(len(microbatches)):
            # Wait for current microbatch transfer to complete
            torch.cuda.current_stream().wait_stream(transfer_stream)
            current_mb = next_mb

            # Start prefetching next microbatch (if any) while current is processed
            if i + 1 < len(microbatches):
                next_mb = self._to_device_on_stream(microbatches[i + 1], transfer_stream)

            yield current_mb

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


    def _make_microbatches(self, inputs, max_tokens_per_microbatch: int | None = None, return_lengths: bool = False):
        # Compute lengths
        enc_len, dec_len, _ = self._compute_lengths_enc_dec(inputs)

        alpha = 2.0  # decoder weight
        effective_len = enc_len + alpha * dec_len
        lengths = [int(L) for L in effective_len.tolist()]
        N = len(lengths)

        budget = int(max_tokens_per_microbatch or self.max_tokens_per_microbatch)
        max_B = self.max_examples_per_microbatch  # may be None
        padding_aware = getattr(self, "padding_aware_budget", False)

        microbatches = []
        cur_indices: list[int] = []
        cur_tokens = 0
        cur_max_len = 0  # for padding-aware mode

        # Sort by length to minimize padding waste within microbatches
        sorted_indices = sorted(range(N), key=lambda i: lengths[i])

        for i in sorted_indices:
            L = lengths[i]

            if padding_aware:
                # Padding-aware: cost is max_len * num_examples (actual tensor size)
                new_max = max(cur_max_len, L)
                new_count = len(cur_indices) + 1
                padded_cost = new_max * new_count
                too_many_tokens = cur_indices and padded_cost > budget
            else:
                # Default: cost is sum of effective lengths
                too_many_tokens = cur_indices and (cur_tokens + L) > budget

            too_many_examples = max_B is not None and len(cur_indices) >= max_B

            if too_many_tokens or too_many_examples:
                microbatches.append(cur_indices)
                cur_indices = [i]
                cur_tokens = L
                cur_max_len = L
            else:
                cur_indices.append(i)
                cur_tokens += L
                cur_max_len = max(cur_max_len, L)

        if cur_indices:
            microbatches.append(cur_indices)

        result = [self._compact_microbatch(self._slice_inputs(inputs, mb_idx)) for mb_idx in microbatches]

        if return_lengths:
            return result, enc_len, dec_len
        return result

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
        # Also use HF default when eval_mode is None/"hf" to avoid loss normalization mismatch
        # (HF aggregates losses by example count, not token count)
        eval_mode = getattr(self, "eval_mode", None)
        use_hf_default = (
            not prediction_loss_only
            or self.max_tokens_per_microbatch is None
            or eval_mode is None  # HF mode - let HF handle loss aggregation
        )
        if use_hf_default:
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
        mode = self._normalize_eval_mode(getattr(self, "eval_mode", None))
        if mode is None or self.max_tokens_per_batch is None:
            return super().get_eval_dataloader(eval_dataset)

        # Otherwise: token-aware eval DataLoader (length-bucketed)
        if hasattr(eval_dataset, "column_names"):
            raw_lengths = eval_dataset[self.length_column_name]
        else:
            raw_lengths = [eval_dataset[i][self.length_column_name] for i in range(len(eval_dataset))]

        if self.max_encoder_len is not None:
            lengths_for_sampler = [min(int(L), self.max_encoder_len) for L in raw_lengths]
        else:
            lengths_for_sampler = [int(L) for L in raw_lengths]

        batch_sampler = LengthBucketedBatchSampler(
            lengths=lengths_for_sampler,
            max_tokens_per_batch=self.max_tokens_per_batch,
            bucket_size=self.sampler_bucket_size,
            shuffle=False,
            drop_last=False,
            long_behavior="truncate",
            max_length_in_batch=self.max_encoder_len,
        )
        
        return DataLoader(
            eval_dataset,
            batch_sampler=batch_sampler,
            collate_fn=(self.eval_data_collator or self.data_collator),
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

    def _eval_regime_init(self):
        if not hasattr(self, "_eval_stable"):
            self._eval_stable = 0

    def _eval_on_success(self):
        """
        Called once per *successful eval batch* (i.e., the batch generated without OOM).
        Ramps up token budget slowly after enough stable batches.
        """
        self._eval_regime_init()
        self._eval_stable += 1

        # only ramp if we have a budget
        if self.max_eval_tokens_per_microbatch is None:
            return

        ramp_every = 50
        ramp_T = 1.05  # +5%
        max_T_cap = getattr(self, "max_tokens_per_microbatch", None)  # never exceed train budget if set

        if (self._eval_stable % ramp_every) == 0:
            new_T = int(self.max_eval_tokens_per_microbatch * ramp_T)
            if max_T_cap is not None:
                new_T = min(new_T, int(max_T_cap))
            self.max_eval_tokens_per_microbatch = max(new_T, self.oom_min_tokens)

    def _eval_on_oom(self):
        """Reset stability counter on OOM so we don't ramp immediately after shrinking."""
        self._eval_regime_init()
        self._eval_stable = 0

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

                    for batch in tqdm(dataloader, desc=desc, leave=False):
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


                ignore_keys = {
                    "labels",
                    self.length_column_name,
                    "decoder_input_ids",
                    "decoder_attention_mask",
                }
                
                # 2) Now prepare inputs for generation (separate from loss)
                # Prepare once
                batch_gpu = self._prepare_inputs(batch)

               # FAST PATH: if everything fits comfortably, generate once for the whole batch
                # (avoids microbatch splitting overhead - critical for throughput)
                # For generation, we can be more aggressive than training since no gradients
                B = int(batch_gpu["input_ids"].size(0))
                max_gen_examples = getattr(self, "max_eval_generate_examples", None)
                fits_B = (max_gen_examples is None) or (B <= max_gen_examples)

                if fits_B and "attention_mask" in batch_gpu and "labels" in batch_gpu:
                    # Single generate call for the whole batch
                    gen_inputs = {k: v for k, v in batch_gpu.items() if k not in ignore_keys}
                    try:
                        gen_out = self.model.generate(**gen_inputs, **gen_kwargs)
                        all_preds.append(gen_out.detach().cpu().numpy())
                        all_labels.append(batch_gpu["labels"].detach().cpu().numpy())
                        self._eval_on_success()
                        continue
                    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                        if not self._is_cuda_oom(e):
                            raise
                        # OOM on fast path - fall through to microbatching
                        self._eval_oom_cleanup()
                        if max_gen_examples is None:
                            self.max_eval_generate_examples = B // 2  # adaptive shrink
                        else:
                            self.max_eval_generate_examples = max(1, max_gen_examples // 2)
  
                if self.use_cpu_microbatch:
                    plan_inputs = self._truncate_batch(self._move_to_cpu(batch_gpu))
                else:
                    plan_inputs = self._truncate_batch(batch_gpu)

                last_err = None

                for attempt in range(self.oom_max_retries + 1):
                    try:
                        # (optional but recommended) enable cache for generation
                        orig_use_cache = getattr(self.model.config, "use_cache", True)
                        self.model.config.use_cache = True

                        # For generation, use larger batches than training (no gradients)
                        orig_max_B = self.max_examples_per_microbatch
                        gen_limit = getattr(self, "max_eval_generate_examples", None)
                        if gen_limit is not None:
                            self.max_examples_per_microbatch = gen_limit

                        microbatches = self._make_microbatches(
                            plan_inputs,
                            max_tokens_per_microbatch=self.max_eval_tokens_per_microbatch,
                        )
                        self.max_examples_per_microbatch = orig_max_B  # restore

                        # IMPORTANT: reset per attempt (fixes double-append after partial OOM)
                        batch_pred_chunks = []
                        batch_label_chunks = []

                        for mb in microbatches:
                            if self.use_cpu_microbatch:
                                mb = self._to_device(mb)  # CPU→GPU only in this mode

                            gen_inputs_mb = {k: v for k, v in mb.items() if k not in ignore_keys}
                            gen_out = self.model.generate(**gen_inputs_mb, **gen_kwargs)

                            if gen_out.ndim == 1:
                                gen_out = gen_out.unsqueeze(0)

                            batch_pred_chunks.append(gen_out.cpu().numpy())
                            batch_label_chunks.append(mb["labels"].cpu().numpy())

                            del gen_out, gen_inputs_mb, mb

                        # success → commit
                        all_preds.extend(batch_pred_chunks)
                        all_labels.extend(batch_label_chunks)
                        last_err = None

                        # ADAPTIVE RAMP (place it here: only after a successful batch)
                        self._eval_on_success()

                        break

                    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                        if not self._is_cuda_oom(e):
                            raise
                        last_err = e

                        self._eval_on_oom()           # reset stability counter
                        self._eval_oom_cleanup()
                        changed = self._shrink_eval_limits()

                        if self.control.should_log:
                            self.log({
                                "eval_oom_event": 1.0,
                                "eval_oom_attempt": float(attempt),
                                "eval_max_T_now": float(self.max_eval_tokens_per_microbatch or 0),
                                "eval_max_B_now": float(self.max_examples_per_microbatch or 0),
                                "eval_changed_limits": float(changed),
                                "eval_stable": float(getattr(self, "_eval_stable", 0)),
                            })

                        if not changed:
                            break

                    finally:
                        # restore cache flag
                        try:
                            self.model.config.use_cache = orig_use_cache
                        except Exception:
                            pass

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


    def _token_aware_loss_only(self, eval_dataset, desc="Eval (token-weighted loss)"):
        from tqdm.auto import tqdm
        import time, torch

        dl = self.get_eval_dataloader(eval_dataset)
        total_loss_tokens = 0.0
        total_tokens = 0
        num_steps = 0
        num_examples = 0
        t0 = time.time()

        for batch in tqdm(dl, desc=desc, leave=False):
            num_steps += 1
            labels = batch.get("labels", None)
            if labels is None:
                continue

            with torch.no_grad():
                loss, _, _ = self.prediction_step(self.model, batch, prediction_loss_only=True)

            # IMPORTANT: count decoder tokens for THIS BATCH (not per microbatch)
            # (prediction_step already token-weights within a batch)
            ntok = int((labels.detach().to("cpu") != -100).sum().item())
            if loss is not None and ntok > 0:
                total_loss_tokens += float(loss.item()) * ntok
                total_tokens += ntok

            num_examples += int(labels.size(0))

        runtime = time.time() - t0
        eval_loss = total_loss_tokens / total_tokens if total_tokens > 0 else float("nan")

        return {
            "eval_loss": float(eval_loss),
            "eval_runtime": float(runtime),
            "eval_samples_per_second": float(num_examples / runtime) if runtime > 0 else 0.0,
            "eval_steps_per_second": float(num_steps / runtime) if runtime > 0 else 0.0,
        }

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
        eval_mode: str | None = None,
    ):
        """
        eval_mode:
          - None / "hf" / "vanilla" / "huggingface": use standard HF evaluation_loop
          - "token_aware_loss": token-weighted loss-only (no preds)
          - "token_aware_metrics": token-aware generation/preds + compute_metrics
          - "auto": (optional) use token-aware loss-only when no metrics/generate, otherwise HF
        """
        eval_dataset = eval_dataset or self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Evaluation requires an eval_dataset.")

        requested = eval_mode if eval_mode is not None else getattr(self, "eval_mode", None)
        mode = self._normalize_eval_mode(requested)

        wants_generate = bool(getattr(self.args, "predict_with_generate", False))
        has_metrics = getattr(self, "compute_metrics", None) is not None

        # ----------------------------
        # Explicit token-aware modes
        # ----------------------------
        if mode == "token_aware_loss":
            if wants_generate or has_metrics:
                raise ValueError(
                    "eval_mode='token_aware_loss' is loss-only. "
                    "Set predict_with_generate=False and compute_metrics=None "
                    "or use eval_mode='token_aware_metrics'."
                )
            return self._token_aware_loss_only(eval_dataset, desc="Eval (token-weighted loss)")

        if mode == "token_aware_metrics":
            if not has_metrics:
                raise ValueError(
                    "eval_mode='token_aware_metrics' requires compute_metrics to be set. "
                    "Use eval_mode='token_aware_loss' for loss-only."
                )
            return self._token_aware_evaluate(
                eval_dataset=eval_dataset,
                max_eval_tokens_per_microbatch=self.max_eval_tokens_per_microbatch,
                desc="eval (token-aware)",
            )

        # ----------------------------
        # 2) Auto mode
        # ----------------------------
        if mode == "auto":
            if (not wants_generate) and (not has_metrics):
                return self._token_aware_loss_only(eval_dataset, desc="Eval (token-weighted loss)")
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        # ----------------------------
        # 3) Pure HF behavior
        # ----------------------------
        return super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

    def _contiguous_inputs(self, inputs: dict) -> dict:
        out = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and not v.is_contiguous():
                out[k] = v.contiguous()
            else:
                out[k] = v
        return out

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        Compute loss, filtering out length column and ensuring contiguous tensors.

        Note: num_items_in_batch is accepted for compatibility with transformers>=4.46
        but not used (we handle loss normalization via microbatch token weighting).
        """
        if self.length_column_name in inputs:
            inputs = {k: v for k, v in inputs.items() if k != self.length_column_name}

        # HF loss computation uses view() which requires contiguous tensors
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

                    inputs_work = self._truncate_batch(inputs_work)
                    microbatches, enc_len, dec_len = self._make_microbatches(inputs_work, return_lengths=True)

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
                    microbatches, enc_len, dec_len = self._make_microbatches(inputs_work, return_lengths=True)

                eff_tokens = int((enc_len + 2.0 * dec_len).sum().item())

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

                # Use prefetching iterator to overlap CPU→GPU transfer with computation
                prefetched = self._prefetch_microbatches(microbatches) if self.use_cpu_microbatch else iter(microbatches)

                for mb, ntok in zip(prefetched, mb_token_counts):
                    if ntok == 0:
                        continue

                    # Note: mb is already on GPU if use_cpu_microbatch (via prefetching)

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

                if regime_key is not None and not getattr(self, "_autotuned_keys", set()).__contains__(regime_key):
                    self._autotune_regime_from_peak(regime_key, eff_tokens, safety=0.90)
                    self._autotuned_keys = getattr(self, "_autotuned_keys", set())
                    self._autotuned_keys.add(regime_key)


                # optional logging
                if attempt > 0 and self.control.should_log:
                    self.log({
                        "oom_recovered": 1.0,
                        "oom_recovery_attempt": float(attempt),
                        "regime_key": float(regime_key if regime_key is not None else -1),
                        "max_B_now": float(self.max_examples_per_microbatch or 0),
                        "max_tokens_now": float(self.max_tokens_per_microbatch or 0),
                    })

                # Return a sane scalar for logging
                return total_loss_weighted



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
