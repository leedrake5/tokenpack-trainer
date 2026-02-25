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

import torch
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer
from transformers.utils import logging

from .samplers import LengthBucketedBatchSampler

logger = logging.get_logger(__name__)

import inspect
import torch.nn.functional as F

_ALWAYS_ALLOW = {
    "input_ids", "attention_mask", "inputs_embeds",
    "decoder_input_ids", "decoder_attention_mask",
    "encoder_outputs", "past_key_values",
    "use_cache", "head_mask", "decoder_head_mask", "cross_attn_head_mask",
    "output_attentions", "output_hidden_states", "return_dict",
}

def _pad_to_len(x: torch.Tensor, pad: int, L: int) -> torch.Tensor:
    if x.size(1) == L:
        return x
    return F.pad(x, (0, L - x.size(1)), value=pad)

def _is_tensorish(x) -> bool:
    return torch.is_tensor(x)


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
        sampler_shuffle_mode: str = "interleave",  # "bucket" = sequential buckets, "interleave" = global batch shuffle across buckets
        padding_aware_budget: bool = False,  # if True, budget by max_len * num_examples (actual memory) vs sum of lengths
        regime_max_B: int | None = None,  # max examples per microbatch the regime can ramp to (None = auto from token budget)
        max_eval_generate_examples: int | None = None,  # max examples per generate call (None = no limit, uses token budget only)
        builtin_metrics: tuple[str, ...] | None = None,   # e.g. ("bleu","chrf") or ("bleu","chrf","meteor")
        builtin_metrics_tokenize: str = "13a",
        builtin_metrics_lowercase: bool = False,
        max_metric_samples: int | None = 2000,  # Max samples for metric computation (None = all, default 2000 for speed)
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eval_data_collator = eval_data_collator
        self.padding_aware_budget = bool(padding_aware_budget)
        self.max_eval_generate_examples = max_eval_generate_examples  # None = use full batch when possible
        self.max_metric_samples = max_metric_samples  # Limit samples for metrics to avoid slow decode
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
        self.sampler_shuffle_mode = str(sampler_shuffle_mode)

        # --- OOM tracking ---
        self._oom_events = 0           # Total OOM events across all batches
        self._oom_skipped_batches = 0  # Batches skipped after exhausting retries

        # --- Adaptive Regime Management ---
        # Regimes group sequences by length to learn stable microbatch limits.
        # Key = bucket of max effective length, Value = {"B": batch_size, "T": token_budget, "stable": success_count}
        self._regime_limits = {}
        self._regime_bucket_size = 128   # Bucket width in tokens (groups similar lengths)
        self._regime_ramp_every = 4      # Ramp limits after N consecutive successes
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
        # Allow autotuning to push T up to 16x the initial budget (the regime will
        # OOM-shrink back down if it overshoots, so a generous ceiling is safe)
        self._regime_max_T = int(max_tokens_per_microbatch * 16) if max_tokens_per_microbatch is not None else (1 << 62)
        if regime_max_B is not None:
            self._regime_max_B = int(regime_max_B)
        elif max_tokens_per_microbatch is not None:
            # Auto: allow enough examples to fill the token budget even with very short sequences
            self._regime_max_B = max(1024, int(max_tokens_per_microbatch))
        else:
            self._regime_max_B = 8192

        self.builtin_metrics = tuple(builtin_metrics) if builtin_metrics else tuple()
        self.builtin_metrics_tokenize = str(builtin_metrics_tokenize)
        self.builtin_metrics_lowercase = bool(builtin_metrics_lowercase)

        # Pre-load metric objects once (with logging suppressed) to avoid verbose output every eval
        self._bleu_metric = None
        self._chrf_metric = None
        self._meteor_metric = None
        if self.builtin_metrics:
            self._load_builtin_metrics()

        self._cached_forward_keys = None

        # Baseline GPU memory (model + optimizer, before any batch).
        # Captured lazily on the first training_step call.
        self._gpu_baseline_mem: int | None = None

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

    def _load_builtin_metrics(self):
        """Load metric objects once at init with logging suppressed."""
        import logging
        import warnings
        import os

        # Suppress evaluate library's verbose caching messages
        evaluate_logger = logging.getLogger("evaluate")
        old_level = evaluate_logger.level
        evaluate_logger.setLevel(logging.ERROR)

        # Suppress nltk download messages (used by METEOR)
        nltk_logger = logging.getLogger("nltk")
        nltk_old_level = nltk_logger.level
        nltk_logger.setLevel(logging.ERROR)

        # Also suppress stdout from nltk.download
        old_nltk_quiet = os.environ.get("NLTK_DATA", None)

        try:
            import evaluate

            # Normalize metric names
            want = set(str(k).strip().lower() for k in self.builtin_metrics)
            aliases = {
                "sacrebleu": "bleu", "bleu1": "bleu", "bleu-1": "bleu",
                "chr_f": "chrf", "chrf++": "chrf", "chrfpp": "chrf",
            }
            want = set(aliases.get(k, k) for k in want)

            # Load with warnings suppressed
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if "bleu" in want:
                    self._bleu_metric = evaluate.load("sacrebleu")
                if "chrf" in want:
                    self._chrf_metric = evaluate.load("chrf")
                if "meteor" in want:
                    # METEOR triggers nltk downloads - suppress those too
                    import nltk
                    nltk.download('wordnet', quiet=True)
                    nltk.download('punkt_tab', quiet=True)
                    nltk.download('omw-1.4', quiet=True)
                    self._meteor_metric = evaluate.load("meteor")

        finally:
            evaluate_logger.setLevel(old_level)
            nltk_logger.setLevel(nltk_old_level)

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

    def _gather_pair_for_metrics(self, pred_ids: torch.Tensor, label_ids: torch.Tensor, pad_id: int):
        """
        Returns gathered (pred_ids, labels) across processes if needed.
        On single GPU, returns inputs unchanged.
        """
        # replace -100 with pad for decode
        labs = torch.where(label_ids != -100, label_ids, torch.full_like(label_ids, pad_id))

        # Ensure 2D
        if pred_ids.ndim == 1:
            pred_ids = pred_ids.unsqueeze(0)
        if labs.ndim == 1:
            labs = labs.unsqueeze(0)

        # pad to same length locally
        L = max(pred_ids.size(1), labs.size(1))
        pred_ids = _pad_to_len(pred_ids, pad_id, L)
        labs     = _pad_to_len(labs,     pad_id, L)

        # multi-proc only
        acc = getattr(self, "accelerator", None)
        if acc is None or getattr(acc, "num_processes", 1) == 1:
            return pred_ids, labs

        pred_g = acc.gather_for_metrics(pred_ids)
        labs_g = acc.gather_for_metrics(labs)
        return pred_g, labs_g

    def _forward_keys(self) -> set[str] | None:
        """
        Return keys accepted by model.forward(). Cached after first call.
        If signature introspection fails, return None (meaning: don't signature-filter).
        """
        if self._cached_forward_keys is not None:
            return self._cached_forward_keys

        try:
            sig = inspect.signature(self.model.forward)
            keys = set(sig.parameters.keys())
            # Some models use **kwargs; if so, signature filtering is pointless.
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                self._cached_forward_keys = None
            else:
                self._cached_forward_keys = keys
        except Exception:
            self._cached_forward_keys = None

        return self._cached_forward_keys


    def _build_gen_kwargs(self) -> dict:
        gen_kwargs: dict[str, Any] = {}
        if getattr(self.args, "generation_max_length", None) is not None:
            gen_kwargs["max_length"] = int(self.args.generation_max_length)
        if getattr(self.args, "generation_num_beams", None) is not None:
            gen_kwargs["num_beams"] = int(self.args.generation_num_beams)
        if getattr(self.args, "do_sample", False):
            gen_kwargs["do_sample"] = True
            if getattr(self.args, "top_k", None) is not None:
                gen_kwargs["top_k"] = int(self.args.top_k)
            if getattr(self.args, "top_p", None) is not None:
                gen_kwargs["top_p"] = float(self.args.top_p)
        # allow overrides
        if hasattr(self, "_gen_kwargs") and isinstance(self._gen_kwargs, dict):
            tmp = dict(self._gen_kwargs)
            tmp.update(gen_kwargs)
            gen_kwargs = tmp
        return gen_kwargs

    def _estimate_max_gen_examples(
        self,
        gen_kwargs: dict,
        enc_len: int | None = None,
        safety: float = 0.70,
    ) -> int | None:
        """
        Estimate the max generation batch size that fits in free VRAM.

        Uses model config to compute KV-cache cost per example, then divides
        available memory by that cost.  Returns None if estimation is not
        possible (e.g. no CUDA, missing config attrs).

        Args:
            gen_kwargs: generation kwargs (needs 'max_length' or 'max_new_tokens')
            enc_len:    encoder sequence length to assume.  Falls back to
                        self.max_encoder_len, then model config max position.
            safety:     fraction of free VRAM to budget (0.70 = 70%).
        """
        if not torch.cuda.is_available():
            return None

        # ---- generation length ------------------------------------------------
        gen_max = (
            gen_kwargs.get("max_length")
            or gen_kwargs.get("max_new_tokens")
            or getattr(self.args, "generation_max_length", None)
        )
        if gen_max is None:
            return None
        gen_max = int(gen_max)

        # ---- encoder length ---------------------------------------------------
        if enc_len is None:
            enc_len = self.max_encoder_len
        if enc_len is None:
            enc_len = getattr(self.model.config, "max_position_embeddings", 512)
        enc_len = int(enc_len)

        # ---- model dimensions -------------------------------------------------
        cfg = self.model.config
        # decoder layers (enc-dec models often split these)
        n_dec_layers = (
            getattr(cfg, "num_decoder_layers", None)
            or getattr(cfg, "num_hidden_layers", None)
            or getattr(cfg, "num_layers", None)
        )
        if n_dec_layers is None:
            return None

        d_model = getattr(cfg, "d_model", None) or getattr(cfg, "hidden_size", None)
        if d_model is None:
            return None

        n_heads = (
            getattr(cfg, "num_attention_heads", None)
            or getattr(cfg, "num_heads", None)
        )
        n_kv_heads = getattr(cfg, "num_key_value_heads", None) or n_heads
        if n_kv_heads is None:
            return None

        d_kv = getattr(cfg, "d_kv", None) or (d_model // n_heads if n_heads else None)
        if d_kv is None:
            return None

        # dtype bytes (match model dtype)
        try:
            dtype_bytes = next(self.model.parameters()).element_size()
        except StopIteration:
            dtype_bytes = 2  # assume fp16/bf16

        # ---- KV cache cost per example ----------------------------------------
        # Each layer stores K and V tensors; for encoder-decoder models there is
        # both cross-attention KV (over enc_len) and self-attention KV (over
        # gen_max tokens) in the decoder.
        kv_per_tok_per_layer = 2 * n_kv_heads * d_kv * dtype_bytes

        is_enc_dec = getattr(cfg, "is_encoder_decoder", False)
        if is_enc_dec:
            # cross-attn KV: decoder reads encoder output at every step
            cross_attn_kv = int(n_dec_layers) * enc_len * kv_per_tok_per_layer
            # self-attn KV: grows to gen_max
            self_attn_kv = int(n_dec_layers) * gen_max * kv_per_tok_per_layer
            kv_per_example = cross_attn_kv + self_attn_kv
        else:
            kv_per_example = int(n_dec_layers) * (enc_len + gen_max) * kv_per_tok_per_layer

        # Activations + intermediates per decode step (rough: ~4× one layer's KV
        # covers hidden states, attention scores, FFN intermediates)
        act_per_example = int(n_dec_layers) * d_model * 4 * dtype_bytes

        mem_per_example = kv_per_example + act_per_example
        if mem_per_example <= 0:
            return None

        # ---- available memory -------------------------------------------------
        free_mem, _total = torch.cuda.mem_get_info(self.args.device)
        available = int(free_mem * safety)

        max_batch = max(1, available // mem_per_example)

        if self.debug:
            logger.info(
                f"[TokenPackTrainer] Gen batch auto-tune: "
                f"free={free_mem / (1 << 30):.2f}GB, "
                f"kv/ex={kv_per_example / (1 << 20):.1f}MB, "
                f"act/ex={act_per_example / (1 << 20):.1f}MB, "
                f"max_batch={max_batch} (safety={safety})"
            )

        return max_batch

    def _filter_for_generate(
        self,
        batch: dict,
        ignore: set[str] | None = None,
    ) -> dict:
        """
        Return a dict safe to pass into model.generate():

        - drop keys in ignore
        - keep tensor values only
        - optionally signature-filter to model.forward() accepted keys
          (plus a small allowlist of common keys)
        """
        if ignore is None:
            ignore = set()

        out = {}
        fwd_keys = self._forward_keys()

        for k, v in batch.items():
            if k in ignore:
                continue
            if not _is_tensorish(v):
                continue

            if fwd_keys is None:
                # introspection unavailable or model.forward has **kwargs
                out[k] = v
            else:
                if (k in fwd_keys) or (k in _ALWAYS_ALLOW):
                    out[k] = v

        return out

    def _normalize_eval_mode(self, mode: str | None) -> str | None:
        if mode is None:
            return None
        m = str(mode).lower()
        if m in ("none", "hf", "vanilla", "huggingface"):
            return None
        if m in ("token_aware", "token-aware", "token_aware_metrics"):
            return "token_aware_metrics"
        return m

    def _regime_key_from_lengths(self, enc_len, dec_len) -> int:
        """Compute regime key from precomputed lengths (avoids redundant recomputation)."""
        alpha = 2.0
        eff = enc_len + alpha * dec_len
        mx = int(eff.max().item()) if eff.numel() else 0
        bs = int(self._regime_bucket_size)
        return int((mx + bs - 1) // bs) if bs > 0 else mx

    def _regime_key_from_inputs(self, inputs_cpu: dict) -> int:
        """
        Compute a coarse "regime key" from the maximum effective length
        in this HF batch: eff = enc_len + alpha * dec_len.
        """
        enc_len, dec_len, _ = self._compute_lengths_enc_dec(inputs_cpu)
        return self._regime_key_from_lengths(enc_len, dec_len)

    def _regime_state(self, key: int) -> dict:
        st = self._regime_limits.get(key)
        if st is None:
            st = {"B": self._regime_default_B, "T": self._regime_default_T, "stable": 0}
            # B=None is valid and means "uncapped" — only default T if missing
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
            # B=None means uncapped — no ramping needed
            if st["B"] is not None:
                st["B"] = max(self._regime_min_B, int(st["B"] * float(self._regime_ramp_B)) + 1)
                st["B"] = self._clamp_int(st["B"], self._regime_min_B, self._regime_max_B)

            st["T"] = max(self._regime_min_T, int(st["T"] * float(self._regime_ramp_T)))
            # hard clamp to practical + int64 safety
            st["T"] = self._clamp_int(st["T"], self._regime_min_T, min(self._regime_max_T, self.INT64_MAX))

    def _regime_on_oom(self, key: int):
        st = self._regime_state(key)
        st["stable"] = 0

        # Track OOM count per regime for safety factor adjustment on re-autotune
        oom_history = getattr(self, "_regime_oom_count", {})
        oom_history[key] = oom_history.get(key, 0) + 1
        self._regime_oom_count = oom_history

        # Allow re-autotuning after recovery stabilizes.
        # Without this, a single OOM can condemn a regime to slow incremental
        # ramp-up (~930 steps to recover from T=64 → T=16384).
        autotuned = getattr(self, "_autotuned_keys", set())
        if key in autotuned:
            autotuned.discard(key)

        # Shrink B first — if uncapped, materialize from regime_max_B so we can shrink
        if st["B"] is None:
            st["B"] = self._regime_max_B
            return

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

    def _autotune_regime_from_peak(self, key: int, eff_tokens_in_step: int, safety: float = 0.90):
        if not torch.cuda.is_available() or eff_tokens_in_step <= 0:
            return

        total = torch.cuda.get_device_properties(self.args.device.index).total_memory
        peak  = torch.cuda.max_memory_allocated(self.args.device)

        # avoid divide-by-zero / nonsense
        if peak <= 0:
            return

        # Use baseline memory (model + optimizer) to compute MARGINAL per-token cost.
        # Without this, model weights (~90% of VRAM) inflate per-token cost by 10x+,
        # making the autotuned token budget far too conservative.
        baseline = self._gpu_baseline_mem or 0
        available = max(total - baseline, total * 0.05)  # headroom beyond model; floor at 5%
        batch_mem = max(peak - baseline, available * 0.05)  # actual batch memory used

        bytes_per_eff_token = batch_mem / float(eff_tokens_in_step)

        # Target: use safety fraction of AVAILABLE memory (not total) for batch data.
        # This correctly handles the common case where model+optimizer use 85-95% of VRAM.
        target_batch_mem = safety * available
        target_eff_tokens = int(target_batch_mem / max(bytes_per_eff_token, 1e-9))

        # clamp hard
        INT64_MAX = (1 << 63) - 1
        st = self._regime_state(key)
        old_T = st["T"]
        st["T"] = max(self._regime_min_T, min(target_eff_tokens, INT64_MAX, getattr(self, "_regime_max_T", INT64_MAX)))

        if self.debug:
            print(
                f"[autotune] regime {key}: baseline={baseline/(1<<30):.1f}G, "
                f"peak={peak/(1<<30):.1f}G, batch_mem={batch_mem/(1<<30):.1f}G, "
                f"bytes/tok={bytes_per_eff_token:.0f}, eff_tokens={eff_tokens_in_step}, "
                f"target_tokens={target_eff_tokens}, T: {old_T} -> {st['T']}"
            )

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

        # ----------------------------
        # 1) HF default path
        # ----------------------------
        if self.max_tokens_per_batch is None:
            dl = super().get_train_dataloader()

            # Optional GPU prefetch wrapper (only meaningful when HF places batches on GPU)
            if torch.cuda.is_available() and not self.use_cpu_microbatch:
                dl = CUDAPrefetcher(dl, self.args.device)

            return dl

        # ----------------------------
        # 2) Token-budget sampler path
        # ----------------------------
        ds = self.train_dataset

        # Only strip columns if HF would have done so; otherwise keep task/metadata for mixed collators
        if hasattr(ds, "column_names") and getattr(self.args, "remove_unused_columns", True):
            keep_cols = {
                "input_ids",
                "attention_mask",
                "labels",
                "decoder_input_ids",
                "decoder_attention_mask",
                self.length_column_name,
                # safe extras
                "task",
                "len_allowed",
                "meteor_ok",
            }
            to_remove = [c for c in ds.column_names if c not in keep_cols]
            if to_remove:
                ds = ds.remove_columns(to_remove)

        # lengths used by sampler
        if hasattr(ds, "column_names"):
            if self.length_column_name not in ds.column_names:
                raise ValueError(
                    f"Column '{self.length_column_name}' doesn't exist in train_dataset. "
                    f"Available columns: {ds.column_names}"
                )
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
            bucket_size=self.sampler_bucket_size,
            shuffle=True,
            drop_last=False,
            long_behavior="truncate",
            max_length_in_batch=self.max_encoder_len,
            shuffle_mode=self.sampler_shuffle_mode,
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

        # Optional GPU prefetch wrapper
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
        dec_max = int(dec_len.max().item()) if dec_len.numel() else 0
        tot_max = int(total_len.max().item()) if total_len.numel() else 0

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

    def _make_microbatches(
        self,
        inputs,
        max_tokens_per_microbatch: int | None = None,
        return_lengths: bool = False,
        precomputed_lengths: tuple | None = None,
    ):
        # Reuse precomputed lengths if available (avoids redundant tensor ops)
        if precomputed_lengths is not None:
            enc_len, dec_len = precomputed_lengths
        else:
            enc_len, dec_len, _ = self._compute_lengths_enc_dec(inputs)

        alpha = 2.0
        effective_len = enc_len + alpha * dec_len
        N = int(effective_len.numel())

        budget = int(max_tokens_per_microbatch or self.max_tokens_per_microbatch)
        max_B = self.max_examples_per_microbatch  # may be None
        padding_aware = getattr(self, "padding_aware_budget", False)

        # --- Sort once on GPU, then bulk-convert to Python lists (no per-element .item()) ---
        order = torch.argsort(effective_len)
        order_cpu = order.cpu()
        sorted_eff_list = effective_len.cpu()[order_cpu].tolist()

        # Reorder entire batch once (single index_select) so microbatches are contiguous slices
        sorted_inputs = self._slice_inputs(inputs, order_cpu.tolist())

        # Infer batch dim for contiguous slicing
        _batch_dim = sorted_inputs["input_ids"].size(0) if "input_ids" in sorted_inputs else None

        # --- Compute split sizes in pure Python (no tensor ops in loop) ---
        split_sizes = []
        cur_tokens = 0
        cur_max_len = 0
        cur_count = 0

        for L in sorted_eff_list:
            if padding_aware:
                new_max = max(cur_max_len, L)
                too_many_tokens = cur_count > 0 and new_max * (cur_count + 1) > budget
            else:
                too_many_tokens = cur_count > 0 and cur_tokens + L > budget

            too_many_examples = max_B is not None and cur_count >= max_B

            if too_many_tokens or too_many_examples:
                split_sizes.append(cur_count)
                cur_tokens = L
                cur_max_len = L
                cur_count = 1
            else:
                cur_tokens += L
                cur_max_len = max(cur_max_len, L)
                cur_count += 1

        if cur_count > 0:
            split_sizes.append(cur_count)

        # --- Split into microbatches using contiguous slicing (views, not copies) ---
        result = []
        offset = 0
        for size in split_sizes:
            mb = {}
            for k, v in sorted_inputs.items():
                if isinstance(v, torch.Tensor) and v.ndim >= 1 and _batch_dim is not None and v.size(0) == _batch_dim:
                    mb[k] = v[offset:offset + size]
                else:
                    mb[k] = v
            result.append(self._compact_microbatch(mb))
            offset += size

        if return_lengths:
            return result, enc_len, dec_len
        return result

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
        # skip empty batch defensively
        am = inputs.get("attention_mask", None)
        if am is not None and torch.is_tensor(am) and am.numel() == 0:
            return (None, None, None)
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
            # HF's get_eval_dataloader uses self.data_collator, but we want
            # to respect self.eval_data_collator if the user provided one.
            # Temporarily swap collators so HF uses the right one.
            if self.eval_data_collator is not None:
                orig_collator = self.data_collator
                self.data_collator = self.eval_data_collator
                try:
                    return super().get_eval_dataloader(eval_dataset)
                finally:
                    self.data_collator = orig_collator
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

        # Use the user's collator - important for pretraining with span corruption, etc.
        collate_fn = self.eval_data_collator or self.data_collator

        num_workers = self.args.dataloader_num_workers
        dl_kwargs = dict(
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        # Only add these if num_workers > 0 (they cause issues otherwise)
        if num_workers > 0:
            dl_kwargs["persistent_workers"] = self.args.dataloader_persistent_workers
            dl_kwargs["prefetch_factor"] = getattr(self.args, "dataloader_prefetch_factor", 2)

        return DataLoader(eval_dataset, **dl_kwargs)
        
        
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
        """
        Token-aware evaluation that is responsive by default.

        Behavior:
          - Always computes token-weighted eval_loss
          - Only runs generation if (predict_with_generate=True) OR metrics are requested
          - Metrics sources:
              * user compute_metrics (if provided) -- computed STREAMING is not possible via HF API
                so we DO NOT call it here.
              * built-in metrics (BLEU/chrF/METEOR) if self.builtin_metrics includes them
                -- computed streaming via evaluate.add_batch()

        Notes:
          - Avoids "tqdm 100% then hang" by not doing giant end-of-epoch concat+decode.
          - METEOR is slow: only run if explicitly included in builtin_metrics.
        """
        import time
        from tqdm import tqdm

        gen_kwargs = self._build_gen_kwargs()

        # ---- pick dataset
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Need an eval_dataset for token-aware evaluation.")

        # ---- override eval token budget for this call if requested
        if max_eval_tokens_per_microbatch is not None:
            self.max_eval_tokens_per_microbatch = int(max_eval_tokens_per_microbatch)

        # ---- determine what we are doing
        wants_generate = bool(getattr(self.args, "predict_with_generate", False))
        wants_builtin = bool(getattr(self, "builtin_metrics", ()))
        do_generate = wants_generate or wants_builtin

        # ---- auto-tune generation batch size from available VRAM
        if do_generate and self.max_eval_generate_examples is None:
            auto_B = self._estimate_max_gen_examples(gen_kwargs)
            if auto_B is not None:
                self.max_eval_generate_examples = auto_B
                logger.info(
                    f"[TokenPackTrainer] Auto-set max_eval_generate_examples={auto_B} "
                    f"from VRAM estimate"
                )

        acc = getattr(self, "accelerator", None)
        num_proc = getattr(acc, "num_processes", 1) if acc is not None else 1
        # On single process, always treat as main; otherwise check accelerator
        is_main = (num_proc == 1) or (acc is not None and bool(acc.is_main_process))

        # Use pre-loaded metrics (loaded once at __init__ to avoid verbose logging every eval)
        bleu_obj = self._bleu_metric if (do_generate and wants_builtin and is_main) else None
        chrf_obj = self._chrf_metric if (do_generate and wants_builtin and is_main) else None
        meteor_obj = self._meteor_metric if (do_generate and wants_builtin and is_main) else None

        if self.debug and is_main:
            logger.info(f"[TokenPackTrainer] Metric setup: bleu={bleu_obj is not None}, "
                       f"chrf={chrf_obj is not None}, meteor={meteor_obj is not None}, "
                       f"is_main={is_main}, num_proc={num_proc}")

        # Warn if we expected to compute metrics but conditions weren't met
        if do_generate and wants_builtin and not is_main:
            logger.warning(
                f"[TokenPackTrainer] builtin_metrics requested but is_main={is_main}. "
                f"Metrics will only be computed on main process. "
                f"(num_proc={num_proc}, acc={acc})"
            )

        total_eval_loss = 0.0
        total_eval_tokens = 0
        num_steps = 0
        num_examples = 0
        metrics = {}

        dataloader = self.get_eval_dataloader(eval_dataset)
        pad_id = getattr(self.processing_class, "pad_token_id", None) or 0

        ignore_keys = {
            "labels",
            self.length_column_name,
            "decoder_input_ids",
            "decoder_attention_mask",
            "meteor_ok",
        }

        def _do_generate(mb_gpu: dict) -> torch.Tensor:
            gen_inputs = self._filter_for_generate(mb_gpu, ignore_keys)
            with torch.inference_mode():
                out = self.model.generate(**gen_inputs, **gen_kwargs)
            if out.ndim == 1:
                out = out.unsqueeze(0)
            return out   # ← REQUIRED

        # Store metric objects and accumulated predictions/references
        # We accumulate decoded strings and compute metrics ONCE at the end (much faster)
        metric_state = {
            "bleu_obj": bleu_obj,
            "chrf_obj": chrf_obj,
            "meteor_obj": meteor_obj,
            "metric_examples": 0,
            "all_preds": [],  # accumulate all predictions
            "all_refs": [],   # accumulate all references
        }

        def _add_streaming_metrics(gen_ids: torch.Tensor, labels: torch.Tensor):
            _bleu = metric_state["bleu_obj"]
            _chrf = metric_state["chrf_obj"]
            _meteor = metric_state["meteor_obj"]

            # Use explicit None checks - EvaluationModule objects are falsy when empty!
            if _bleu is None and _chrf is None and _meteor is None:
                return

            # gather if multi-proc; no-op on 1 GPU
            pred_g, lab_g = self._gather_pair_for_metrics(gen_ids, labels, pad_id)

            if not is_main:
                return

            # Move to CPU before decoding for efficiency
            if pred_g.device.type != 'cpu':
                pred_g = pred_g.cpu()
            if lab_g.device.type != 'cpu':
                lab_g = lab_g.cpu()

            preds_txt = self.processing_class.batch_decode(pred_g, skip_special_tokens=True)
            refs_txt  = self.processing_class.batch_decode(lab_g,  skip_special_tokens=True)

            preds = [p.strip() for p in preds_txt]
            refs  = [[r.strip()] for r in refs_txt]

            if len(preds) == 0:
                return

            # Accumulate instead of calling add_batch every time (MUCH faster)
            metric_state["metric_examples"] += len(preds)
            metric_state["all_preds"].extend(preds)
            metric_state["all_refs"].extend(refs)

        # Compute generation stride to evenly sample across the full eval set
        # (ensures all tasks/domains are represented in metrics, not just the first block)
        _gen_stride = 1
        if self.max_metric_samples is not None and do_generate:
            _n_eval = len(eval_dataset) if hasattr(eval_dataset, "__len__") else None
            if _n_eval is not None and _n_eval > self.max_metric_samples:
                _gen_stride = max(1, _n_eval // self.max_metric_samples)

        start_time = time.time()

        for batch in tqdm(dataloader, desc=desc, leave=True, disable=bool(getattr(self.args, "disable_tqdm", False))):
            num_steps += 1

            labels = batch.get("labels", None)
            if labels is None:
                raise ValueError("Eval dataset must have labels for token-aware evaluation.")

            # Skip truly empty batches (0 examples)
            batch_size = labels.size(0) if torch.is_tensor(labels) else 0
            if batch_size == 0:
                continue

            # loss computation with OOM handling - try full batch first, fall back to microbatches
            batch_loss = None
            batch_ntok = 0
            try:
                with torch.no_grad():
                    loss, _, _ = self.prediction_step(self.model, batch, prediction_loss_only=True)
                if loss is not None:
                    batch_ntok = int((labels.detach().cpu() != -100).sum().item())
                    if batch_ntok > 0:
                        batch_loss = float(loss.item())
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if not self._is_cuda_oom(e):
                    raise
                # OOM on loss - fall back to microbatch loss computation
                self._eval_oom_cleanup()

                # Compute loss in microbatches
                if self.use_cpu_microbatch:
                    loss_inputs = self._truncate_batch(self._move_to_cpu(batch))
                else:
                    loss_inputs = self._truncate_batch(batch)

                loss_microbatches = self._make_microbatches(
                    loss_inputs,
                    max_tokens_per_microbatch=self.max_eval_tokens_per_microbatch,
                )

                mb_loss_sum = 0.0
                mb_tok_sum = 0
                with torch.no_grad():
                    for mb in loss_microbatches:
                        if self.use_cpu_microbatch:
                            mb = self._to_device(mb)
                        else:
                            mb = self._prepare_inputs(mb)
                        try:
                            mb_labels = mb.get("labels", None)
                            loss_mb, _, _ = self.prediction_step(self.model, mb, prediction_loss_only=True)
                            if loss_mb is not None and mb_labels is not None:
                                mb_ntok = int((mb_labels.detach().cpu() != -100).sum().item())
                                if mb_ntok > 0:
                                    mb_loss_sum += float(loss_mb.item()) * mb_ntok
                                    mb_tok_sum += mb_ntok
                        except (RuntimeError, torch.cuda.OutOfMemoryError) as mb_e:
                            if not self._is_cuda_oom(mb_e):
                                raise
                            self._eval_oom_cleanup()
                            # Skip this microbatch on OOM
                            continue

                if mb_tok_sum > 0:
                    batch_loss = mb_loss_sum / mb_tok_sum
                    batch_ntok = mb_tok_sum

            if batch_loss is not None and batch_ntok > 0:
                total_eval_loss += batch_loss * batch_ntok
                total_eval_tokens += batch_ntok

            num_examples += int(labels.size(0))

            if not do_generate:
                continue

            # Gate generation: skip batches via stride to sample evenly across full dataset
            if _gen_stride > 1 and (num_steps % _gen_stride) != 1:
                continue

            if self.use_cpu_microbatch:
                batch_for_gen = self._to_device(self._truncate_batch(self._move_to_cpu(batch)))
            else:
                batch_for_gen = self._truncate_batch(self._prepare_inputs(batch))
            # ---- fast path generate whole batch ----
            # Skip fast path if batch exceeds max_eval_generate_examples (avoids guaranteed OOM)
            gen_limit = getattr(self, "max_eval_generate_examples", None)
            fast_path_ok = (gen_limit is None or batch_for_gen["input_ids"].size(0) <= gen_limit)
            if fast_path_ok:
                try:
                    gen_ids = _do_generate(batch_for_gen)
                    _add_streaming_metrics(gen_ids, batch_for_gen["labels"])
                    self._eval_on_success()
                    continue
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    if not self._is_cuda_oom(e):
                        raise
                    self._eval_oom_cleanup()
                    # fall through to microbatching

            # microbatch generation path (OOM-safe)
            # When use_cpu_microbatch=True, plan on CPU and _prefetch_microbatches
            # streams each microbatch to GPU.  When False, move the whole batch to
            # GPU first (input tensors are tiny vs. model activations, so this
            # won't OOM) so that _do_generate receives GPU tensors.
            if self.use_cpu_microbatch:
                plan_inputs = self._truncate_batch(self._move_to_cpu(batch))
            else:
                plan_inputs = self._truncate_batch(self._prepare_inputs(batch))

            last_err = None
            for attempt in range(int(self.oom_max_retries) + 1):
                try:
                    orig_use_cache = getattr(self.model.config, "use_cache", True)
                    self.model.config.use_cache = True

                    orig_max_B = self.max_examples_per_microbatch
                    gen_limit = getattr(self, "max_eval_generate_examples", None)
                    if gen_limit is not None:
                        self.max_examples_per_microbatch = int(gen_limit)

                    microbatches = self._make_microbatches(
                        plan_inputs,
                        max_tokens_per_microbatch=self.max_eval_tokens_per_microbatch,
                    )
                    self.max_examples_per_microbatch = orig_max_B

                    mb_iter = self._prefetch_microbatches(microbatches) if self.use_cpu_microbatch else iter(microbatches)

                    for mb in mb_iter:
                        gen_ids = _do_generate(mb)
                        _add_streaming_metrics(gen_ids, mb["labels"])
                        

                    last_err = None
                    self._eval_on_success()
                    break

                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    if not self._is_cuda_oom(e):
                        raise
                    last_err = e
                    self._eval_on_oom()
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


        # Note: metrics are computed in the finalize section below using metric_state

        runtime = time.time() - start_time if num_steps > 0 else 0.0
        eval_loss = (total_eval_loss / total_eval_tokens) if total_eval_tokens > 0 else float("nan")

        # Warn if we processed steps but got no examples (indicates a problem)
        if num_steps > 0 and num_examples == 0:
            logger.warning(
                f"[TokenPackTrainer] Processed {num_steps} eval steps but num_examples=0. "
                f"Check that eval_dataset is not empty and batches contain data."
            )

        metrics["eval_loss"] = float(eval_loss)
        if runtime > 0:
            metrics["eval_runtime"] = float(runtime)
            metrics["eval_samples_per_second"] = float(num_examples / runtime)
            metrics["eval_steps_per_second"] = float(num_steps / runtime)

        # finalize metrics (main only in multi-proc, but we still return on all ranks)
        # Use metric_state to get the updated values
        _bleu_obj = metric_state["bleu_obj"]
        _chrf_obj = metric_state["chrf_obj"]
        _meteor_obj = metric_state["meteor_obj"]
        _metric_examples = metric_state["metric_examples"]
        _all_preds = metric_state["all_preds"]
        _all_refs = metric_state["all_refs"]

        if wants_builtin:
            if is_main and _metric_examples > 0 and len(_all_preds) > 0:
                # Compute metrics ONCE on all accumulated predictions (much faster than streaming)
                if _bleu_obj is not None:
                    bleu_res = _bleu_obj.compute(
                        predictions=_all_preds,
                        references=_all_refs,
                        tokenize=self.builtin_metrics_tokenize,
                        lowercase=self.builtin_metrics_lowercase,
                        force=True  # Suppress "forgot to detokenize" warning - both preds and refs go through same decode
                    )
                    metrics["eval_bleu"] = float(bleu_res["score"])
                else:
                    metrics["eval_bleu"] = float("nan")

                if _chrf_obj is not None:
                    chrf_res = _chrf_obj.compute(
                        predictions=_all_preds,
                        references=_all_refs
                    )
                    metrics["eval_chrf"] = float(chrf_res["score"])
                else:
                    metrics["eval_chrf"] = float("nan")

                if _meteor_obj is not None:
                    # METEOR expects flat references, not nested
                    flat_refs = [r[0] for r in _all_refs]
                    mres = _meteor_obj.compute(
                        predictions=_all_preds,
                        references=flat_refs
                    )
                    metrics["eval_meteor"] = float(mres["meteor"])
                else:
                    metrics["eval_meteor"] = float("nan")

                metrics["metric_examples"] = float(_metric_examples)
            else:
                # On non-main ranks (multi-proc) or if nothing was added:
                metrics.setdefault("eval_bleu", float("nan"))
                metrics.setdefault("eval_chrf", float("nan"))
                metrics.setdefault("eval_meteor", float("nan"))
                metrics.setdefault("metric_examples", float(_metric_examples))

                # Warn if we expected metrics but got none (helps debug)
                if is_main and _metric_examples == 0 and (_bleu_obj is not None or _chrf_obj is not None):
                    logger.warning(
                        f"[TokenPackTrainer] Expected metrics but metric_examples=0. "
                        f"bleu_obj={_bleu_obj is not None}, chrf_obj={_chrf_obj is not None}, "
                        f"do_generate={do_generate}, num_examples={num_examples}"
                    )

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
        Modes:
          - None/"hf"/"vanilla": pure HF super().evaluate (respects predict_with_generate + compute_metrics)
          - "token_aware_loss": token-weighted loss only (fast, no generate)
          - "token_aware_metrics": token-aware eval that computes loss always and
                optionally generation + built-in metrics (if requested)
          - "auto": if predict_with_generate or builtin_metrics -> token_aware_metrics else token_aware_loss
        """
        eval_dataset = eval_dataset or self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Evaluation requires an eval_dataset.")

        requested = eval_mode if eval_mode is not None else getattr(self, "eval_mode", None)
        mode = self._normalize_eval_mode(requested)

        wants_generate = bool(getattr(self.args, "predict_with_generate", False))
        wants_builtin = bool(getattr(self, "builtin_metrics", ()))
        has_user_metrics = getattr(self, "compute_metrics", None) is not None

        # ---- 1) HF path (fully HF)
        if mode is None:
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        # ---- 2) token-aware loss-only
        if mode == "token_aware_loss":
            # Force no-generate semantics for token-aware loss-only
            if wants_generate:
                # don't mutate args permanently; just call your loss-only helper
                pass
            metrics = self._token_aware_loss_only(eval_dataset, desc="Eval (token-weighted loss)")
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

        # ---- 3) token-aware metrics (responsive)
        if mode == "token_aware_metrics":
            # If user provided compute_metrics, you have two choices:
            #   A) keep token-aware eval built-in metrics only (streaming)
            #   B) fall back to HF so user compute_metrics runs
            #
            # For a package, the least surprising is:
            #   - If user provided compute_metrics, and they asked for token-aware metrics,
            #     we still DO token-aware eval, but we DO NOT call user compute_metrics (HF API isn't streaming).
            #     We'll log a warning once.
            if has_user_metrics and self.control.should_log:
                self.log({"warn_compute_metrics_ignored_in_token_aware": 1.0})

            metrics = self._token_aware_evaluate(
                eval_dataset=eval_dataset,
                max_eval_tokens_per_microbatch=self.max_eval_tokens_per_microbatch,
                desc="eval (token-aware)",
            )
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

        # ---- 4) auto
        if mode == "auto":
            if wants_generate or wants_builtin:
                metrics = self._token_aware_evaluate(
                    eval_dataset=eval_dataset,
                    max_eval_tokens_per_microbatch=self.max_eval_tokens_per_microbatch,
                    desc="eval (token-aware)",
                )
                self.log(metrics)
                self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
                return metrics
            metrics = self._token_aware_loss_only(eval_dataset, desc="Eval (token-weighted loss)")
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

        # fallback: treat unknown as HF
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


    def _regime_key_from_inputs_any_device(self, inputs: dict) -> int:
        # expects attention_mask and (optionally) labels as tensors on CPU or GPU
        am = inputs["attention_mask"]
        enc_len = am.sum(dim=-1)

        labels = inputs.get("labels", None)
        if labels is None or labels.ndim != 2:
            dec_len = torch.zeros_like(enc_len)
        else:
            dec_len = (labels != -100).sum(dim=-1)

        eff = enc_len + 2.0 * dec_len
        mx = int(eff.max().item())  # <-- scalar sync only
        bs = int(self._regime_bucket_size)
        return int((mx + bs - 1) // bs) if bs > 0 else mx

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()

        # turn off KV cache during training to reduce memory spikes
        if hasattr(model, "config") and getattr(model.config, "use_cache", None) is True:
            model.config.use_cache = False

        # Track baseline GPU memory (model + optimizer, before any batch data).
        # This lets _autotune_regime_from_peak compute marginal per-token cost
        # instead of attributing fixed model memory to each token.
        # We take the MAX across the first few steps because Adam optimizer states
        # are allocated lazily after the first optimizer.step() (which runs after
        # training_step returns), so step 0 baseline only has model weights.
        if torch.cuda.is_available():
            current_mem = torch.cuda.memory_allocated(self.args.device)
            if self._gpu_baseline_mem is None or current_mem > self._gpu_baseline_mem:
                old_baseline = self._gpu_baseline_mem
                self._gpu_baseline_mem = current_mem
                if self.debug:
                    if old_baseline is not None:
                        print(f"[TokenPackTrainer] GPU baseline updated: {old_baseline / (1<<30):.2f} -> {current_mem / (1<<30):.2f} GB")
                    else:
                        print(f"[TokenPackTrainer] GPU baseline memory: {current_mem / (1<<30):.2f} GB")
                # When baseline jumps (e.g. optimizer states allocated after step 0),
                # clear autotuned keys so regimes re-autotune with accurate baseline.
                if old_baseline is not None and current_mem > old_baseline * 1.05:
                    autotuned = getattr(self, "_autotuned_keys", set())
                    if autotuned:
                        if self.debug:
                            print(f"[TokenPackTrainer] Baseline jumped {(current_mem - old_baseline)/(1<<30):.2f}G — re-autotuning {len(autotuned)} regime(s)")
                        self._autotuned_keys = set()

        last_err = None
        regime_key = None  # <<< keep across retries for this same HF batch

        for attempt in range(self.oom_max_retries + 1):
            try:
                # ---------------- PLAN MICROBATCHE(S) ----------------
                if self.use_cpu_microbatch:
                    inputs_work = self._move_to_cpu(inputs)
                    inputs_work = self._truncate_batch(inputs_work)

                    # >>> compute lengths ONCE (post-truncation), reuse for regime key + microbatch splitting
                    enc_len, dec_len, _ = self._compute_lengths_enc_dec(inputs_work)
                    if regime_key is None:
                        regime_key = self._regime_key_from_lengths(enc_len, dec_len)

                    # >>> apply per-regime best limits BEFORE planning microbatches
                    self._apply_regime_limits(regime_key)

                    microbatches, enc_len, dec_len = self._make_microbatches(
                        inputs_work, return_lengths=True, precomputed_lengths=(enc_len, dec_len))

                else:
                    # GPU-native path
                    inputs_work = self._prepare_inputs(inputs)
                    inputs_work = self._truncate_batch(inputs_work)

                    # >>> compute lengths once (post-truncation), reuse
                    enc_len, dec_len, _ = self._compute_lengths_enc_dec(inputs_work)
                    if regime_key is None:
                        regime_key = self._regime_key_from_lengths(enc_len, dec_len)

                    self._apply_regime_limits(regime_key)

                    microbatches, enc_len, dec_len = self._make_microbatches(
                        inputs_work, return_lengths=True, precomputed_lengths=(enc_len, dec_len))

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
                    # Use lower safety factor if this regime recently OOM'd
                    # (previous autotune overshot, so be more conservative)
                    st = self._regime_state(regime_key)
                    oom_history = getattr(self, "_regime_oom_count", {})
                    recent_oom = oom_history.get(regime_key, 0) > 0
                    safety = 0.75 if recent_oom else 0.90
                    self._autotune_regime_from_peak(regime_key, eff_tokens, safety=safety)
                    self._autotuned_keys = getattr(self, "_autotuned_keys", set())
                    self._autotuned_keys.add(regime_key)
                    # Clear OOM history for this key now that we've re-autotuned
                    if recent_oom:
                        oom_history[regime_key] = 0
                        self._regime_oom_count = oom_history


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
