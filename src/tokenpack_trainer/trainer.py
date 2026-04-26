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

import json
import os
import threading
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

        def _safe_to_device(batch):
            """Move batch to GPU with OOM recovery.

            On OOM: empty the CUDA cache to defragment reserved-but-unused
            memory and retry once.  If that also fails, return the batch on
            CPU — training_step will handle device placement itself.
            """
            try:
                return _to_device(batch)
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                try:
                    torch.cuda.empty_cache()
                    return _to_device(batch)
                except (torch.cuda.OutOfMemoryError, RuntimeError):
                    return batch  # stay on CPU; training_step handles it

        with torch.cuda.stream(self.stream):
            next_batch = next(it, None)
            if next_batch is not None:
                next_batch = _safe_to_device(next_batch)

        while True:
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = next_batch
            if batch is None:
                break

            with torch.cuda.stream(self.stream):
                next_batch = next(it, None)
                if next_batch is not None:
                    next_batch = _safe_to_device(next_batch)

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

    use_cpu_microbatch_eval : bool or None
        Override use_cpu_microbatch for evaluation only. Set to False to keep
        eval batches on GPU (avoids repeated CPU→GPU transfers over slow PCIe).
        None (default) inherits from use_cpu_microbatch.

    eval_mode : str, optional
        "hf" or None: Use standard HF evaluation (faster, no generation).
        "token_aware_metrics": Use token-aware eval with generation for BLEU/chrF.

    Adaptive OOM Parameters:
    ------------------------
    decoder_cost_multiplier : float
        Multiplier for decoder tokens in the effective length formula
        (eff = enc_len + alpha * dec_len). Higher values allocate more memory
        budget per decoder token. Default: "auto" (auto-detected from model
        vocab_size: 2.0 for vocab ≤ 64K, scales up for larger vocabs like
        Gemma2's 256K).

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
        use_cpu_microbatch_eval: bool | None = None,
        eval_mode: str | None = None,  #use classic hf or switch to "token_aware_metrics" to use token packing
        debug: bool = False,
        decoder_cost_multiplier: float | str = "auto",  # "auto" = detect from vocab_size; or set manually (e.g. 8.0 for 256K vocab)
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
        regime_ramp_every: int | str = "auto",       # ramp limits after N consecutive successes ("auto" = VRAM-scaled)
        regime_ramp_B: float | str = "auto",         # factor to increase B on ramp ("auto" = VRAM-scaled)
        regime_ramp_T: float | str = "auto",         # factor to increase T on ramp ("auto" = VRAM-scaled)
        autotune_safety: float | str = "auto",       # safety factor for autotune (0-1, "auto" = VRAM-scaled)
        autotune_oom_safety: float | str = "auto",   # safety factor after OOM ("auto" = VRAM-scaled)
        eval_ramp_every: int | str = "auto",         # eval: ramp T after N successes ("auto" = VRAM-scaled)
        eval_ramp_T: float | str = "auto",           # eval: factor to increase T on ramp ("auto" = VRAM-scaled)
        max_microbatches_per_step: int = 8,              # cap microbatches per step; if exceeded, budget is bumped for the step
        seed_regime_state: str | None = None,          # path to regime_state.json or checkpoint dir to seed B/T defaults from a previous run
        adaptive_regime_state: bool = False,              # when True with seed_regime_state, explore beyond seeded B/T limits for new data
        max_eval_generate_examples: int | None = None,  # max examples per generate call (None = no limit, uses token budget only)
        builtin_metrics: tuple[str, ...] | None = None,   # e.g. ("bleu","chrf") or ("bleu","chrf","meteor")
        builtin_metrics_tokenize: str = "13a",
        builtin_metrics_lowercase: bool = False,
        max_metric_samples: int | None = 100000,  # Max samples for metric computation (None = all, default 2000 for speed)
        diagnostic_interval: int = 10,  # write diagnostics every N steps (0 = disable); file: {output_dir}/tokenpack_diagnostics.jsonl
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eval_data_collator = eval_data_collator
        self.padding_aware_budget = bool(padding_aware_budget)
        self.max_eval_generate_examples = max_eval_generate_examples  # None = auto-tune from VRAM
        self._user_set_gen_B = max_eval_generate_examples  # track if user explicitly set a value
        self.max_metric_samples = max_metric_samples  # Limit samples for metrics to avoid slow decode
        self.max_tokens_per_microbatch = int(max_tokens_per_microbatch)
        self.max_encoder_len = int(max_encoder_len) if max_encoder_len is not None else None
        self.max_decoder_len = int(max_decoder_len) if max_decoder_len is not None else None
        self.max_tokens_per_batch = max_tokens_per_batch
        self.length_column_name = length_column_name
        self.log_longest_every = int(log_longest_every)
        self.use_cpu_microbatch = bool(use_cpu_microbatch)
        self._use_cpu_microbatch_eval = (
            bool(use_cpu_microbatch_eval)
            if use_cpu_microbatch_eval is not None
            else self.use_cpu_microbatch
        )
        self.max_examples_per_microbatch = max_examples_per_microbatch
        self.max_eval_tokens_per_microbatch = (
            int(max_eval_tokens_per_microbatch)
            if max_eval_tokens_per_microbatch is not None
            else self.max_tokens_per_microbatch
        )
        # Preserve the user's initial eval budget so evaluate() can restore it
        # (training regime management may ratchet it down via _apply_regime_limits)
        self._initial_eval_tokens_per_microbatch = self.max_eval_tokens_per_microbatch
        self.eval_mode = self._normalize_eval_mode(eval_mode)
        self.debug = debug
        # --- Decoder cost multiplier (alpha) ---
        # Auto-detect from model vocab_size: large vocabs (e.g. Gemma2 256K)
        # make the lm_head output tensor dominant, so decoder tokens cost much
        # more than encoder tokens. alpha=2.0 is fine for ≤64K vocab.
        self.decoder_cost_multiplier = self._resolve_decoder_cost_multiplier(decoder_cost_multiplier)

        self.oom_max_retries = int(oom_max_retries)
        self.oom_shrink_B = float(oom_shrink_B)
        self.oom_shrink_tokens = float(oom_shrink_tokens)
        self.oom_min_B = int(oom_min_B)
        self.oom_min_tokens = int(oom_min_tokens)
        self.oom_skip_batch_on_fail = bool(oom_skip_batch_on_fail)
        self.sampler_bucket_size = int(sampler_bucket_size)
        self.sampler_shuffle_mode = str(sampler_shuffle_mode)
        self.max_microbatches_per_step = int(max_microbatches_per_step)

        # --- OOM tracking ---
        self._oom_events = 0           # Total OOM events across all batches
        self._oom_skipped_batches = 0  # Batches skipped after exhausting retries

        # --- VRAM-aware adaptive defaults ---
        vram_gb = self._detect_vram_gb()
        self._vram_gb = vram_gb

        # --- Adaptive Regime Management ---
        # Regimes group sequences by length to learn stable microbatch limits.
        # Key = bucket of max effective length, Value = {"B": batch_size, "T": token_budget, "stable": success_count}
        self._regime_limits = {}
        self._regime_bucket_size = 128   # Bucket width in tokens (groups similar lengths)
        self._regime_ramp_every = self._vram_default(regime_ramp_every, self._vram_ramp_every, int)
        self._regime_ramp_B = self._vram_default(regime_ramp_B, self._vram_ramp_B, float)
        self._regime_ramp_T = self._vram_default(regime_ramp_T, self._vram_ramp_T, float)
        self._autotune_safety = self._vram_default(autotune_safety, self._vram_autotune_safety, float)
        self._autotune_oom_safety = self._vram_default(autotune_oom_safety, self._vram_autotune_oom_safety, float)
        self._eval_ramp_every = self._vram_default(eval_ramp_every, self._vram_eval_ramp_every, int)
        self._eval_ramp_T = self._vram_default(eval_ramp_T, self._vram_eval_ramp_T, float)
        self._regime_min_B = getattr(self, "oom_min_B", 1)
        self._regime_min_T = getattr(self, "oom_min_tokens", 64)

        if self.debug or vram_gb > 0:
            logger.info(
                f"[TokenPackTrainer] VRAM={vram_gb:.0f}GB — regime_ramp_every={self._regime_ramp_every}, "
                f"ramp_B={self._regime_ramp_B:.2f}, ramp_T={self._regime_ramp_T:.2f}, "
                f"autotune_safety={self._autotune_safety:.2f}, oom_safety={self._autotune_oom_safety:.2f}, "
                f"eval_ramp_every={self._eval_ramp_every}, eval_ramp_T={self._eval_ramp_T:.2f}"
            )

        # Initial values for new regimes (from user config)
        self._regime_default_B = self.max_examples_per_microbatch
        self._regime_default_T = self.max_tokens_per_microbatch

        # --- Seed regime state from a previous run ---
        # Loads calibrated B/T values as starting points, avoiding cold-start
        # calibration (which can take thousands of OOM events for large-vocab models).
        # Autotuned keys are cleared so each regime re-autotunes T with the current
        # model's memory baseline — only the B/T starting points are reused.
        self._seed_regime_state_path = seed_regime_state
        self._adaptive_regime_state = bool(adaptive_regime_state)
        if seed_regime_state is not None:
            self._seed_regime_state(str(seed_regime_state))
        elif self._adaptive_regime_state:
            logger.warning("[TokenPackTrainer] adaptive_regime_state=True has no effect without seed_regime_state")
            self._adaptive_regime_state = False

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

        # Cache vocab_size for pre-flight lm_head check.  Only enable for
        # large vocabs (>64K) where the lm_head output can dominate VRAM.
        # For standard T5/mT5 (≤32K vocab) this stays 0 and the check is
        # skipped entirely — no GPU sync overhead in the hot loop.
        _vs = 0
        _cfg = getattr(getattr(self, "model", None), "config", None)
        if _cfg is not None:
            _vs = getattr(_cfg, "vocab_size", 0)
        self._lm_head_vocab_size = _vs if _vs > 64_000 else 0

        # Baseline GPU memory (model + optimizer, before any batch).
        # Captured lazily on the first training_step call.
        self._gpu_baseline_mem: int | None = None
        self._gpu_baseline_mem_per_device: dict[int, int] = {}  # per-device baselines for multi-GPU

        # --- Diagnostics file ---
        # Writes a JSONL file every N steps with GPU/CPU metrics, timing
        # breakdown, regime state, and batch stats.  No stdout clutter.
        # Clearly distinguishes "small T" from "dataloader starvation".
        self._diag_interval = int(diagnostic_interval)
        self._diag_file = None  # opened lazily on first write
        self._last_step_end_time = None  # for inter-step gap measurement

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

    def _resolve_decoder_cost_multiplier(self, value) -> float:
        """Resolve decoder_cost_multiplier: auto-detect from vocab_size or use explicit value."""
        if value != "auto":
            return float(value)

        # Auto-detect from model's vocab_size
        vocab_size = 0
        model = getattr(self, "model", None)
        if model is not None:
            config = getattr(model, "config", None)
            if config is not None:
                vocab_size = getattr(config, "vocab_size", 0)

        if vocab_size <= 0:
            alpha = 2.0
        elif vocab_size <= 64_000:
            alpha = 2.0      # Standard T5/mT5 (~32K vocab)
        elif vocab_size <= 128_000:
            alpha = 4.0      # Large vocab
        else:
            # Very large vocab (e.g. Gemma2 256K): lm_head output dominates
            # Scale roughly as sqrt(vocab / 32K) to avoid being too aggressive
            alpha = min(16.0, 2.0 * (vocab_size / 32_000) ** 0.5)

        if vocab_size > 64_000:
            print(
                f"[TokenPackTrainer] Auto-detected vocab_size={vocab_size:,}, "
                f"setting decoder_cost_multiplier={alpha:.1f} "
                f"(override with decoder_cost_multiplier=<float>)"
            )
        return alpha

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
        safety: float = 0.85,
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

    # ------------------------------------------------------------------
    # VRAM-aware adaptive defaults
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_vram_gb() -> float:
        """Return VRAM in GB for tuning adaptive parameters.

        For multi-GPU setups (model parallel / device_map), returns the
        *minimum* VRAM across all devices, since the bottleneck GPU
        determines the safe token budget.
        """
        try:
            import torch
            if torch.cuda.is_available():
                n = torch.cuda.device_count()
                if n == 0:
                    return 0.0
                vrams = [
                    torch.cuda.get_device_properties(i).total_memory / (1 << 30)
                    for i in range(n)
                ]
                return min(vrams)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _vram_default(user_value, auto_fn, cast):
        """Resolve 'auto' to a VRAM-scaled default, or use the user's explicit value."""
        if isinstance(user_value, str) and user_value.lower() == "auto":
            return cast(auto_fn())
        return cast(user_value)

    def _vram_ramp_every(self) -> int:
        """Ramp every N successes. Larger VRAM → more aggressive (fewer successes needed).
        >=80GB: 2,  48-80GB: 3,  24-48GB: 4,  <24GB: 6"""
        v = self._vram_gb
        if v >= 80:   return 2
        if v >= 48:   return 3
        if v >= 24:   return 4
        return 6

    def _vram_ramp_B(self) -> float:
        """B ramp factor. Larger VRAM → more aggressive ramp.
        >=80GB: 1.4,  48-80GB: 1.30,  24-48GB: 1.25,  <24GB: 1.15"""
        v = self._vram_gb
        if v >= 80:   return 1.40
        if v >= 48:   return 1.30
        if v >= 24:   return 1.25
        return 1.15

    def _vram_ramp_T(self) -> float:
        """T ramp factor. Larger VRAM → more aggressive ramp.
        >=80GB: 1.20,  48-80GB: 1.15,  24-48GB: 1.10,  <24GB: 1.05"""
        v = self._vram_gb
        if v >= 80:   return 1.20
        if v >= 48:   return 1.15
        if v >= 24:   return 1.10
        return 1.05

    def _vram_autotune_safety(self) -> float:
        """Normal autotune safety factor — targets this fraction of available
        VRAM (total minus model/optimizer baseline) for batch data.
        Previous values (0.85-0.92) pushed total utilization to 95-98%,
        causing CUDA allocator pressure without triggering OOM.
        >=80GB: 0.75,  48-80GB: 0.72,  24-48GB: 0.70,  <24GB: 0.66"""
        v = self._vram_gb
        if v >= 80:   return 0.75
        if v >= 48:   return 0.72
        if v >= 24:   return 0.70
        return 0.66

    def _vram_autotune_oom_safety(self) -> float:
        """Post-OOM autotune safety factor — more conservative than normal.
        >=80GB: 0.66,  48-80GB: 0.60,  24-48GB: 0.55,  <24GB: 0.50"""
        v = self._vram_gb
        if v >= 80:   return 0.66
        if v >= 48:   return 0.60
        if v >= 24:   return 0.55
        return 0.50

    def _vram_eval_ramp_every(self) -> int:
        """Eval ramp every N successes. Larger VRAM → faster recovery.
        >=80GB: 10,  48-80GB: 20,  24-48GB: 30,  <24GB: 50"""
        v = self._vram_gb
        if v >= 80:   return 10
        if v >= 48:   return 20
        if v >= 24:   return 30
        return 50

    def _vram_eval_ramp_T(self) -> float:
        """Eval T ramp factor. Larger VRAM → more aggressive.
        >=80GB: 1.10,  48-80GB: 1.08,  24-48GB: 1.05,  <24GB: 1.03"""
        v = self._vram_gb
        if v >= 80:   return 1.10
        if v >= 48:   return 1.08
        if v >= 24:   return 1.05
        return 1.03

    # ------------------------------------------------------------------

    def _regime_key_from_lengths(self, enc_len, dec_len) -> int:
        """Compute regime key from precomputed lengths (avoids redundant recomputation)."""
        alpha = self.decoder_cost_multiplier
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
            st = {"B": self._regime_default_B, "T": self._regime_default_T, "stable": 0,
                  "hwm_T": None, "hwm_B": None, "bytes_per_token": None}
            # B=None is valid and means "uncapped" — only default T if missing
            if st["T"] is None:
                st["T"] = 400
            self._regime_limits[key] = st
        # Backfill fields for regimes created before these fields existed
        if "hwm_T" not in st:
            st["hwm_T"] = None
        if "hwm_B" not in st:
            st["hwm_B"] = None
        if "bytes_per_token" not in st:
            st["bytes_per_token"] = None
        if "adaptive_ramps" not in st:
            st["adaptive_ramps"] = 0
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
        # NOTE: do NOT clamp max_eval_tokens_per_microbatch here.
        # Training regimes vary per sequence-length bucket; clamping eval to
        # the current regime's T ratchets the eval budget down to the minimum
        # T across all regimes (a one-way trip). Eval budgets are managed
        # independently — restored at the start of each evaluate() call.

    INT64_MAX = (1 << 63) - 1

    @staticmethod
    def _clamp_int(x: int, lo: int, hi: int) -> int:
        return max(int(lo), min(int(x), int(hi)))

    def _regime_on_success(self, key: int):
        st = self._regime_state(key)
        st["stable"] += 1

        # Update high-water marks after consecutive successes confirm stability.
        # These are the "proven safe operating point" — the (B, T) that ran
        # reliably.  On OOM, the regime falls back HERE, not to minimums.
        if st["stable"] >= 5:
            hwm_t = st.get("hwm_T") or 0
            if st["T"] > hwm_t:
                st["hwm_T"] = st["T"]
            if st["B"] is not None:
                hwm_b = st.get("hwm_B") or 0
                if st["B"] > hwm_b:
                    st["hwm_B"] = st["B"]

        hwm_t = st.get("hwm_T")
        hwm_b = st.get("hwm_B")

        # Fast recovery ramp: when B or T is well below HWM (post-OOM),
        # ramp toward the proven-stable level.
        _recovering = False
        if hwm_t and st["T"] < int(hwm_t * 0.9):
            _recovering = True
        if hwm_b and st["B"] is not None and st["B"] < int(hwm_b * 0.9):
            _recovering = True

        if _recovering:
            if st["stable"] % 5 == 0:  # every 5 successes
                if hwm_t and st["T"] < hwm_t:
                    new_T = int(st["T"] * 1.20)
                    st["T"] = min(new_T, hwm_t)
                    st["T"] = self._clamp_int(st["T"], self._regime_min_T, min(self._regime_max_T, self.INT64_MAX))
                if st["B"] is not None and hwm_b and st["B"] < hwm_b:
                    new_B = max(self._regime_min_B, int(st["B"] * 1.20) + 1)
                    st["B"] = min(new_B, hwm_b)
                    st["B"] = self._clamp_int(st["B"], self._regime_min_B, self._regime_max_B)
            return

        # Adaptive exploration: when seeded in adaptive mode, cautiously probe
        # beyond the seed's operating point to discover better limits for the
        # new data distribution.  Each regime gets a limited number of ramp
        # attempts; OOM terminates exploration immediately.
        adaptive_ramps = st.get("adaptive_ramps", 0)
        if adaptive_ramps > 0 and st["stable"] >= self._regime_ramp_every:
            old_T, old_B = st["T"], st["B"]
            st["T"] = self._clamp_int(
                int(st["T"] * self._regime_ramp_T),
                self._regime_min_T,
                min(self._regime_max_T, self.INT64_MAX),
            )
            if st["B"] is not None:
                st["B"] = self._clamp_int(
                    max(self._regime_min_B, int(st["B"] * self._regime_ramp_B) + 1),
                    self._regime_min_B,
                    self._regime_max_B,
                )
            st["adaptive_ramps"] = adaptive_ramps - 1
            st["stable"] = 0
            if self.debug:
                logger.info(
                    f"[TokenPackTrainer] Adaptive ramp regime {key}: "
                    f"T={old_T}→{st['T']}, B={old_B}→{st['B']}, "
                    f"ramps_left={st['adaptive_ramps']}"
                )
            return

        # --- GPU utilization-aware ramp ---
        # If the GPU is consistently underutilized, the regime is likely too
        # conservative (the autotune may have set T too low due to safety
        # margins or the eff_tokens denominator bug).  Cautiously ramp T
        # when utilization is low — if it OOMs, normal recovery handles it.
        #
        # Conversely, if utilization is ≥80% for 20+ steps, the regime is
        # at a good operating point — leave it alone.
        _util_history = getattr(self, "_gpu_util_history", [])
        if len(_util_history) >= 20 and st["stable"] >= 20:
            _avg_util = sum(_util_history[-20:]) / 20

            if _avg_util >= 80:
                # GPU is well-utilized — confirmed good operating point.
                # Reset the utilization ramp counter (we found a good T).
                st["_util_ramp_attempts"] = 0
                return

            if _avg_util < 50:
                # GPU is underutilized — cautiously ramp T.
                # Cap at 10 consecutive ramps to prevent runaway escalation
                # (e.g., if low utilization is from dataloader starvation,
                # not conservative T, ramping T won't help).
                _ramp_attempts = st.get("_util_ramp_attempts", 0)
                if _ramp_attempts < 10:
                    old_T = st["T"]
                    st["T"] = self._clamp_int(
                        int(st["T"] * 1.05),   # 5% ramp
                        self._regime_min_T,
                        min(self._regime_max_T, self.INT64_MAX),
                    )
                    if st["T"] != old_T:
                        st["stable"] = 0  # wait for stability at new T
                        st["_util_ramp_attempts"] = _ramp_attempts + 1
                        if self.debug:
                            logger.info(
                                f"[TokenPackTrainer] GPU util ramp regime {key}: "
                                f"T={old_T}→{st['T']} (5% ramp, "
                                f"avg_util={_avg_util:.0f}%, "
                                f"attempt {_ramp_attempts + 1}/10)"
                            )
                return

        return

    def _safe_floor_T(self) -> int:
        """Compute a dynamic minimum T that maintains useful GPU throughput.

        Uses the learned bytes_per_token and VRAM headroom to compute the
        token budget that fills ~66% of available VRAM.  This is the "known
        steady-state" — large enough to keep the GPU busy, conservative
        enough to never cause OOM or memory pressure.

        Falls back to oom_min_tokens when bytes_per_token isn't known yet.
        """
        # Find best bytes_per_token across all regimes (any calibration is
        # better than none — the per-token cost varies by regime but is
        # the same order of magnitude)
        bpt = None
        for st in self._regime_limits.values():
            b = st.get("bytes_per_token")
            if b is not None and b > 0:
                if bpt is None or b > bpt:
                    bpt = b  # use the most conservative (highest) estimate

        if bpt is None or not torch.cuda.is_available():
            return self._regime_min_T

        baselines = self._gpu_baseline_mem_per_device
        if not baselines:
            baseline = self._gpu_baseline_mem or 0
            total = int(self._vram_gb * (1 << 30))
        else:
            # Use bottleneck device
            baseline = 0
            total = 0
            best_headroom = None
            for dev_i, base_i in baselines.items():
                tot_i = torch.cuda.get_device_properties(dev_i).total_memory
                head_i = tot_i - base_i
                if best_headroom is None or head_i < best_headroom:
                    best_headroom = head_i
                    baseline = base_i
                    total = tot_i

        available = max(total - baseline, total * 0.05)
        # Target 66% of available VRAM — a comfortable level that keeps the
        # GPU busy without risking pressure or fragmentation
        safe_T = int(0.66 * available / max(bpt, 1e-9))
        return max(self._regime_min_T, safe_T)

    def _regime_on_oom(self, key: int, shrink_b: bool = True):
        """Shrink regime limits after an OOM event.

        Args:
            key: regime key
            shrink_b: if True, try shrinking B first (default). If False,
                skip B and go straight to T.  The retry loop sets this to
                False after the first retry so that B is only shrunk once
                per failing batch — preventing cascades like B=457 → B=7
                from 12 consecutive shrinks across 3 failing batches.
        """
        st = self._regime_state(key)

        # Before resetting stable, capture HWMs if this regime was working.
        # These define the "proven stable operating point" to fall back to.
        if st["stable"] >= 3:
            if st["T"] > (st.get("hwm_T") or 0):
                st["hwm_T"] = st["T"]
            if st["B"] is not None and st["B"] > (st.get("hwm_B") or 0):
                st["hwm_B"] = st["B"]

        st["stable"] = 0
        st["adaptive_ramps"] = 0  # OOM ends adaptive exploration for this regime
        st["_util_ramp_attempts"] = 0  # reset utilization ramp counter (can try again after recovery)

        # Invalidate bytes_per_token — the calibration that said "this T is safe"
        # was wrong, so don't let the predictive check trust it until re-autotune.
        st["bytes_per_token"] = None

        # Track OOM count per regime for safety factor adjustment on re-autotune
        oom_history = getattr(self, "_regime_oom_count", {})
        oom_history[key] = oom_history.get(key, 0) + 1
        self._regime_oom_count = oom_history

        # Allow re-autotuning after recovery stabilizes.
        autotuned = getattr(self, "_autotuned_keys", set())
        if key in autotuned:
            autotuned.discard(key)

        safe_floor = self._safe_floor_T()
        hwm_b = st.get("hwm_B")
        hwm_t = st.get("hwm_T")

        # --- Shrink B (only on first retry per batch) ---
        if shrink_b:
            if st["B"] is None:
                st["B"] = self._regime_max_B
                return

            # Fall back toward hwm_B (proven stable), not toward minimums.
            # If above hwm_B: drop to hwm_B.
            # If at/near hwm_B: drop to 85% of hwm_B.
            # If below hwm_B or no hwm: proportional 0.7x shrink.
            if hwm_b and st["B"] > hwm_b:
                st["B"] = hwm_b
            elif hwm_b and st["B"] > int(hwm_b * 0.85):
                st["B"] = max(self._regime_min_B, int(hwm_b * 0.85))
            else:
                st["B"] = max(self._regime_min_B, int(st["B"] * 0.7))
            return

        # --- Shrink T (on subsequent retries, or when B can't shrink) ---
        floor = max(safe_floor, self._regime_min_T)
        if hwm_t and st["T"] > hwm_t:
            st["T"] = max(floor, hwm_t)
        elif hwm_t and st["T"] > int(hwm_t * 0.85):
            st["T"] = max(floor, int(hwm_t * 0.85))
        else:
            st["T"] = max(floor, int(st["T"] * 0.85))

    def get_regime_stats(self) -> dict:
        """
        Return current adaptive regime limits for debugging and monitoring.

        Returns a dict with:
            - regimes: dict mapping regime_key → {"B": int, "T": int, "stable": int, "hwm_T": int|None, "bytes_per_token": float|None}
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
            "vram_gb": getattr(self, "_vram_gb", 0),
            "regime_ramp_every": self._regime_ramp_every,
            "regime_ramp_B": self._regime_ramp_B,
            "regime_ramp_T": self._regime_ramp_T,
            "autotune_safety": self._autotune_safety,
            "autotune_oom_safety": self._autotune_oom_safety,
            "eval_ramp_every": self._eval_ramp_every,
            "eval_ramp_T": self._eval_ramp_T,
            "max_microbatches_per_step": self.max_microbatches_per_step,
            "adaptive_regime_state": getattr(self, "_adaptive_regime_state", False),
            "gpu_util_history": list(getattr(self, "_gpu_util_history", [])),
            "gpu_util_avg": (
                sum(getattr(self, "_gpu_util_history", [])) / len(getattr(self, "_gpu_util_history", [1]))
                if getattr(self, "_gpu_util_history", []) else None
            ),
        }

    # ------------------------------------------------------------------
    # Diagnostics file writer
    # ------------------------------------------------------------------

    def _write_step_diagnostics(
        self,
        step: int,
        regime_key,
        plan_ms: float,
        exec_ms: float,
        total_ms: float,
        inter_step_gap_ms: float | None,
        num_examples: int,
        num_tokens: int,
        eff_tokens: int,
        num_microbatches: int,
        mb_eff_tokens: list | None = None,
    ):
        """Write one diagnostic record to {output_dir}/tokenpack_diagnostics.jsonl.

        Called every diagnostic_interval steps.  Captures GPU/CPU state,
        timing breakdown, and regime state in a single JSON line.
        No stdout output — file only.
        """
        import json

        if self._diag_interval <= 0:
            return

        # Lazy open
        if self._diag_file is None:
            diag_path = os.path.join(self.args.output_dir, "tokenpack_diagnostics.jsonl")
            os.makedirs(self.args.output_dir, exist_ok=True)
            self._diag_file = open(diag_path, "a", buffering=1)  # line-buffered

        record = {
            "step": step,
            "timestamp": time.time(),
        }

        # --- GPU metrics ---
        if torch.cuda.is_available():
            dev = self.args.device
            dev_idx = dev.index if hasattr(dev, "index") and dev.index is not None else 0
            try:
                record["gpu_utilization_pct"] = torch.cuda.utilization(dev)
            except Exception:
                record["gpu_utilization_pct"] = None
            record["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(dev) / (1 << 30)
            record["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved(dev) / (1 << 30)
            record["gpu_memory_peak_gb"] = torch.cuda.max_memory_allocated(dev) / (1 << 30)
            try:
                props = torch.cuda.get_device_properties(dev)
                record["gpu_memory_total_gb"] = props.total_memory / (1 << 30)
            except Exception:
                pass
            # Power and temperature via pynvml (optional)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(dev_idx)
                record["gpu_power_w"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                record["gpu_temp_c"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                pass

        # --- CPU metrics ---
        try:
            record["cpu_load_avg_1m"] = os.getloadavg()[0]
        except (OSError, AttributeError):
            pass
        try:
            import psutil
            record["cpu_percent"] = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            record["ram_used_gb"] = mem.used / (1 << 30)
            record["ram_total_gb"] = mem.total / (1 << 30)
            record["ram_percent"] = mem.percent
        except ImportError:
            pass

        # --- Timing ---
        record["plan_ms"] = round(plan_ms, 1)
        record["exec_ms"] = round(exec_ms, 1)
        record["total_step_ms"] = round(total_ms, 1)
        record["inter_step_gap_ms"] = round(inter_step_gap_ms, 1) if inter_step_gap_ms is not None else None
        # Derived: what fraction of wall time is the GPU actually computing?
        if inter_step_gap_ms is not None and total_ms > 0:
            wall = total_ms + inter_step_gap_ms
            record["gpu_busy_pct_walltime"] = round(100.0 * exec_ms / wall, 1)
        else:
            record["gpu_busy_pct_walltime"] = None

        # --- Batch stats ---
        record["num_examples"] = num_examples
        record["num_tokens"] = num_tokens
        record["eff_tokens"] = eff_tokens
        record["num_microbatches"] = num_microbatches
        if mb_eff_tokens:
            record["max_mb_eff_tokens"] = max(mb_eff_tokens)
            record["min_mb_eff_tokens"] = min(mb_eff_tokens)

        # --- Regime state ---
        if regime_key is not None:
            st = self._regime_state(regime_key)
            record["regime_key"] = regime_key
            record["regime_T"] = st.get("T")
            record["regime_B"] = st.get("B")
            record["regime_hwm_T"] = st.get("hwm_T")
            record["regime_stable"] = st.get("stable")
            record["regime_bytes_per_token"] = st.get("bytes_per_token")
            record["regime_util_ramp_attempts"] = st.get("_util_ramp_attempts", 0)

        # --- GPU utilization rolling average ---
        _uh = getattr(self, "_gpu_util_history", [])
        if _uh:
            record["gpu_util_avg_20"] = round(sum(_uh) / len(_uh), 1)
        record["oom_events_total"] = getattr(self, "_oom_events", 0)

        try:
            self._diag_file.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def _bottleneck_gpu_stats(self) -> tuple[int, int, int] | None:
        """Return (total, peak, baseline) for the GPU with least headroom.

        For model-parallel setups the OOM bottleneck is whichever GPU has
        the smallest (total - peak) gap.  Querying only self.args.device
        misses layers on other GPUs and leads to over-aggressive autotuning.
        """
        if not torch.cuda.is_available():
            return None

        n = torch.cuda.device_count()
        baselines = getattr(self, "_gpu_baseline_mem_per_device", {})

        best = None       # (headroom, total, peak, baseline)
        for i in range(n):
            total_i = torch.cuda.get_device_properties(i).total_memory
            peak_i  = torch.cuda.max_memory_allocated(i)
            base_i  = baselines.get(i, 0)
            if peak_i <= 0:
                continue  # device not used
            headroom = total_i - peak_i
            if best is None or headroom < best[0]:
                best = (headroom, total_i, peak_i, base_i)

        if best is None:
            # Fall back to self.args.device
            dev = self.args.device
            total = torch.cuda.get_device_properties(dev.index if hasattr(dev, 'index') else 0).total_memory
            peak  = torch.cuda.max_memory_allocated(dev)
            baseline = self._gpu_baseline_mem or 0
            return (total, peak, baseline)

        return (best[1], best[2], best[3])

    def _autotune_regime_from_peak(self, key: int, eff_tokens_in_step: int, safety: float = 0.90):
        if not torch.cuda.is_available() or eff_tokens_in_step <= 0:
            return

        stats = self._bottleneck_gpu_stats()
        if stats is None:
            return
        total, peak, baseline = stats

        # avoid divide-by-zero / nonsense
        if peak <= 0:
            return

        # Use baseline memory (model + optimizer) to compute MARGINAL per-token cost.
        # Without this, model weights (~90% of VRAM) inflate per-token cost by 10x+,
        # making the autotuned token budget far too conservative.
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

        # Capture hwm_T immediately when autotune sets T.
        # Previously hwm_T was only set in _regime_on_success after stable >= 3,
        # meaning an OOM within the first 3 steps would lose the pre-OOM T.
        # The post-OOM re-autotune (with conservative safety=0.60) would then
        # become the hwm_T, permanently capping the regime.
        if st["T"] > (st.get("hwm_T") or 0):
            st["hwm_T"] = st["T"]

        # Persist bytes_per_token for the predictive VRAM pre-flight check.
        # This lets the next step predict memory before sending data to GPU.
        st["bytes_per_token"] = bytes_per_eff_token

        if self.debug:
            print(
                f"[autotune] regime {key}: baseline={baseline/(1<<30):.1f}G, "
                f"peak={peak/(1<<30):.1f}G, batch_mem={batch_mem/(1<<30):.1f}G, "
                f"bytes/tok={bytes_per_eff_token:.0f}, eff_tokens={eff_tokens_in_step}, "
                f"target_tokens={target_eff_tokens}, T: {old_T} -> {st['T']}, "
                f"hwm_T={st.get('hwm_T')}"
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

        # Free CUDA memory quickly: empty cache first (fast), then do a
        # lightweight gc pass. Full gc.collect() can take 10+ seconds after
        # a massive failed allocation and is interruptible by Ctrl+C.
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        try:
            import gc
            gc.collect(0)  # generation-0 only — fast, catches recent allocations
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
                    changed = True

        return changed

    def _estimate_lm_head_bytes(self, dec_tokens: int) -> int:
        """Estimate the lm_head output tensor size in bytes for a microbatch.

        Only called for large-vocab models (gated by _lm_head_vocab_size).
        Uses pre-computed dec_tokens (B * dec_max, computed on CPU before GPU
        transfer) to avoid any GPU synchronization inside the hot loop.
        """
        vocab_size = self._lm_head_vocab_size
        if vocab_size <= 0 or dec_tokens <= 0:
            return 0

        # bf16/fp16 = 2 bytes, fp32 = 4 bytes
        dtype_bytes = 2 if (self.args.bf16 or self.args.fp16) else 4
        return dec_tokens * vocab_size * dtype_bytes

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

    @staticmethod
    def _concat_eval_batches(batches: list) -> dict:
        """Concatenate a list of batch dicts along the batch (first) dimension.

        Handles variable sequence lengths across batches (from dynamic padding)
        by right-padding shorter batches to the maximum length before concat.
        """
        if len(batches) == 1:
            return batches[0]
        merged = {}
        for k in batches[0]:
            vals = [b[k] for b in batches if k in b]
            if not vals:
                continue
            if isinstance(vals[0], torch.Tensor) and vals[0].ndim >= 1:
                # Check if non-batch dimensions match across all batches
                shapes = [v.shape[1:] for v in vals]
                if all(s == shapes[0] for s in shapes):
                    merged[k] = torch.cat(vals, dim=0)
                else:
                    # Dynamic padding: right-pad shorter batches to max length
                    max_shape = list(shapes[0])
                    for s in shapes[1:]:
                        for i, d in enumerate(s):
                            max_shape[i] = max(max_shape[i], d)
                    # labels pad with -100 (ignore index), everything else with 0
                    pad_val = -100 if k == "labels" else 0
                    padded = []
                    for v in vals:
                        if list(v.shape[1:]) == max_shape:
                            padded.append(v)
                        else:
                            new_shape = [v.shape[0]] + max_shape
                            p = torch.full(new_shape, pad_val, dtype=v.dtype, device=v.device)
                            slices = [slice(None)] + [slice(0, s) for s in v.shape[1:]]
                            p[slices] = v
                            padded.append(p)
                    merged[k] = torch.cat(padded, dim=0)
            else:
                merged[k] = vals[0]
        return merged

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

        # lengths used by sampler — compute EFFECTIVE lengths (enc + α × dec)
        # so the sampler's token budget matches the microbatch planner's cost model.
        # Without this, the sampler packs by encoder-only lengths, creating batches
        # that are much larger than the microbatch budget when decoder cost is high
        # (e.g. α=5.66 for 256K vocab), resulting in 20-30 microbatches per step.
        if hasattr(ds, "column_names"):
            if self.length_column_name not in ds.column_names:
                raise ValueError(
                    f"Column '{self.length_column_name}' doesn't exist in train_dataset. "
                    f"Available columns: {ds.column_names}"
                )
            raw_lengths = ds[self.length_column_name]
        else:
            raw_lengths = [ds[i][self.length_column_name] for i in range(len(ds))]

        enc_cap = self.max_encoder_len
        dec_cap = self.max_decoder_len
        alpha = self.decoder_cost_multiplier

        # Try to get decoder lengths from labels column (vectorized for speed)
        dec_lengths = None
        if alpha > 1.0:
            try:
                import numpy as np
                if hasattr(ds, "column_names") and "labels" in ds.column_names:
                    all_labels = ds["labels"]
                    # HF datasets return list-of-lists; try numpy for speed
                    try:
                        # Fixed-length labels (all same shape) — fast path
                        lab_arr = np.array(all_labels)
                        dec_lengths = (lab_arr != -100).sum(axis=-1)
                        if dec_cap is not None:
                            dec_lengths = np.minimum(dec_lengths, dec_cap)
                        dec_lengths = dec_lengths.tolist()
                    except (ValueError, TypeError):
                        # Ragged labels — per-row numpy, still much faster than pure Python
                        dec_lengths = []
                        for lab in all_labels:
                            arr = np.asarray(lab)
                            dlen = int((arr != -100).sum())
                            if dec_cap is not None:
                                dlen = min(dlen, dec_cap)
                            dec_lengths.append(dlen)
            except Exception as e:
                logger.warning(f"[TokenPackTrainer] Could not compute decoder lengths for sampler: {e}")
                dec_lengths = None

        if dec_lengths is not None:
            lengths_for_sampler = []
            for enc_L, dec_L in zip(raw_lengths, dec_lengths):
                enc = min(int(enc_L), enc_cap) if enc_cap is not None else int(enc_L)
                eff = int(enc + alpha * dec_L)
                lengths_for_sampler.append(eff)
            logger.info(
                f"[TokenPackTrainer] Sampler using effective lengths (enc + {alpha:.1f}×dec), "
                f"mean_eff={sum(lengths_for_sampler)/max(len(lengths_for_sampler),1):.0f}"
            )
        else:
            if enc_cap is not None:
                lengths_for_sampler = [min(int(L), enc_cap) for L in raw_lengths]
            else:
                lengths_for_sampler = [int(L) for L in raw_lengths]

        # max_length_in_batch: scale by alpha if using effective lengths,
        # so the sampler's length cap matches the microbatch planner's cap.
        sampler_max_len = None
        if enc_cap is not None:
            if dec_lengths is not None and dec_cap is not None:
                sampler_max_len = int(enc_cap + alpha * dec_cap)
            else:
                sampler_max_len = enc_cap

        batch_sampler = LengthBucketedBatchSampler(
            lengths=lengths_for_sampler,
            max_tokens_per_batch=self.max_tokens_per_batch,
            bucket_size=self.sampler_bucket_size,
            shuffle=True,
            drop_last=False,
            long_behavior="truncate",
            max_length_in_batch=sampler_max_len,
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
        max_examples_per_microbatch: int | None = None,
        return_lengths: bool = False,
        precomputed_lengths: tuple | None = None,
        generation_max_length: int | None = None,
    ):
        # Reuse precomputed lengths if available (avoids redundant tensor ops)
        if precomputed_lengths is not None:
            enc_len, dec_len = precomputed_lengths
        else:
            enc_len, dec_len, _ = self._compute_lengths_enc_dec(inputs)

        alpha = self.decoder_cost_multiplier
        # When planning microbatches for generation, use generation_max_length
        # instead of label length — KV cache grows to gen_max, not to dec_len.
        if generation_max_length is not None:
            effective_len = enc_len + alpha * generation_max_length
        else:
            effective_len = enc_len + alpha * dec_len
        N = int(effective_len.numel())

        budget = int(max_tokens_per_microbatch or self.max_tokens_per_microbatch)
        max_B = max_examples_per_microbatch if max_examples_per_microbatch is not None else self.max_examples_per_microbatch
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
        if self._use_cpu_microbatch_eval:
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
        # Use _to_device directly — our _prepare_input(s) overrides no-op
        # when use_cpu_microbatch=True (training flag), so super() wouldn't
        # transfer either since it calls self._prepare_input() per tensor.
        inputs = self._to_device(inputs)
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
    # Eval VRAM optimisation: free training-only memory for generation
    # --------------------------------------------------------------

    def _prepare_eval_vram(self) -> dict:
        """
        Maximise free VRAM before generation by:
          1. Disabling gradient checkpointing (pure overhead during inference —
             recomputes activations at every decode step for no benefit).
          2. Offloading optimizer states to CPU (AdamW momentum/variance can
             consume 2-4× model size in fp32).
          3. Clearing the CUDA cache so the freed blocks are visible to
             mem_get_info / the auto-tuner.

        Returns a state dict that _restore_eval_vram() uses to undo everything.
        """
        state: dict = {}

        # ---- 1) gradient checkpointing ------------------------------------
        model = self.model
        gc_was_enabled = getattr(model, "is_gradient_checkpointing", False)
        if not gc_was_enabled:
            # some models use a private flag instead
            gc_was_enabled = getattr(model.config, "gradient_checkpointing", False)
        state["gc_was_enabled"] = gc_was_enabled

        if gc_was_enabled:
            try:
                model.gradient_checkpointing_disable()
            except Exception:
                state["gc_was_enabled"] = False  # couldn't toggle, don't try to restore

        # ---- 2) optimizer state → CPU -------------------------------------
        moved_keys: list[tuple] = []
        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None:
            try:
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        pstate = optimizer.state.get(p)
                        if not pstate:
                            continue
                        for k, v in pstate.items():
                            if isinstance(v, torch.Tensor) and v.is_cuda:
                                pstate[k] = v.to("cpu", non_blocking=True)
                                moved_keys.append((id(p), k))
            except Exception:
                pass  # non-standard optimizer — skip silently
        state["moved_keys"] = moved_keys

        # ---- 3) flush cache -----------------------------------------------
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return state

    def _restore_eval_vram(self, state: dict):
        """Undo _prepare_eval_vram: re-enable grad-ckpt, move optimizer back."""

        # ---- gradient checkpointing ---------------------------------------
        if state.get("gc_was_enabled", False):
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass

        # ---- optimizer state → GPU ----------------------------------------
        moved_keys = set(state.get("moved_keys", []))
        if moved_keys:
            optimizer = getattr(self, "optimizer", None)
            if optimizer is not None:
                try:
                    device = self.args.device
                    for group in optimizer.param_groups:
                        for p in group["params"]:
                            pstate = optimizer.state.get(p)
                            if not pstate:
                                continue
                            for k, v in pstate.items():
                                if (id(p), k) in moved_keys and isinstance(v, torch.Tensor):
                                    pstate[k] = v.to(device, non_blocking=True)
                except Exception:
                    pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()

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

        Uses high-water marks as intelligent floors: if we're above the HWM,
        fall back to it first.  If at/near the HWM, nudge 15% below it.
        Only applies the blind proportional shrink as a last resort.

        Order: shrink generation batch size first (biggest VRAM consumer due
        to KV cache), then token budget, then microbatch examples.
        """
        self._eval_regime_init()
        changed = False

        # --- 1. Shrink generation batch size (gen_B) first ---
        cur_gen_B = getattr(self, "max_eval_generate_examples", None)
        hwm_gen_B = self._eval_hwm_gen_B
        if cur_gen_B is not None and cur_gen_B > 1:
            if hwm_gen_B and cur_gen_B > hwm_gen_B:
                # Above HWM → fall to proven-safe level
                self.max_eval_generate_examples = hwm_gen_B
                changed = True
            elif hwm_gen_B and cur_gen_B > int(hwm_gen_B * 0.85):
                # At/near HWM → nudge below it
                self.max_eval_generate_examples = max(1, int(hwm_gen_B * 0.85))
                changed = True
            else:
                # Below HWM or no HWM → proportional shrink
                new_gen_B = max(1, int(cur_gen_B * self.oom_shrink_B))
                if new_gen_B < cur_gen_B:
                    self.max_eval_generate_examples = new_gen_B
                    changed = True

        if changed:
            return True

        # --- 2. Shrink eval token budget (T) ---
        cur_T = self.max_eval_tokens_per_microbatch
        hwm_T = self._eval_hwm_T
        if cur_T is not None and cur_T > self.oom_min_tokens:
            if hwm_T and cur_T > hwm_T:
                # Above HWM → fall to proven-safe level
                new_T = max(self.oom_min_tokens, hwm_T)
                if new_T < cur_T:
                    self.max_eval_tokens_per_microbatch = new_T
                    changed = True
            elif hwm_T and cur_T > int(hwm_T * 0.85):
                # At/near HWM → nudge below it
                new_T = max(self.oom_min_tokens, int(hwm_T * 0.85))
                if new_T < cur_T:
                    self.max_eval_tokens_per_microbatch = new_T
                    changed = True
            else:
                # Below HWM or no HWM → proportional shrink
                new_T = max(self.oom_min_tokens, int(cur_T * self.oom_shrink_tokens))
                if new_T < cur_T:
                    self.max_eval_tokens_per_microbatch = new_T
                    changed = True

        if changed:
            return True

        # --- 3. Shrink microbatch examples (B) as last resort ---
        if self.max_examples_per_microbatch is not None and self.max_examples_per_microbatch > self.oom_min_B:
            new_B = max(self.oom_min_B, int(self.max_examples_per_microbatch * self.oom_shrink_B))
            if new_B < self.max_examples_per_microbatch:
                self.max_examples_per_microbatch = new_B
                changed = True

        return changed

    def _eval_regime_init(self):
        if not hasattr(self, "_eval_stable"):
            self._eval_stable = 0
        if not hasattr(self, "_eval_hwm_T"):
            self._eval_hwm_T = None   # proven-safe token budget
        if not hasattr(self, "_eval_hwm_gen_B"):
            self._eval_hwm_gen_B = None  # proven-safe generation batch size

    def _eval_on_success(self):
        """
        Called once per *successful eval batch* (i.e., the batch generated without OOM).
        Updates high-water marks after consecutive successes and ramps toward
        the proven-stable operating point when recovering from OOM.
        """
        self._eval_regime_init()
        self._eval_stable += 1

        cur_T = self.max_eval_tokens_per_microbatch
        cur_gen_B = getattr(self, "max_eval_generate_examples", None)

        # Update HWMs after enough consecutive successes confirm stability.
        if self._eval_stable >= 5:
            if cur_T is not None and cur_T > (self._eval_hwm_T or 0):
                self._eval_hwm_T = cur_T
            if cur_gen_B is not None and cur_gen_B > (self._eval_hwm_gen_B or 0):
                self._eval_hwm_gen_B = cur_gen_B

        # only ramp if we have a budget
        if cur_T is None:
            return

        hwm_T = self._eval_hwm_T
        hwm_gen_B = self._eval_hwm_gen_B

        # Fast recovery ramp: when T or gen_B is well below HWM (post-OOM),
        # ramp toward the proven-stable level instead of creeping up slowly.
        _recovering = False
        if hwm_T and cur_T < int(hwm_T * 0.9):
            _recovering = True
        if hwm_gen_B and cur_gen_B is not None and cur_gen_B < int(hwm_gen_B * 0.9):
            _recovering = True

        if _recovering:
            if self._eval_stable % 3 == 0:  # every 3 successes (faster than training)
                if hwm_T and cur_T < hwm_T:
                    new_T = min(int(cur_T * 1.20), hwm_T)
                    self.max_eval_tokens_per_microbatch = max(new_T, self.oom_min_tokens)
                if cur_gen_B is not None and hwm_gen_B and cur_gen_B < hwm_gen_B:
                    new_gen_B = min(int(cur_gen_B * 1.20) + 1, hwm_gen_B)
                    self.max_eval_generate_examples = max(1, new_gen_B)
            return  # don't also apply the normal ramp

        # Normal ramp (at or above HWM — regime is at its proven stable pace)
        ramp_every = self._eval_ramp_every
        ramp_T = self._eval_ramp_T
        max_T_cap = getattr(self, "max_tokens_per_microbatch", None)  # never exceed train budget if set

        if (self._eval_stable % ramp_every) == 0:
            new_T = int(cur_T * ramp_T)
            if max_T_cap is not None:
                new_T = min(new_T, int(max_T_cap))
            self.max_eval_tokens_per_microbatch = max(new_T, self.oom_min_tokens)

    def _eval_on_oom(self):
        """Capture HWMs (if stable enough) before resetting, so we have a
        proven-safe fallback after shrinking."""
        self._eval_regime_init()

        # Capture HWMs before resetting stable counter
        if self._eval_stable >= 3:
            cur_T = self.max_eval_tokens_per_microbatch
            cur_gen_B = getattr(self, "max_eval_generate_examples", None)
            if cur_T is not None and cur_T > (self._eval_hwm_T or 0):
                self._eval_hwm_T = cur_T
            if cur_gen_B is not None and cur_gen_B > (self._eval_hwm_gen_B or 0):
                self._eval_hwm_gen_B = cur_gen_B

        self._eval_stable = 0

    def _eval_bootstrap_hwm(self):
        """Record current limits as HWMs immediately (no stability requirement).

        Called after the first successful generation retry within an OOM loop,
        so the rest of the eval pass has a proven floor.  Normal
        _eval_on_success() still gates further HWM *increases* on 5 consec
        successes — this only bootstraps when no HWM exists yet.
        """
        self._eval_regime_init()
        cur_T = self.max_eval_tokens_per_microbatch
        cur_gen_B = getattr(self, "max_eval_generate_examples", None)
        if cur_T is not None and self._eval_hwm_T is None:
            self._eval_hwm_T = cur_T
        if cur_gen_B is not None and self._eval_hwm_gen_B is None:
            self._eval_hwm_gen_B = cur_gen_B

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

        # ---- free training-only VRAM (grad-ckpt, optimizer) for generation
        _vram_state = self._prepare_eval_vram() if do_generate else None

        # ---- auto-tune generation batch size from available VRAM
        # (runs AFTER _prepare_eval_vram so it sees the freed memory)
        # Re-estimate every eval call: available VRAM varies as optimizer
        # states are offloaded/restored between train and eval phases.
        _user_set_gen_B = getattr(self, "_user_set_gen_B", None)
        self._eval_regime_init()
        if do_generate:
            auto_B = self._estimate_max_gen_examples(gen_kwargs)
            if auto_B is not None:
                # Only override if user didn't explicitly set a value at init
                if _user_set_gen_B is None:
                    # Cap with HWM: the VRAM estimate is theoretical; the HWM
                    # is what actually ran without OOM.  Don't re-discover the
                    # OOM boundary every eval call.
                    hwm_gen_B = self._eval_hwm_gen_B
                    if hwm_gen_B is not None and auto_B > hwm_gen_B:
                        logger.info(
                            f"[TokenPackTrainer] Auto-tune gen_B={auto_B} exceeds "
                            f"HWM={hwm_gen_B}; capping at HWM"
                        )
                        auto_B = hwm_gen_B
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

        # Background decode: run batch_decode in a thread so the GPU can
        # proceed with the next batch's loss computation in parallel.
        _decode_threads: list[threading.Thread] = []

        def _drain_decode_threads():
            """Wait for all pending background decode threads to finish."""
            for t in _decode_threads:
                t.join()
            _decode_threads.clear()

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

            # Transfer to CPU synchronously (fast, frees GPU memory)
            pred_cpu = pred_g.cpu()
            lab_cpu = lab_g.cpu()

            # Decode + accumulate in a background thread so the GPU can
            # start the next loss / generate step immediately.
            def _decode_and_accumulate(pred_ids, lab_ids):
                preds_txt = self.processing_class.batch_decode(pred_ids, skip_special_tokens=True)
                refs_txt  = self.processing_class.batch_decode(lab_ids,  skip_special_tokens=True)

                preds = [p.strip() for p in preds_txt]
                refs  = [[r.strip()] for r in refs_txt]

                if len(preds) == 0:
                    return

                # metric_state is only touched by decode threads + main after
                # _drain_decode_threads(), so no lock needed.
                metric_state["metric_examples"] += len(preds)
                metric_state["all_preds"].extend(preds)
                metric_state["all_refs"].extend(refs)

            t = threading.Thread(target=_decode_and_accumulate, args=(pred_cpu, lab_cpu))
            t.start()
            _decode_threads.append(t)

        # Compute generation stride to evenly sample across the full eval set
        # (ensures all tasks/domains are represented in metrics, not just the first block)
        _gen_stride = 1
        if self.max_metric_samples is not None and do_generate:
            _n_eval = len(eval_dataset) if hasattr(eval_dataset, "__len__") else None
            if _n_eval is not None and _n_eval > self.max_metric_samples:
                _gen_stride = max(1, _n_eval // self.max_metric_samples)

        # ---- generation batching state (used by _flush_gen_buffer closure) ----
        _gen_buffer: list = []
        _gen_buffer_count = 0
        _gen_target_B = self.max_eval_generate_examples or 0  # 0 = no accumulation

        if _gen_target_B > 0 and do_generate:
            logger.info(
                f"[TokenPackTrainer] Gen batch accumulation: target={_gen_target_B} examples "
                f"(dataloader yields ~{self.args.per_device_eval_batch_size})"
            )

        # Get generation_max_length for microbatch budget (KV cache cost)
        _gen_max_len = (
            gen_kwargs.get("max_length")
            or gen_kwargs.get("max_new_tokens")
            or getattr(self.args, "generation_max_length", None)
        )

        def _flush_gen_buffer():
            """Concatenate accumulated batches and run generation."""
            nonlocal _gen_buffer, _gen_buffer_count
            # Ensure previous decode threads are done before starting new generation
            _drain_decode_threads()
            if not _gen_buffer:
                return

            super_batch = self._concat_eval_batches(_gen_buffer)
            _gen_buffer = []
            _gen_buffer_count = 0

            B = super_batch["input_ids"].size(0)
            gen_limit = getattr(self, "max_eval_generate_examples", None)
            fast_path_ok = (gen_limit is None or B <= gen_limit)

            # ---- fast path: generate whole super-batch at once ----
            _fast_path_oom = False
            if fast_path_ok:
                try:
                    gpu_batch = self._to_device(super_batch)
                    gen_ids = _do_generate(gpu_batch)
                    _add_streaming_metrics(gen_ids, gpu_batch["labels"])
                    self._eval_on_success()
                    return
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    if not self._is_cuda_oom(e):
                        raise
                    _fast_path_oom = True
                    self._eval_oom_cleanup()
                    # fall through to microbatching

            # ---- microbatch generation path (OOM-safe) ----
            last_err = None
            for attempt in range(int(self.oom_max_retries) + 1):
                try:
                    orig_use_cache = getattr(self.model.config, "use_cache", True)
                    self.model.config.use_cache = True

                    orig_max_B = self.max_examples_per_microbatch
                    if gen_limit is not None:
                        self.max_examples_per_microbatch = int(gen_limit)

                    microbatches = self._make_microbatches(
                        super_batch,
                        max_tokens_per_microbatch=self.max_eval_tokens_per_microbatch,
                        generation_max_length=_gen_max_len,
                    )
                    self.max_examples_per_microbatch = orig_max_B

                    for mb in microbatches:
                        mb_gpu = self._to_device(mb)
                        gen_ids = _do_generate(mb_gpu)
                        _add_streaming_metrics(gen_ids, mb_gpu["labels"])

                    last_err = None
                    self._eval_on_success()
                    # Bootstrap HWM immediately after any OOM recovery
                    # (fast-path fallthrough or microbatch retry) so
                    # subsequent flushes have a proven floor.
                    if attempt > 0 or _fast_path_oom:
                        self._eval_bootstrap_hwm()
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
                            "eval_gen_B_now": float(getattr(self, "max_eval_generate_examples", 0) or 0),
                            "eval_hwm_T": float(self._eval_hwm_T or 0),
                            "eval_hwm_gen_B": float(self._eval_hwm_gen_B or 0),
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
                else:
                    raise last_err

        start_time = time.time()

        # ================================================================
        # PHASE 1: Loss computation (fast forward passes, high GPU util)
        # ================================================================
        # Accumulate gen-eligible batches on the side; generation runs in
        # Phase 2 so the loss progress bar never stalls on generate().
        _all_gen_batches: list = []

        for batch in tqdm(dataloader, desc=f"{desc} [loss]", leave=True, disable=bool(getattr(self.args, "disable_tqdm", False))):
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
                if self._use_cpu_microbatch_eval:
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
                        mb = self._to_device(mb)
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

            # Stage batch for Phase 2 generation.  Store on CPU to free
            # GPU memory for loss computation of remaining batches.
            acc_batch = self._truncate_batch(batch)
            acc_batch = {
                k: v.cpu() if isinstance(v, torch.Tensor) and v.is_cuda else v
                for k, v in acc_batch.items()
            }
            _all_gen_batches.append(acc_batch)

        # ================================================================
        # PHASE 2: Generation (autoregressive decode, memory-bandwidth bound)
        # ================================================================
        if _all_gen_batches:
            # Build super-batches up to _gen_target_B, then flush each.
            _gen_buffer = []
            _gen_buffer_count = 0

            # Pre-count exact flushes by simulating the accumulation
            _sim_count = 0
            expected_flushes = 0
            for b in _all_gen_batches:
                _sim_count += b["input_ids"].size(0)
                if _gen_target_B <= 0 or _sim_count >= _gen_target_B:
                    expected_flushes += 1
                    _sim_count = 0
            if _sim_count > 0:
                expected_flushes += 1  # remainder flush

            gen_pbar = tqdm(
                total=expected_flushes, desc=f"{desc} [generate]", leave=True,
                disable=bool(getattr(self.args, "disable_tqdm", False)),
            )

            for acc_batch in _all_gen_batches:
                _gen_buffer.append(acc_batch)
                _gen_buffer_count += acc_batch["input_ids"].size(0)

                if _gen_target_B > 0 and _gen_buffer_count < _gen_target_B:
                    continue

                _flush_gen_buffer()
                gen_pbar.update(1)

            # Flush remainder
            if _gen_buffer:
                _flush_gen_buffer()
                gen_pbar.update(1)

            gen_pbar.close()

        # Wait for any in-flight decode threads before reading metric_state
        _drain_decode_threads()

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

        # ---- restore training VRAM state (grad-ckpt, optimizer) ------
        if _vram_state is not None:
            self._restore_eval_vram(_vram_state)

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
        _accel_err = getattr(torch, "AcceleratorError", None)
        is_accelerator_cuda_error = (
            "CUDA error" in msg
            or (_accel_err is not None and isinstance(err, _accel_err))
        )

        should_save = is_cuda_oom or is_launch_timeout or is_accelerator_cuda_error
        if not should_save:
            raise err

        ckpt_dir = os.path.join(self.args.output_dir, "last_error_checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # After a fatal CUDA error (launch timeout, ECC error, etc.), the GPU
        # is in an unrecoverable state — all CUDA ops will fail, including the
        # .to("cpu") inside save_pretrained.  Move weights to CPU first so
        # safetensors can serialize them without touching the dead GPU.
        saved = False
        try:
            try:
                self.model.to("cpu")
            except Exception:
                pass  # GPU might already be dead; weights may already be on CPU
            self.save_model(ckpt_dir)
            saved = True
        except Exception as save_err:
            print(f"\n[TokenPackTrainer] WARNING: emergency save failed: {save_err}")
            # Last resort: try saving just the state_dict with torch.save
            try:
                state_dict = {k: v.cpu() if hasattr(v, 'cpu') else v
                              for k, v in self.model.state_dict().items()}
                torch.save(state_dict, os.path.join(ckpt_dir, "pytorch_model_emergency.bin"))
                print(f"[TokenPackTrainer] Saved raw state_dict to {ckpt_dir}/pytorch_model_emergency.bin")
                saved = True
            except Exception as e2:
                print(f"[TokenPackTrainer] WARNING: raw state_dict save also failed: {e2}")

        if saved and self.args.should_save:
            try:
                self.state.save_to_json(os.path.join(ckpt_dir, "trainer_state.json"))
                if self.optimizer is not None:
                    self._save_optimizer_and_scheduler(ckpt_dir)
            except Exception as state_err:
                print(f"[TokenPackTrainer] WARNING: could not save trainer state: {state_err}")
            try:
                self._save_regime_state(ckpt_dir)
            except Exception:
                pass

        if saved:
            print(f"\n[TokenPackTrainer] Saved emergency checkpoint to {ckpt_dir} "
                  f"after CUDA error: {msg}\n")
        else:
            print(f"\n[TokenPackTrainer] CUDA error occurred but emergency save failed: {msg}\n")

        raise err
        
    # --------------------------------------------------------------
    # OOM-aware wrappers around train/evaluate
    # --------------------------------------------------------------
    # ------------------------------------------------------------------
    # Regime state persistence
    # ------------------------------------------------------------------
    REGIME_STATE_FILENAME = "regime_state.json"

    def _save_regime_state(self, output_dir: str):
        """Persist regime limits and OOM history to a checkpoint directory."""
        state = {
            "regime_limits": {
                str(k): v for k, v in self._regime_limits.items()
            },
            "autotuned_keys": list(getattr(self, "_autotuned_keys", set())),
            "regime_oom_count": getattr(self, "_regime_oom_count", {}),
            "oom_events": getattr(self, "_oom_events", 0),
            "oom_skipped_batches": getattr(self, "_oom_skipped_batches", 0),
            "gpu_baseline_mem": getattr(self, "_gpu_baseline_mem", None),
            "gpu_baseline_mem_per_device": {
                str(k): v for k, v in getattr(self, "_gpu_baseline_mem_per_device", {}).items()
            },
            "seeded_from": getattr(self, "_seed_regime_state_path", None),
        }
        path = os.path.join(output_dir, self.REGIME_STATE_FILENAME)
        try:
            with open(path, "w") as f:
                json.dump(state, f, indent=2)
            if self.debug:
                logger.info(f"[TokenPackTrainer] Saved regime state ({len(self._regime_limits)} regimes) to {path}")
        except Exception as e:
            logger.warning(f"[TokenPackTrainer] Could not save regime state: {e}")

    def _load_regime_state(self, checkpoint_dir: str) -> bool:
        """Restore regime limits from a checkpoint. Returns True if loaded successfully."""
        path = os.path.join(checkpoint_dir, self.REGIME_STATE_FILENAME)
        if not os.path.isfile(path):
            return False
        try:
            with open(path, "r") as f:
                state = json.load(f)

            # Restore regime limits (keys are ints, JSON saves them as strings)
            loaded_limits = {}
            for k, v in state.get("regime_limits", {}).items():
                loaded_limits[int(k)] = {
                    "B": v.get("B"),
                    "T": int(v["T"]),
                    "stable": int(v.get("stable", 0)),
                    "hwm_T": int(v["hwm_T"]) if v.get("hwm_T") is not None else None,
                    "hwm_B": int(v["hwm_B"]) if v.get("hwm_B") is not None else None,
                    "bytes_per_token": float(v["bytes_per_token"]) if v.get("bytes_per_token") is not None else None,
                }
            self._regime_limits = loaded_limits
            # Always clear autotuned_keys on resume.  This forces each regime
            # to re-autotune on its first successful step, calibrating T to the
            # current safety factor and actual GPU memory state.  Without this,
            # stale T values from a previous run (computed with a different
            # safety factor or on a different GPU) persist indefinitely.
            self._autotuned_keys = set()
            self._regime_oom_count = {
                int(k): int(v) for k, v in state.get("regime_oom_count", {}).items()
            }
            self._oom_events = int(state.get("oom_events", 0))
            self._oom_skipped_batches = int(state.get("oom_skipped_batches", 0))

            # Restore baseline memory if available (avoids conservative first-batch autotuning)
            baseline = state.get("gpu_baseline_mem")
            if baseline is not None:
                self._gpu_baseline_mem = int(baseline)
            per_dev = state.get("gpu_baseline_mem_per_device", {})
            if per_dev:
                self._gpu_baseline_mem_per_device = {
                    int(k): int(v) for k, v in per_dev.items()
                }

            logger.info(
                f"[TokenPackTrainer] Restored regime state from {path}: "
                f"{len(loaded_limits)} regimes, {self._oom_events} prior OOM events"
            )
            return True
        except Exception as e:
            logger.warning(f"[TokenPackTrainer] Could not load regime state from {path}: {e}")
            return False

    def _seed_regime_state(self, source: str) -> bool:
        """Seed regime B/T defaults from a previous run's regime_state.json.

        Unlike _load_regime_state (which fully restores a checkpoint's state for
        resuming), this method uses the previous run's calibrated B and T values
        as *starting points* for a fresh training run.  This avoids the expensive
        cold-start calibration that can take thousands of OOM events for
        large-vocab models.

        What is preserved:
          - B (max examples per microbatch): reflects data distribution + vocab
            size constraints that are unlikely to change between runs.
          - T (token budget): a reasonable starting point, though the model's
            memory footprint may have changed.

        What is reset:
          - autotuned_keys: cleared so every regime re-autotunes T with the
            current model's actual memory baseline on first encounter.
          - stable counts: reset to 0 so ramp-up behaviour starts fresh.
          - OOM counts/history: cleared since the new model may have different
            memory characteristics.
          - gpu_baseline_mem: not loaded — re-measured from current model.

        Args:
            source: Path to a regime_state.json file or a checkpoint directory
                    containing one.

        Returns:
            True if seeding succeeded, False otherwise.
        """
        path = source
        if os.path.isdir(source):
            path = os.path.join(source, self.REGIME_STATE_FILENAME)
        if not os.path.isfile(path):
            logger.warning(f"[TokenPackTrainer] seed_regime_state: file not found: {path}")
            return False
        try:
            with open(path, "r") as f:
                state = json.load(f)

            adaptive = self._adaptive_regime_state
            seeded = {}
            for k, v in state.get("regime_limits", {}).items():
                seeded[int(k)] = {
                    "B": v.get("B"),
                    "T": int(v["T"]),
                    "stable": 0,  # fresh start — don't inherit stability
                    # In adaptive mode, clear HWMs so regimes can explore beyond
                    # the previous run's proven limits (which may be suboptimal
                    # for the new data distribution).
                    "hwm_T": None if adaptive else (int(v["hwm_T"]) if v.get("hwm_T") is not None else None),
                    "hwm_B": None if adaptive else (int(v["hwm_B"]) if v.get("hwm_B") is not None else None),
                    "bytes_per_token": None,  # reset — model memory footprint may differ
                    "adaptive_ramps": 3 if adaptive else 0,
                }
            self._regime_limits = seeded

            # Force re-autotune: the model's memory footprint may differ
            # (e.g. untied embeddings, different model size, different optimizer).
            self._autotuned_keys = set()
            self._regime_oom_count = {}
            # Don't load gpu_baseline_mem — let it re-measure with current model.

            mode = "adaptive" if adaptive else "seeded"
            logger.info(
                f"[TokenPackTrainer] Seeded regime state from {path}: "
                f"{len(seeded)} regimes ({mode}, B/T loaded, autotuned_keys cleared for re-calibration)"
            )
            return True
        except Exception as e:
            logger.warning(f"[TokenPackTrainer] Could not seed regime state from {path}: {e}")
            return False

    def _save_checkpoint(self, model, trial):
        """Override HF Trainer to also persist regime state alongside each checkpoint."""
        super()._save_checkpoint(model, trial)

        # Determine the checkpoint dir that was just created (mirrors HF logic)
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        if os.path.isdir(output_dir) and self.args.should_save:
            self._save_regime_state(output_dir)

    def train(self, *args, **kwargs):
        # Catch CUDA errors (OOM, launch timeout, ECC) for emergency checkpoint.
        # torch.AcceleratorError (PyTorch ≥2.3) wraps fatal CUDA errors and may
        # not be a RuntimeError subclass, so catch it explicitly if available.
        _catch = [torch.cuda.OutOfMemoryError, RuntimeError]
        _accel_err = getattr(torch, "AcceleratorError", None)
        if _accel_err is not None:
            _catch.append(_accel_err)
        _catch = tuple(_catch)

        # Restore regime state from checkpoint if resuming
        resume = kwargs.get("resume_from_checkpoint") or (args[0] if args else None)
        if resume:
            ckpt_dir = str(resume)
            if isinstance(resume, bool) and resume:
                # HF resolves True → latest checkpoint in output_dir
                from transformers.trainer_utils import get_last_checkpoint
                ckpt_dir = get_last_checkpoint(self.args.output_dir)
            if ckpt_dir and os.path.isdir(ckpt_dir):
                self._load_regime_state(ckpt_dir)

        try:
            return super().train(*args, **kwargs)
        except _catch as err:
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

        # Restore eval token budget for this pass.  If we have a HWM from a
        # previous eval pass (proven-safe ceiling), start there instead of
        # re-discovering the OOM boundary from the user's initial value.
        # This prevents the first few batches from OOM-ing every eval call.
        self._eval_regime_init()
        initial_T = getattr(self, "_initial_eval_tokens_per_microbatch", None)
        hwm_T = self._eval_hwm_T
        if initial_T is not None:
            if hwm_T is not None and hwm_T < initial_T:
                # HWM is lower than initial — start from the proven ceiling
                self.max_eval_tokens_per_microbatch = hwm_T
            else:
                self.max_eval_tokens_per_microbatch = initial_T

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

    def _get_num_items_in_batch(self, batch_samples, device=None):
        """
        Override HF's _get_num_items_in_batch to compute on CPU.

        HF's default runs batch["labels"].ne(-100) on whatever device the
        labels live on.  With use_cpu_microbatch=False the CUDAPrefetcher has
        already moved labels to GPU, so this tiny allocation can OOM when
        VRAM is fragmented.  Moving to CPU first is essentially free and
        makes it impossible to OOM here.
        """
        total = 0
        for batch in batch_samples:
            labels = batch.get("labels")
            if labels is None:
                continue
            if labels.is_cuda:
                labels = labels.cpu()
            total += int(labels.ne(-100).sum().item())
        return total

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

        eff = enc_len + self.decoder_cost_multiplier * dec_len
        mx = int(eff.max().item())  # <-- scalar sync only
        bs = int(self._regime_bucket_size)
        return int((mx + bs - 1) // bs) if bs > 0 else mx

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        _t0 = time.time()  # step-level timing

        # Inter-step gap: time between previous training_step return and this call.
        # This is the dataloader + collator time — invisible to within-step timing.
        _inter_step_gap_ms = None
        if self._last_step_end_time is not None:
            _inter_step_gap_ms = (_t0 - self._last_step_end_time) * 1000

        # turn off KV cache during training to reduce memory spikes
        if hasattr(model, "config") and getattr(model.config, "use_cache", None) is True:
            model.config.use_cache = False

        # Track baseline GPU memory (model + optimizer, before any batch data).
        # This lets _autotune_regime_from_peak compute marginal per-token cost
        # instead of attributing fixed model memory to each token.
        #
        # Use LATEST value (not max).  The max approach captured transient
        # spikes from checkpoint loading (optimizer states briefly on GPU,
        # then freed), locking in a baseline of 93GB when steady-state was
        # 13GB — making the autotune think there was no headroom.
        #
        # The latest-value approach: each measurement overwrites the previous.
        # By step 5, the steady-state (model + optimizer if on GPU) is captured.
        _local_step = getattr(self, "_timing_step_count", 0)
        if torch.cuda.is_available() and _local_step <= 5:
            for dev_i in range(torch.cuda.device_count()):
                mem_i = torch.cuda.memory_allocated(dev_i)
                if mem_i <= 0:
                    continue
                old_i = self._gpu_baseline_mem_per_device.get(dev_i)
                self._gpu_baseline_mem_per_device[dev_i] = mem_i
                if self.debug and (old_i is None or abs(mem_i - old_i) > old_i * 0.05 if old_i else True):
                    print(f"[TokenPackTrainer] GPU baseline (device {dev_i}): {mem_i / (1<<30):.2f} GB")

            # Legacy single-device baseline
            self._gpu_baseline_mem = torch.cuda.memory_allocated(self.args.device)

        last_err = None
        regime_key = None  # <<< keep across retries for this same HF batch
        _b_shrunk_this_batch = False  # only shrink B once per batch

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

                eff_tokens = int((enc_len + self.decoder_cost_multiplier * dec_len).sum().item())
                _t_plan = time.time()  # planning done

                # --- Cap microbatches per step ---
                # If a regime would produce too many microbatches, bump BOTH
                # the token budget (T) and examples-per-microbatch (B) for
                # THIS step.  Previously only T was bumped, but for regimes
                # with many short examples (regime 1), B is the bottleneck —
                # 25,000 examples / B=171 = 151 microbatches regardless of T.
                #
                # IMPORTANT: Only apply on attempt 0 (not OOM retries).
                _max_mb = self.max_microbatches_per_step
                if attempt == 0 and _max_mb > 0 and len(microbatches) > _max_mb:
                    step_T = max(
                        int(eff_tokens / _max_mb) + 1,
                        self.max_tokens_per_microbatch,
                    )
                    # Also bump B — but cap at 2× the regime's current B to avoid
                    # jumping far past what's proven stable.  If the HWM is known,
                    # use it as the ceiling (don't exceed what worked before).
                    _total_ex = sum(mb["input_ids"].size(0) for mb in microbatches)
                    _ideal_B = int(_total_ex / _max_mb) + 1
                    _cur_B = self.max_examples_per_microbatch or _ideal_B
                    _regime_hwm_b = self._regime_state(regime_key).get("hwm_B") if regime_key is not None else None
                    _cap_B = max(_cur_B * 2, _cur_B + 16)
                    if _regime_hwm_b:
                        _cap_B = min(_cap_B, int(_regime_hwm_b * 1.5))
                    step_B = min(_ideal_B, max(_cap_B, _cur_B))
                    microbatches, enc_len, dec_len = self._make_microbatches(
                        inputs_work,
                        max_tokens_per_microbatch=step_T,
                        max_examples_per_microbatch=step_B,
                        return_lengths=True,
                        precomputed_lengths=(enc_len, dec_len),
                    )
                    if self.debug:
                        print(
                            f"[TokenPackTrainer] Microbatch cap: {_total_ex} examples, "
                            f"regime B={self.max_examples_per_microbatch}, T={self.max_tokens_per_microbatch} "
                            f"→ bumped to B={step_B}, T={step_T} "
                            f"→ {len(microbatches)} microbatches for this step"
                        )

                num_micro = max(len(microbatches), 1)
                total_loss = 0.0
                total_examples = sum(mb["input_ids"].size(0) for mb in microbatches)

                if torch.cuda.is_available():
                    for _di in range(torch.cuda.device_count()):
                        torch.cuda.reset_peak_memory_stats(_di)

                # ---------------- EXECUTE MICROBATCHE(S) ----------------
                total_tokens = 0
                mb_token_counts = []
                mb_eff_tokens = []   # per-microbatch effective tokens (enc + alpha*dec) for predictive VRAM check
                _need_lm_check = self._lm_head_vocab_size > 0
                mb_lm_head_tokens = []  # B * dec_max per microbatch (for pre-flight check)
                _alpha = self.decoder_cost_multiplier
                for mb in microbatches:
                    labels = mb.get("labels", None)
                    if isinstance(labels, torch.Tensor) and labels.ndim == 2:
                        if _need_lm_check:
                            non_pad = (labels != -100).sum(dim=-1)
                            ntok = int(non_pad.sum().item())
                            dec_max = int(non_pad.max().item()) if ntok > 0 else 0
                            lm_tok = labels.size(0) * dec_max
                        else:
                            ntok = int((labels != -100).sum().item())
                            lm_tok = 0
                    else:
                        ntok = 0
                        lm_tok = 0
                    # Compute effective tokens: enc_tokens + alpha * dec_tokens
                    _am = mb.get("attention_mask", None)
                    if _am is not None and isinstance(_am, torch.Tensor) and _am.ndim == 2:
                        _mb_enc = int(_am.sum().item())
                    else:
                        _mb_enc = 0
                    mb_eff_tokens.append(int(_mb_enc + _alpha * ntok))
                    mb_token_counts.append(ntok)
                    mb_lm_head_tokens.append(lm_tok)
                    total_tokens += ntok

                total_loss_weighted = 0.0

                # Pre-flight check: run ONCE per step using the worst-case
                # microbatch (max lm_tok) to avoid GPU sync stalls in the hot
                # loop.  memory_allocated() forces a GPU sync, so calling it
                # per-microbatch tanks throughput on fast models.
                if _need_lm_check and mb_lm_head_tokens and torch.cuda.is_available():
                    max_lm_tok = max(mb_lm_head_tokens)
                    if max_lm_tok > 0:
                        lm_bytes = self._estimate_lm_head_bytes(max_lm_tok)
                        if lm_bytes > 0:
                            # Check free memory on the bottleneck GPU (least headroom)
                            min_free = None
                            for _di in range(torch.cuda.device_count()):
                                _alloc = torch.cuda.memory_allocated(_di)
                                if _alloc <= 0:
                                    continue  # device not used
                                _free_i = torch.cuda.get_device_properties(_di).total_memory - _alloc
                                if min_free is None or _free_i < min_free:
                                    min_free = _free_i
                            if min_free is None:
                                dev = self.args.device
                                min_free = torch.cuda.get_device_properties(dev).total_memory - torch.cuda.memory_allocated(dev)
                            if lm_bytes > min_free * 0.85:
                                raise torch.cuda.OutOfMemoryError(
                                    f"[TokenPackTrainer] Pre-flight: lm_head output would need "
                                    f"{lm_bytes / (1 << 30):.1f} GiB but only "
                                    f"{min_free / (1 << 30):.1f} GiB free (skipping to OOM recovery)"
                                )

                # Predictive VRAM check: estimate peak memory for the worst-case
                # microbatch using the learned bytes_per_eff_token from the last
                # successful autotune.  This is entirely sync-free — all values
                # are cached from prior steps.  If the predicted peak exceeds
                # total VRAM * safety, raise synthetic OOM to trigger the regime
                # shrink BEFORE burning GPU time on a doomed forward pass.
                # Falls back gracefully: skipped when bytes_per_token is None
                # (first few steps or after OOM invalidation).
                if regime_key is not None and mb_eff_tokens and torch.cuda.is_available():
                    _regime_st = self._regime_state(regime_key)
                    _bpt = _regime_st.get("bytes_per_token")
                    if _bpt is not None and _bpt > 0:
                        _max_mb_eff = max(mb_eff_tokens)
                        # Find bottleneck device using cached values (no GPU sync)
                        _baselines = self._gpu_baseline_mem_per_device
                        _pv_total = None
                        _pv_baseline = None
                        for _di, _base_i in _baselines.items():
                            _tot_i = torch.cuda.get_device_properties(_di).total_memory
                            _head_i = _tot_i - _base_i
                            if _pv_total is None or _head_i < (_pv_total - _pv_baseline):
                                _pv_total = _tot_i
                                _pv_baseline = _base_i
                        if _pv_total is None:
                            _pv_total = int(self._vram_gb * (1 << 30))
                            _pv_baseline = self._gpu_baseline_mem or 0
                        _predicted_peak = _pv_baseline + _bpt * _max_mb_eff
                        _safe_limit = _pv_total * self._autotune_safety
                        if _predicted_peak > _safe_limit:
                            raise torch.cuda.OutOfMemoryError(
                                f"[TokenPackTrainer] Predictive VRAM check: worst microbatch "
                                f"({_max_mb_eff} eff tokens, {_bpt:.0f} bytes/tok) would need "
                                f"~{_predicted_peak / (1 << 30):.1f} GiB but safety limit is "
                                f"{_safe_limit / (1 << 30):.1f} GiB — shrinking regime before "
                                f"attempting forward pass"
                            )

                _t_pre = time.time()  # pre-flight done

                # Use prefetching iterator to overlap CPU→GPU transfer with computation
                prefetched = self._prefetch_microbatches(microbatches) if self.use_cpu_microbatch else iter(microbatches)

                _completed_mb = 0
                _oom_mid_batch = False

                for mb, ntok, lm_tok in zip(prefetched, mb_token_counts, mb_lm_head_tokens):
                    if ntok == 0:
                        continue

                    # Note: mb is already on GPU if use_cpu_microbatch (via prefetching)

                    if self.length_column_name in mb:
                        mb = {k: v for k, v in mb.items() if k != self.length_column_name}

                    try:
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
                        _completed_mb += 1

                    except (RuntimeError, torch.cuda.OutOfMemoryError) as mb_err:
                        if not self._is_cuda_oom(mb_err):
                            raise
                        # --- Mid-batch OOM recovery ---
                        # Some microbatches already completed and their gradients
                        # are accumulated.  Rather than discarding everything and
                        # retrying (which likely OOMs again at the same T due to
                        # CUDA fragmentation), keep the partial gradients and
                        # finish the step.  The optimizer gets a smaller-than-usual
                        # gradient (like a reduced batch size for one step) which
                        # is far better than zero gradient from a skipped batch.
                        if _completed_mb > 0:
                            _oom_mid_batch = True
                            self._oom_events += 1
                            # Free cache WITHOUT zeroing gradients
                            if torch.cuda.is_available():
                                try:
                                    torch.cuda.empty_cache()
                                except Exception:
                                    pass
                            # Shrink regime for future steps
                            if regime_key is not None:
                                self._regime_on_oom(regime_key)
                                self._apply_regime_limits(regime_key)
                            if self.control.should_log:
                                self.log({
                                    "oom_mid_batch_recovery": 1.0,
                                    "oom_mid_batch_completed": float(_completed_mb),
                                    "oom_mid_batch_total": float(num_micro),
                                    "regime_key": float(regime_key if regime_key is not None else -1),
                                    "max_tokens_now": float(self.max_tokens_per_microbatch or 0),
                                })
                            break
                        else:
                            # No microbatches completed — fall through to the
                            # outer OOM handler for full retry logic.
                            raise

                # >>> on success (full or partial), mark regime and possibly ramp up
                if regime_key is not None and not _oom_mid_batch:
                    self._regime_on_success(regime_key)

                # --- Autotune T from observed peak memory ---
                #
                # FIX: use per-microbatch eff_tokens, not total across all MBs.
                # Peak memory reflects the SINGLE largest microbatch (only one is
                # on GPU at a time with use_cpu_microbatch=True), but eff_tokens
                # was the sum across ALL microbatches.  Dividing peak by total
                # underestimates bytes_per_token by ~Nx, causing T to overshoot
                # → OOM → conservative shrink → stuck at a local minimum.
                #
                # The correct denominator is max(mb_eff_tokens) — the effective
                # tokens in the microbatch that actually caused the peak.
                _autotune_eff_tokens = max(mb_eff_tokens) if mb_eff_tokens else eff_tokens

                # Skip autotune when peak memory stats are unrepresentative:
                #   - mid-batch recovery (partial step)
                #   - retry attempts (attempt > 0): tiny desperation-mode
                #     microbatches produce artificially low bytes_per_token,
                #     causing the autotune to think tokens are cheap → sets
                #     T=max → next cap-bumped attempt OOMs → B shrinks → repeat
                _autotuned_keys = getattr(self, "_autotuned_keys", set())
                _needs_autotune = regime_key not in _autotuned_keys

                # Periodic re-autotune: even after the initial calibration,
                # allow re-autotune every 500 steps using the normal safety
                # factor.  This prevents a conservative initial autotune
                # (e.g., from a post-OOM step with safety=0.60) from permanently
                # capping the regime.  Uses a dedicated counter (not 'stable',
                # which resets on probes and OOMs).
                if not _needs_autotune and regime_key is not None:
                    _retune_counts = getattr(self, "_regime_retune_counter", {})
                    _retune_count = _retune_counts.get(regime_key, 0) + 1
                    _retune_counts[regime_key] = _retune_count
                    self._regime_retune_counter = _retune_counts
                    if _retune_count % 500 == 0:
                        _needs_autotune = True  # re-calibrate with normal safety

                if attempt == 0 and not _oom_mid_batch and regime_key is not None and _needs_autotune:
                    # Use lower safety factor if this regime recently OOM'd
                    # (previous autotune overshot, so be more conservative)
                    st = self._regime_state(regime_key)
                    oom_history = getattr(self, "_regime_oom_count", {})
                    recent_oom = oom_history.get(regime_key, 0) > 0

                    # Skip autotune on tiny post-OOM batches: when eff_tokens is
                    # far below the regime's known capacity (HWM), the per-token
                    # cost is inflated by fixed overhead, producing a conservative
                    # estimate that prolongs recovery.  Let the fast ramp handle
                    # recovery instead; autotune on a properly-sized batch later.
                    hwm = st.get("hwm_T") or st["T"]
                    if recent_oom and _autotune_eff_tokens < hwm * 0.1:
                        pass  # defer autotune to a larger batch
                    else:
                        safety = self._autotune_oom_safety if recent_oom else self._autotune_safety
                        self._autotune_regime_from_peak(regime_key, _autotune_eff_tokens, safety=safety)
                        self._autotuned_keys = getattr(self, "_autotuned_keys", set())
                        self._autotuned_keys.add(regime_key)
                        # Clear OOM history for this key now that we've re-autotuned
                        if recent_oom:
                            oom_history[regime_key] = 0
                            self._regime_oom_count = oom_history


                # --- GPU utilization tracking ---
                # Sample GPU utilization after execution to detect under-
                # utilization caused by conservative regimes.  The rolling
                # window feeds _regime_on_success for utilization-aware ramps.
                if torch.cuda.is_available() and not _oom_mid_batch:
                    try:
                        _gpu_util = torch.cuda.utilization(self.args.device)
                        _uh = getattr(self, "_gpu_util_history", [])
                        _uh.append(_gpu_util)
                        # Keep last 20 samples
                        if len(_uh) > 20:
                            self._gpu_util_history = _uh[-20:]
                        else:
                            self._gpu_util_history = _uh
                    except Exception:
                        pass  # NVML not available or device error

                # optional logging
                if attempt > 0 and self.control.should_log:
                    self.log({
                        "oom_recovered": 1.0,
                        "oom_recovery_attempt": float(attempt),
                        "regime_key": float(regime_key if regime_key is not None else -1),
                        "max_B_now": float(self.max_examples_per_microbatch or 0),
                        "max_tokens_now": float(self.max_tokens_per_microbatch or 0),
                    })

                # --- Step timing (log every 100 steps) ---
                _t_end = time.time()
                _step_count = getattr(self, "_timing_step_count", 0) + 1
                self._timing_step_count = _step_count
                _plan_ms = (_t_plan - _t0) * 1000
                _exec_ms = (_t_end - _t_pre) * 1000
                _total_ms = (_t_end - _t0) * 1000

                # Accumulate for averaging
                if not hasattr(self, "_timing_accum"):
                    self._timing_accum = {"plan": 0.0, "exec": 0.0, "total": 0.0, "n": 0,
                                          "examples": 0, "tokens": 0, "microbatches": 0,
                                          "inter_step_gap": 0.0, "inter_step_n": 0}
                ta = self._timing_accum
                ta["plan"] += _plan_ms
                ta["exec"] += _exec_ms
                ta["total"] += _total_ms
                ta["n"] += 1
                ta["examples"] += total_examples
                ta["tokens"] += total_tokens
                ta["microbatches"] += num_micro
                if _inter_step_gap_ms is not None:
                    ta["inter_step_gap"] += _inter_step_gap_ms
                    ta["inter_step_n"] += 1

                if ta["n"] >= 100:
                    n = ta["n"]
                    _uh = getattr(self, "_gpu_util_history", [])
                    _avg_util = sum(_uh) / len(_uh) if _uh else -1
                    _avg_gap = ta["inter_step_gap"] / max(ta["inter_step_n"], 1)
                    _gap_pct = 100.0 * _avg_gap / (_avg_gap + ta["total"]/n) if _avg_gap > 0 else 0
                    logger.info(
                        f"[TokenPackTrainer] Step timing (avg of {n}): "
                        f"plan={ta['plan']/n:.1f}ms, exec={ta['exec']/n:.1f}ms, "
                        f"total={ta['total']/n:.1f}ms, "
                        f"data_wait={_avg_gap:.0f}ms ({_gap_pct:.0f}% idle) | "
                        f"examples/step={ta['examples']/n:.0f}, "
                        f"tokens/step={ta['tokens']/n:.0f}, "
                        f"microbatches/step={ta['microbatches']/n:.1f} | "
                        f"gpu_util={_avg_util:.0f}%"
                    )
                    self._timing_accum = {"plan": 0.0, "exec": 0.0, "total": 0.0, "n": 0,
                                          "examples": 0, "tokens": 0, "microbatches": 0,
                                          "inter_step_gap": 0.0, "inter_step_n": 0}

                # --- Step diagnostic ---
                # Log regime details (gated by debug=True).
                # Triggers:
                #   - First 3 steps unconditionally (immediate visibility on resume)
                #   - Every 200th step (periodic health check)
                #   - >3× baseline or >5s absolute (regression detection)
                # Throttled on slow steps: first 3, then every 50th.
                _base_now = getattr(self, "_step_exec_baseline_ms", _exec_ms)
                _is_slow = _step_count > 3 and (_exec_ms > _base_now * 3 or _exec_ms > 5_000)
                if self.debug:
                    _regime_st = self._regime_state(regime_key) if regime_key is not None else {}
                    _vram_mb = torch.cuda.memory_allocated(self.args.device) / (1 << 20) if torch.cuda.is_available() else 0
                    _vram_total_mb = torch.cuda.get_device_properties(self.args.device).total_memory / (1 << 20) if torch.cuda.is_available() else 0
                    _slow_tag = " [SLOW]" if _is_slow else ""
                    print(
                        f"[TokenPackTrainer] Step {_step_count}{_slow_tag}: "
                        f"{_exec_ms/1000:.1f}s "
                        f"(baseline={_base_now/1000:.1f}s) | "
                        f"regime={regime_key}, B={_regime_st.get('B')}, "
                        f"T={_regime_st.get('T')}, hwm_T={_regime_st.get('hwm_T')}, "
                        f"stable={_regime_st.get('stable', 0)}, "
                        f"bpt={_regime_st.get('bytes_per_token', 'None')} | "
                        f"microbatches={num_micro}, examples={total_examples}, "
                        f"tokens={total_tokens}, eff_tokens={eff_tokens} | "
                        f"VRAM={_vram_mb:.0f}/{_vram_total_mb:.0f}MB"
                    )

                # --- Sustained slowdown watchdog ---
                # Defragment CUDA cache when training has been slow for a
                # sustained period.  A single slow step can happen for many
                # reasons (long sequence, GC pause, data loading); only act
                # when the slowdown is persistent, suggesting CUDA memory
                # fragmentation.  empty_cache() costs ~1-10ms and consolidates
                # free blocks without moving data.
                if not _oom_mid_batch and torch.cuda.is_available():
                    # Track best-known step time (healthy baseline)
                    _baseline_ms = getattr(self, "_step_exec_baseline_ms", None)
                    if _baseline_ms is None or _exec_ms < _baseline_ms:
                        self._step_exec_baseline_ms = _exec_ms

                    # Accumulate slow-step duration: count consecutive seconds
                    # spent in steps that are ≥10× the baseline.  Only trigger
                    # defrag when this exceeds a sustained threshold (2 minutes).
                    _base_now = self._step_exec_baseline_ms or _exec_ms
                    _slow_accum = getattr(self, "_slow_step_accum_s", 0.0)
                    _last_defrag_step = getattr(self, "_last_defrag_step", 0)

                    if _step_count > 20 and _exec_ms > max(_base_now * 10, 10_000):
                        # This step was slow — accumulate its duration
                        _slow_accum += _exec_ms / 1000.0
                    else:
                        # Normal step — reset accumulator
                        _slow_accum = 0.0
                    self._slow_step_accum_s = _slow_accum

                    # Fire when ≥120 seconds of sustained slow steps have
                    # accumulated AND at least 100 steps since last defrag
                    if _slow_accum >= 120 and (_step_count - _last_defrag_step) > 100:
                        _alloc = torch.cuda.memory_allocated(self.args.device)
                        _total_dev = torch.cuda.get_device_properties(self.args.device).total_memory
                        _util = _alloc / max(_total_dev, 1)

                        if self.debug:
                            print(
                                f"[TokenPackTrainer] Sustained slowdown: "
                                f"{_slow_accum:.0f}s of slow steps accumulated "
                                f"(baseline={_base_now/1000:.1f}s), VRAM {_util:.0%}. "
                                f"Defragmenting CUDA cache."
                            )
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

                        self._last_defrag_step = _step_count
                        self._slow_step_accum_s = 0.0
                        if self.control.should_log:
                            self.log({
                                "sustained_defrag": 1.0,
                                "slow_accum_s": _slow_accum,
                                "pressure_vram_util": _util,
                            })

                # --- Diagnostics file (every N steps) ---
                if self._diag_interval > 0 and _step_count % self._diag_interval == 0:
                    try:
                        self._write_step_diagnostics(
                            step=_step_count,
                            regime_key=regime_key,
                            plan_ms=_plan_ms,
                            exec_ms=_exec_ms,
                            total_ms=_total_ms,
                            inter_step_gap_ms=_inter_step_gap_ms,
                            num_examples=total_examples,
                            num_tokens=total_tokens,
                            eff_tokens=eff_tokens,
                            num_microbatches=num_micro,
                            mb_eff_tokens=mb_eff_tokens,
                        )
                    except Exception:
                        pass  # never let diagnostics crash training

                self._last_step_end_time = time.time()

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
                    # Only shrink B on the first OOM per batch.  Subsequent
                    # retries shrink T instead.  This prevents cascading B
                    # from e.g. 457 → 7 across 4 retries in one batch.
                    self._regime_on_oom(regime_key, shrink_b=not _b_shrunk_this_batch)
                    _b_shrunk_this_batch = True
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
