from __future__ import annotations

from typing import Callable, List, Optional, Union

import numpy as np


def _postprocess_text(preds: List[str], labels: List[str]) -> tuple:
    """Normalize predictions and labels for metric computation."""
    preds = [p.strip() for p in preds]

    # labels can be list[str] OR list[list[str]]; normalize to list[list[str]]
    norm_labels = []
    for lab in labels:
        if isinstance(lab, (list, tuple)):
            s = lab[0] if len(lab) > 0 else ""
        else:
            s = lab
        norm_labels.append([str(s).strip()])

    return preds, norm_labels


def _truncate_by_words(s: str, max_words: int = 64) -> str:
    """Truncate string to max_words."""
    words = s.split()
    return " ".join(words[:max_words])


def is_english_target(prompt: str) -> bool:
    """
    Heuristic: return True if the prompt indicates that the *target*
    language is English. Used to filter METEOR computation (English-only).
    """
    if not isinstance(prompt, str):
        return False
    p = prompt.lower()
    return (
        " to english" in p
        or " into english" in p
        or "english translation" in p
    )


def make_seq2seq_compute_metrics(
    tokenizer,
    eval_dataset=None,
    source_column: str = "source",
    max_ref_tokens: int = 64,
    meteor_only_for_english: bool = True,
    is_english_fn: Optional[Callable[[str], bool]] = None,
) -> Callable:
    """
    Factory function that creates a compute_metrics function for seq2seq evaluation.

    Args:
        tokenizer: HuggingFace tokenizer for decoding predictions/labels
        eval_dataset: Optional dataset to extract source prompts for METEOR filtering.
                      If None, METEOR is computed on all examples.
        source_column: Column name containing source text (for METEOR English filtering)
        max_ref_tokens: Maximum words to keep in predictions/references
        meteor_only_for_english: If True, only compute METEOR for English targets
        is_english_fn: Custom function to determine if target is English.
                       Defaults to heuristic based on prompt text.

    Returns:
        A compute_metrics function compatible with HuggingFace Trainer.

    Example:
        >>> compute_metrics = make_seq2seq_compute_metrics(
        ...     tokenizer=tokenizer,
        ...     eval_dataset=eval_dataset,
        ...     source_column="source",
        ... )
        >>> trainer = TokenPackTrainer(..., compute_metrics=compute_metrics)
    """
    # Lazy-load evaluate metrics to avoid import overhead if not used
    try:
        import evaluate
        bleu_metric = evaluate.load("sacrebleu")
        chrf_metric = evaluate.load("chrf")
        meteor_metric = evaluate.load("meteor")
    except ImportError:
        raise ImportError(
            "The 'evaluate' package is required for metrics computation. "
            "Install it with: pip install evaluate"
        )

    # Build METEOR mask if filtering by English targets
    meteor_mask: Optional[List[bool]] = None
    if meteor_only_for_english and eval_dataset is not None:
        check_fn = is_english_fn or is_english_target
        try:
            if hasattr(eval_dataset, "__getitem__") and hasattr(eval_dataset, "__len__"):
                # Dataset-like object
                if hasattr(eval_dataset, "column_names") and source_column in eval_dataset.column_names:
                    sources = eval_dataset[source_column]
                else:
                    sources = [eval_dataset[i].get(source_column, "") for i in range(len(eval_dataset))]
                meteor_mask = [check_fn(src) for src in sources]
        except Exception:
            # If we can't build the mask, compute METEOR on all examples
            meteor_mask = None

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def compute_metrics(eval_preds) -> dict:
        """Compute BLEU, chrF, METEOR, and generation length metrics."""
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Ensure numpy arrays
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        # Trim to same number of examples
        n = min(labels.shape[0], preds.shape[0])
        preds = preds[:n]
        labels = labels[:n]

        # Restore label pad positions before decoding (-100 -> pad_id)
        labels = np.where(labels != -100, labels, pad_id)

        # Decode
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Truncate and normalize
        decoded_preds = [_truncate_by_words(p, max_ref_tokens) for p in decoded_preds]
        decoded_labels = [[_truncate_by_words(lab, max_ref_tokens)] for lab in decoded_labels]
        decoded_preds, decoded_labels = _postprocess_text(decoded_preds, decoded_labels)

        # BLEU
        bleu_res = bleu_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            tokenize="none",
            lowercase=False,
        )

        # chrF
        chrf_res = chrf_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
        )

        # METEOR (optionally filtered to English targets)
        if meteor_mask is not None and len(meteor_mask) >= n:
            mask = meteor_mask[:n]
            meteor_preds = [p for p, m in zip(decoded_preds, mask) if m]
            meteor_refs = [lbl[0] for lbl, m in zip(decoded_labels, mask) if m]
        else:
            # Compute on all examples
            meteor_preds = decoded_preds
            meteor_refs = [lbl[0] for lbl in decoded_labels]

        if meteor_preds:
            meteor_res = meteor_metric.compute(predictions=meteor_preds, references=meteor_refs)
            meteor_score = float(meteor_res["meteor"])
        else:
            meteor_score = float("nan")

        # Generation length (excluding padding)
        gen_len = float(np.mean(np.sum(preds != pad_id, axis=1)))

        return {
            "bleu": float(round(bleu_res["score"], 4)),
            "chrf": float(round(chrf_res["score"], 4)),
            "meteor": float(round(meteor_score, 4)) if meteor_score == meteor_score else float("nan"),
            "gen_len": float(round(gen_len, 2)),
        }

    return compute_metrics
