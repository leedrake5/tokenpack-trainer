from __future__ import annotations

import gc
from typing import Callable, Iterable, Optional

import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset


def default_is_english_target_from_prompt(prompt: str) -> bool:
    """
    Heuristic: return True if the prompt indicates that the *target*
    language is English.
    """
    if not isinstance(prompt, str):
        return False
    p = prompt.lower()
    return (
        " to english" in p
        or " into english" in p
        or "english translation" in p
    )


def compute_metrics(
    tokenizer,
    eval_dataset: Optional[Dataset] = None,
    *,
    source_column: str = "source",
    meteor_only_for_english: bool = True,
    is_english_target_fn: Optional[Callable[[str], bool]] = None,
):
    """
    Build a HuggingFace-compatible compute_metrics function that:

      - Computes BLEU + chrF on all examples
      - Computes METEOR only for rows whose prompt indicates English target
        (unless meteor_only_for_english=False, in which case METEOR is on all)
      - Uses the same logic regardless of where the eval data came from
        (CSV, HF hub, etc.), as long as eval_dataset has `source_column`.

    Parameters
    ----------
    tokenizer:
        HF tokenizer used for decoding predictions / labels.
    eval_dataset:
        The *same* dataset object you pass to Trainer.eval_dataset
        (or None if no need for METEOR filtering by prompt).
    source_column:
        Column name containing the prompt / source text in eval_dataset.
    meteor_only_for_english:
        If True, compute METEOR only on a subset where is_english_target_fn(prompt) is True.
        If False, compute METEOR on all examples.
    is_english_target_fn:
        A function prompt -> bool. Defaults to a simple heuristic that
        looks for "to English", "into English", etc.

    Returns
    -------
    compute_metrics(eval_preds) -> dict
        A function suitable to pass into Trainer(..., compute_metrics=...).
    """
    # ---- Load metrics once ----
    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")
    meteor_metric = evaluate.load("meteor")

    if is_english_target_fn is None:
        is_english_target_fn = default_is_english_target_from_prompt

    # ---- Build mask aligned to eval_dataset, if possible ----
    meteor_target_mask_full = None
    if meteor_only_for_english and eval_dataset is not None:
        if source_column not in eval_dataset.column_names:
            raise ValueError(
                f"make_seq2seq_compute_metrics: eval_dataset has no column '{source_column}'. "
                "Either set `source_column` correctly or pass `meteor_only_for_english=False`."
            )

        # This will be a list of booleans aligned with eval_dataset rows
        prompts = eval_dataset[source_column]
        meteor_target_mask_full = np.array(
            [is_english_target_fn(str(p)) for p in prompts],
            dtype=bool,
        )

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    max_id = tokenizer.vocab_size - 1

    # ---- Actual HF-compatible compute_metrics closure ----
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # 1) True length (HF may give larger arrays than n_rows)
        true_len = labels.shape[0]

        preds = preds[:true_len]
        labels = labels[:true_len]

        # 2) Clip IDs into valid vocab range
        nonlocal max_id
        preds_clipped = np.clip(preds, 0, max_id)
        labels_clipped = np.clip(labels, 0, max_id)

        # 3) Convert to Python lists
        preds_list = preds_clipped.tolist()
        labels_list = labels_clipped.tolist()

        # 4) Decode token IDs -> strings
        decoded_preds = tokenizer.batch_decode(
            preds_list, skip_special_tokens=True
        )
        decoded_labels = tokenizer.batch_decode(
            labels_list, skip_special_tokens=True
        )

        # 5) Strip whitespace, wrap labels as [[ref]]
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [[lbl.strip()] for lbl in decoded_labels]

        # ---- BLEU & chrF on all examples ----
        bleu_res = bleu_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            tokenize="none",
            lowercase=False,
        )
        chrf_res = chrf_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
        )

        # ---- METEOR ----
        if meteor_only_for_english and meteor_target_mask_full is not None:
            meteor_mask = meteor_target_mask_full[:true_len]
            meteor_preds = [p for p, m in zip(decoded_preds, meteor_mask) if m]
            meteor_refs = [lbl[0] for lbl, m in zip(decoded_labels, meteor_mask) if m]
        else:
            # Either we don't care about English-only, or we don't have a mask
            meteor_preds = decoded_preds
            meteor_refs = [lbl[0] for lbl in decoded_labels]

        if len(meteor_preds) > 0:
            meteor_res = meteor_metric.compute(
                predictions=meteor_preds,
                references=meteor_refs,
            )
            meteor_score = float(round(meteor_res["meteor"], 4))
        else:
            meteor_score = float("nan")

        # ---- Average generated length in tokens (excluding pad) ----
        gen_len = float(
            round(
                np.mean(
                    [np.count_nonzero(np.array(p) != pad_token_id) for p in preds_list]
                ),
                4,
            )
        )

        result = {
            "bleu": float(round(bleu_res["score"], 4)),
            "chrf": float(round(chrf_res["score"], 4)),
            "meteor": meteor_score,
            "gen_len": float(round(gen_len, 2)),
        }

        gc.collect()
        return result

    return compute_metrics
