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


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Trim to same number of examples
    n = labels.shape[0]
    preds  = preds[:n]
    labels = labels[:n]

    # IMPORTANT: restore label pad positions before decoding
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    labels = np.where(labels != -100, labels, pad_id)

    # (Optional) If you ever see truly invalid pred ids, clip *preds only*.
    # But I'd rather you assert than silently clip.
    # preds = np.clip(preds, 0, tokenizer.vocab_size - 1)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

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

    # METEOR: WARNING — mask likely misaligned if eval order changes!
    meteor_mask = meteor_target_mask_full[:n]
    meteor_preds = [p for p, m in zip(decoded_preds, meteor_mask) if m]
    meteor_refs  = [lbl[0] for lbl, m in zip(decoded_labels, meteor_mask) if m]

    if meteor_preds:
        meteor_res = meteor_metric.compute(predictions=meteor_preds, references=meteor_refs)
        meteor_score = float(meteor_res["meteor"])
    else:
        meteor_score = float("nan")

    # gen_len from token ids (exclude pad)
    gen_len = float(np.mean(np.sum(preds != pad_id, axis=1)))

    return {
        "bleu": float(round(bleu_res["score"], 4)),
        "chrf": float(round(chrf_res["score"], 4)),
        "meteor": float(round(meteor_score, 4)) if meteor_score == meteor_score else float("nan"),
        "gen_len": float(round(gen_len, 2)),
    }
