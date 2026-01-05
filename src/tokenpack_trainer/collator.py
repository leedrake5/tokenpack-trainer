import math
import numpy as np
import torch
from typing import List, Dict, Any
from transformers import PreTrainedTokenizerBase

def shift_tokens_right(input_ids: torch.Tensor,
                       pad_token_id: int,
                       decoder_start_token_id: int) -> torch.Tensor:
    """
    Shift input ids one token to the right, and replace -100 in the
    labels by pad_token_id (so that cross‐entropy ignores it).
    """
    # make a new tensor full of pad tokens
    shifted = input_ids.new_zeros(input_ids.shape)

    # copy everything except the last token
    shifted[..., 1:] = input_ids[..., :-1].clone()
    # first token is the decoder start token
    shifted[..., 0] = decoder_start_token_id

    # replace -100 (ignore_index) by pad_id so they’re not completely lost
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted


class T5SpanCorruptionCollatorFast:
    """
    Faster span corruption collator following the T5 'random spans' recipe.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        noise_density: float = 0.15,
        mean_noise_span_length: float = 3.0,
        input_length: int = 64,
        target_length: int = 64,
        pad_to_multiple_of: int = 8,
        dynamic_inputs: bool = True,
        max_sentinels: int = 128,
        add_eos_to_target: bool = True,
        rng: np.random.Generator = None,
    ):
        self.tok = tokenizer
        self.noise_density = float(noise_density)
        self.mean_noise_span_length = float(mean_noise_span_length)
        self.input_length = int(input_length)
        self.target_length = int(target_length)
        self.p2m = int(pad_to_multiple_of) if pad_to_multiple_of else None
        self.dynamic_inputs = bool(dynamic_inputs)
        self.max_sentinels = int(max_sentinels)
        self.add_eos_to_target = bool(add_eos_to_target)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.pad_id = tokenizer.pad_token_id
        self.eos_id = getattr(tokenizer, "eos_token_id", None)

        # --- UL2 / T5Gemma-aware sentinel ids ---
        if hasattr(tokenizer, "get_sentinel_token_ids"):
            # For T5 / T5Gemma:
            sentinel_ids = tokenizer.get_sentinel_token_ids()
            self.sentinel_ids = sentinel_ids[: self.max_sentinels]
        else:
            # Fallback for older tokenizers: use extra_id_* strings
            self.sentinel_ids = [
                tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")
                for i in range(self.max_sentinels)
            ]

        # Make sure every sentinel id is in range
        for i, sid in enumerate(self.sentinel_ids):
            if not (0 <= sid < tokenizer.vocab_size):
                raise ValueError(
                    f"Sentinel id out of range: index {i} -> id {sid}, "
                    f"but vocab_size={tokenizer.vocab_size}"
                )

    # --------- helpers ---------

    def _random_spans(self, L: int):
        if L <= 0:
            return [L], []

        n_noise = max(1, int(round(L * self.noise_density)))
        n_nonnoise = max(0, L - n_noise)

        S = max(1, int(round(n_noise / self.mean_noise_span_length)))

        if n_noise >= S:
            noise_lengths = self.rng.multinomial(
                n_noise - S, [1.0 / S] * S
            ) + 1
        else:
            noise_lengths = np.ones(S, dtype=int)

        if (S + 1) > 0:
            if n_nonnoise >= (S + 1):
                nonnoise_lengths = self.rng.multinomial(
                    n_nonnoise - (S + 1),
                    [1.0 / (S + 1)] * (S + 1)
                ) + 1
            else:
                nonnoise_lengths = np.zeros(S + 1, dtype=int)
                if n_nonnoise > 0:
                    picks = self.rng.choice(
                        S + 1, size=n_nonnoise, replace=True
                    )
                    for p in picks:
                        nonnoise_lengths[p] += 1
        else:
            nonnoise_lengths = np.array([n_nonnoise], dtype=int)

        noise_total = int(noise_lengths.sum())
        non_total = int(nonnoise_lengths.sum())
        over = (noise_total + non_total) - L
        if over > 0:
            nonnoise_lengths[-1] = max(0, nonnoise_lengths[-1] - over)

        return nonnoise_lengths.tolist(), noise_lengths.tolist()

    def _sentinel(self, i: int) -> int:
        if i < len(self.sentinel_ids):
            return self.sentinel_ids[i]
        return self.sentinel_ids[-1]

    def _apply_spans(self, tokens: List[int], nonnoise: List[int], noise: List[int]):
        corrupted = []
        target = []
        idx = 0
        span_id = 0

        for gap_len, noise_len in zip(nonnoise, noise):
            if gap_len > 0:
                corrupted.extend(tokens[idx: idx + gap_len])
                idx += gap_len

            sentinel_id = self._sentinel(span_id)
            corrupted.append(sentinel_id)
            target.append(sentinel_id)

            if noise_len > 0:
                target.extend(tokens[idx: idx + noise_len])
                idx += noise_len

            span_id += 1

        if len(nonnoise) > len(noise):
            final_gap = nonnoise[-1]
            if final_gap > 0:
                corrupted.extend(tokens[idx: idx + final_gap])
                idx += final_gap

        if self.add_eos_to_target and (self.eos_id is not None):
            target.append(self.eos_id)

        return corrupted, target

    # --------- main collate ---------

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        B = len(examples)
        corrupted_list = []
        target_list = []

        for ex in examples:
            toks: List[int] = ex["input_ids"]
            L = len(toks)

            nonnoise, noise = self._random_spans(L)
            corrupted, target = self._apply_spans(toks, nonnoise, noise)

            corrupted_list.append(corrupted)
            target_list.append(target)

        # ---- enforce a *hard* cap on input length ----
        if self.dynamic_inputs:
            raw_max_in = max(len(x) for x in corrupted_list) if corrupted_list else 0
            max_in = min(raw_max_in, self.input_length)
        else:
            max_in = self.input_length

        # targets: already partially clamped
        max_tgt = max(len(x) for x in target_list) if target_list else 0
        max_tgt = min(self.target_length, max_tgt)

        if self.p2m:
            if max_in % self.p2m != 0:
                max_in = int(math.ceil(max_in / self.p2m) * self.p2m)
                # but never exceed the global input_length
                max_in = min(max_in, self.input_length)

            if max_tgt % self.p2m != 0:
                max_tgt = int(math.ceil(max_tgt / self.p2m) * self.p2m)
                max_tgt = min(max_tgt, self.target_length)

        inputs = torch.full((B, max_in), self.pad_id, dtype=torch.long)
        labels = torch.full((B, max_tgt), self.pad_id, dtype=torch.long)

        for i, (cin, tgt) in enumerate(zip(corrupted_list, target_list)):
            li = min(len(cin), max_in)
            lt = min(len(tgt), max_tgt)
            if li > 0:
                inputs[i, :li] = torch.tensor(cin[:li], dtype=torch.long)
            if lt > 0:
                labels[i, :lt] = torch.tensor(tgt[:lt], dtype=torch.long)

        attention_mask = (inputs != self.pad_id).long()
        labels[labels == self.pad_id] = -100

        # sanity check (optional but nice during debugging)
        assert inputs.shape[1] <= self.input_length, inputs.shape
        assert labels.shape[1] <= self.target_length, labels.shape

        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_length": attention_mask.sum(dim=-1),  # pre-computed for token-aware batching
        }

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union

import torch
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase, PreTrainedModel


def _truncate_list_like(x, max_len: int):
    """
    Truncate something that might be a list, numpy array, or tensor,
    and always return a plain Python list of ints.
    """
    if isinstance(x, torch.Tensor):
        x = x.tolist()
    # If it's None, just return as-is
    if x is None:
        return x
    return x[:max_len]


@dataclass
class CappedSeq2SeqCollator:
    """
    Thin wrapper around HF's DataCollatorForSeq2Seq that:
      - Caps encoder length to `input_length`,
      - Caps decoder/label length to `target_length`,
      - Then defers to DataCollatorForSeq2Seq for padding, shifting, etc.
    """

    base_collator: DataCollatorForSeq2Seq
    input_length: int = 512
    target_length: int = 512

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        capped_features: List[Dict[str, Any]] = []

        for f in features:
            f_new = dict(f)  # shallow copy

            # --- Truncate encoder side ---
            if "input_ids" in f_new and f_new["input_ids"] is not None:
                ids = _truncate_list_like(f_new["input_ids"], self.input_length)
                f_new["input_ids"] = ids

                if "attention_mask" in f_new and f_new["attention_mask"] is not None:
                    am = _truncate_list_like(f_new["attention_mask"], self.input_length)
                    f_new["attention_mask"] = am

            # --- Truncate labels / decoder side ---
            if "labels" in f_new and f_new["labels"] is not None:
                lbl = _truncate_list_like(f_new["labels"], self.target_length)
                f_new["labels"] = lbl

            capped_features.append(f_new)

        batch = self.base_collator(capped_features)

        # Add pre-computed input_length for token-aware batching
        if "attention_mask" in batch and isinstance(batch["attention_mask"], torch.Tensor):
            batch["input_length"] = batch["attention_mask"].sum(dim=-1)

        return batch
