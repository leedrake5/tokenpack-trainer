"""
Length-bucketed batch sampling for token-aware training.

This module provides LengthBucketedBatchSampler, which groups dataset examples
by sequence length and packs them into batches based on token count rather than
example count. This minimizes padding waste for variable-length sequences.

The sampler works in two phases:
1. Bucketing: Group examples by length into buckets (e.g., lengths 1-16, 17-32, ...)
2. Packing: Fill batches within each bucket until token budget is reached

This approach is particularly effective for datasets with high length variance,
such as cuneiform texts (avg ~13 tokens, some reaching thousands).
"""

import math
import random
from collections import defaultdict
from torch.utils.data import Sampler


class LengthBucketedBatchSampler(Sampler):
    """
    Batch sampler that groups examples by length and packs to a token budget.

    Instead of fixed batch_size, this sampler ensures:
        sum(lengths in batch) <= max_tokens_per_batch

    This eliminates wasted computation on padding tokens, especially for
    datasets with highly variable sequence lengths.

    Parameters:
    -----------
    lengths : Iterable[int]
        Sequence length for each example in the dataset. Must be indexable
        with the same ordering as the dataset.

    max_tokens_per_batch : int
        Maximum total tokens per batch. Batches are filled greedily until
        this budget would be exceeded.

    bucket_size : int, default=16
        Width of length buckets. Smaller = tighter grouping, less padding.
        For short sequences (avg <20 tokens), try 4-8.

    shuffle : bool, default=True
        Whether to shuffle bucket order and examples within buckets.

    drop_last : bool, default=False
        Whether to drop incomplete final batches.

    long_behavior : {"truncate", "skip", "single"}, default="truncate"
        How to handle examples longer than max_tokens_per_batch:
        - "truncate": Include in batch (will be truncated by collator)
        - "skip": Exclude from all batches
        - "single": Put in a batch by itself

    max_length_in_batch : int, optional
        Maximum sequence length allowed in any batch. Examples exceeding
        this are handled according to long_behavior.

    Example:
    --------
        >>> lengths = [len(ex["input_ids"]) for ex in dataset]
        >>> sampler = LengthBucketedBatchSampler(
        ...     lengths=lengths,
        ...     max_tokens_per_batch=4096,
        ...     bucket_size=8,
        ... )
        >>> dataloader = DataLoader(dataset, batch_sampler=sampler, ...)
    """

    def __init__(
        self,
        lengths,
        max_tokens_per_batch: int,
        bucket_size: int = 16,
        shuffle: bool = True,
        drop_last: bool = False,
        long_behavior: str = "truncate",
        max_length_in_batch: int | None = None,
        shuffle_mode: str = "bucket",
    ):
        assert long_behavior in {"truncate", "skip", "single"}
        assert shuffle_mode in {"bucket", "interleave"}, (
            f"shuffle_mode must be 'bucket' or 'interleave', got '{shuffle_mode}'"
        )

        self.lengths = [int(L) for L in lengths]
        self.max_tokens_per_batch = int(max_tokens_per_batch)
        self.bucket_size = int(bucket_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.long_behavior = long_behavior
        self.max_length_in_batch = int(max_length_in_batch) if max_length_in_batch is not None else None
        self.shuffle_mode = shuffle_mode

        buckets = defaultdict(list)
        for idx, L in enumerate(self.lengths):
            bucket_id = int(math.ceil(L / self.bucket_size))
            buckets[bucket_id].append(idx)

        self.bucket_ids = sorted(buckets.keys())
        self.buckets = {b: idxs for b, idxs in buckets.items()}

        # Lazy-computed batch count (deferred until __len__ is called)
        self._num_batches: int | None = None

    def _count_batches(self) -> int:
        """Count batches by simulating iteration without shuffling."""
        count = 0
        for b in self.bucket_ids:
            idxs = self.buckets[b]

            current_batch_size = 0
            current_tokens = 0
            current_max = 0

            for i in idxs:
                L = self.lengths[i]

                # Handle examples that exceed max_length_in_batch
                if self.max_length_in_batch is not None and L > self.max_length_in_batch:
                    if self.long_behavior == "skip":
                        continue
                    elif self.long_behavior == "single":
                        if current_batch_size > 0 and not self.drop_last:
                            count += 1
                        count += 1  # the single-item batch
                        current_batch_size, current_tokens, current_max = 0, 0, 0
                        continue

                # Handle examples that exceed max_tokens_per_batch
                if L > self.max_tokens_per_batch:
                    if self.long_behavior == "skip":
                        continue
                    elif self.long_behavior == "single":
                        if current_batch_size > 0 and not self.drop_last:
                            count += 1
                        count += 1
                        current_batch_size, current_tokens, current_max = 0, 0, 0
                        continue

                if current_batch_size == 0:
                    current_batch_size = 1
                    current_tokens = L
                    current_max = L
                    continue

                would_exceed_sum = (current_tokens + L > self.max_tokens_per_batch)
                would_exceed_max = (
                    self.max_length_in_batch is not None
                    and max(current_max, L) > self.max_length_in_batch
                )

                if would_exceed_sum or would_exceed_max:
                    if not self.drop_last:
                        count += 1
                    current_batch_size = 1
                    current_tokens = L
                    current_max = L
                else:
                    current_batch_size += 1
                    current_tokens += L
                    current_max = max(current_max, L)

            if current_batch_size > 0 and not self.drop_last:
                count += 1

        return max(1, count)

    def _pack_bucket(self, idxs):
        """Pack a single bucket's indices into batches using greedy token budgeting."""
        batches = []
        current_batch = []
        current_tokens = 0
        current_max = 0

        for i in idxs:
            L = self.lengths[i]

            # Handle examples that exceed max_length_in_batch
            if self.max_length_in_batch is not None and L > self.max_length_in_batch:
                if self.long_behavior == "skip":
                    continue
                elif self.long_behavior == "single":
                    if current_batch and not self.drop_last:
                        batches.append(current_batch)
                    batches.append([i])
                    current_batch, current_tokens, current_max = [], 0, 0
                    continue
                # "truncate": allow through (will be truncated later)

            # Handle examples that exceed max_tokens_per_batch
            if L > self.max_tokens_per_batch:
                if self.long_behavior == "skip":
                    continue
                elif self.long_behavior == "single":
                    if current_batch and not self.drop_last:
                        batches.append(current_batch)
                    batches.append([i])
                    current_batch, current_tokens, current_max = [], 0, 0
                    continue
                # "truncate": allow through

            if not current_batch:
                current_batch = [i]
                current_tokens = L
                current_max = L
                continue

            would_exceed_sum = (current_tokens + L > self.max_tokens_per_batch)
            would_exceed_max = (
                self.max_length_in_batch is not None
                and max(current_max, L) > self.max_length_in_batch
            )

            if would_exceed_sum or would_exceed_max:
                if not self.drop_last:
                    batches.append(current_batch)
                current_batch = [i]
                current_tokens = L
                current_max = L
            else:
                current_batch.append(i)
                current_tokens += L
                current_max = max(current_max, L)

        if current_batch and not self.drop_last:
            batches.append(current_batch)

        return batches

    def __iter__(self):
        bucket_ids = list(self.bucket_ids)
        if self.shuffle:
            random.shuffle(bucket_ids)

        if self.shuffle_mode == "interleave":
            # Collect ALL batches from ALL buckets, then shuffle globally.
            # This ensures consecutive batches come from random length buckets,
            # giving much more even GPU utilization with variable-length data.
            all_batches = []
            for b in bucket_ids:
                idxs = list(self.buckets[b])
                if self.shuffle:
                    random.shuffle(idxs)
                all_batches.extend(self._pack_bucket(idxs))

            if self.shuffle:
                random.shuffle(all_batches)

            yield from all_batches
        else:
            # Original bucket-sequential mode
            for b in bucket_ids:
                idxs = list(self.buckets[b])
                if self.shuffle:
                    random.shuffle(idxs)

                for batch in self._pack_bucket(idxs):
                    yield batch

    def __len__(self):
        if self._num_batches is None:
            self._num_batches = self._count_batches()
        return self._num_batches
