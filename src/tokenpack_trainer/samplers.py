import math
import random
from collections import defaultdict
from torch.utils.data import Sampler


class LengthBucketedBatchSampler(Sampler):
    """
    Buckets by length and packs indices so that:
      - sum(lengths in batch) <= max_tokens_per_batch
      - optionally: max(length in batch) <= max_length_in_batch
    """

    def __init__(
        self,
        lengths,
        max_tokens_per_batch: int,
        bucket_size: int = 16,
        shuffle: bool = True,
        drop_last: bool = False,
        long_behavior: str = "truncate",  # {"truncate", "skip", "single"}
        max_length_in_batch: int | None = None,
    ):
        assert long_behavior in {"truncate", "skip", "single"}

        self.lengths = [int(L) for L in lengths]
        self.max_tokens_per_batch = int(max_tokens_per_batch)
        self.bucket_size = int(bucket_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.long_behavior = long_behavior
        self.max_length_in_batch = int(max_length_in_batch) if max_length_in_batch is not None else None

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

    def __iter__(self):
        bucket_ids = list(self.bucket_ids)
        if self.shuffle:
            random.shuffle(bucket_ids)

        for b in bucket_ids:
            idxs = list(self.buckets[b])
            if self.shuffle:
                random.shuffle(idxs)

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
                            yield current_batch
                        yield [i]
                        current_batch, current_tokens, current_max = [], 0, 0
                        continue
                    # "truncate": allow through (will be truncated later)

                # Handle examples that exceed max_tokens_per_batch
                if L > self.max_tokens_per_batch:
                    if self.long_behavior == "skip":
                        continue
                    elif self.long_behavior == "single":
                        if current_batch and not self.drop_last:
                            yield current_batch
                        yield [i]
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
                        yield current_batch
                    current_batch = [i]
                    current_tokens = L
                    current_max = L
                else:
                    current_batch.append(i)
                    current_tokens += L
                    current_max = max(current_max, L)

            if current_batch and not self.drop_last:
                yield current_batch

    def __len__(self):
        # Lazy computation - only count batches when __len__ is first called
        if self._num_batches is None:
            self._num_batches = self._count_batches()
        return self._num_batches
