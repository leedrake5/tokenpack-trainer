import math
import random
from collections import defaultdict
from torch.utils.data import Sampler


class LengthBucketedBatchSampler(Sampler):
    """
    Batch sampler that:
      - Buckets examples by (rounded) length,
      - Packs indices into batches so that sum(lengths) <= max_tokens_per_batch.
    """

    def __init__(
        self,
        lengths,
        max_tokens_per_batch: int,
        bucket_size: int = 16,
        shuffle: bool = True,
        drop_last: bool = False,
        long_behavior: str = "truncate",  # How do we handle long sequences?: {"truncate", "skip", "single"}
    ):
        """
        long_behavior:
          - "truncate": allow into batch & let model/collator truncate (SAFE DEFAULT)
          - "skip": drop the example entirely 
          - "single": force into its own batch
        """
        assert long_behavior in {"truncate", "skip", "single"}

        self.lengths = [int(L) for L in lengths]
        self.max_tokens_per_batch = int(max_tokens_per_batch)
        self.bucket_size = int(bucket_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.long_behavior = long_behavior

        buckets = defaultdict(list)
        for idx, L in enumerate(self.lengths):
            bucket_id = int(math.ceil(L / self.bucket_size))
            buckets[bucket_id].append(idx)

        self.bucket_ids = sorted(buckets.keys())
        self.buckets = {b: idxs for b, idxs in buckets.items()}

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

            for i in idxs:
                L = self.lengths[i]

                if L > self.max_tokens_per_batch:
                    if self.long_behavior == "skip":
                        continue

                    elif self.long_behavior == "single":
                        if current_batch:
                            if not self.drop_last:
                                yield current_batch
                        yield [i]
                        current_batch = []
                        current_tokens = 0
                        continue

                    elif self.long_behavior == "truncate":
                        # Allow it through; truncation happens later in the collator/trainer
                        pass

                # Normal packing logic
                if current_batch and (current_tokens + L > self.max_tokens_per_batch):
                    if not self.drop_last:
                        yield current_batch
                    current_batch = [i]
                    current_tokens = L
                else:
                    current_batch.append(i)
                    current_tokens += L

            if current_batch and not self.drop_last:
                yield current_batch

    def __len__(self):
        n = len(self.lengths)
        avg_len = sum(self.lengths) / max(1, n)
        approx_per_batch = max(1, int(self.max_tokens_per_batch // max(1, int(avg_len))))
        return max(1, n // approx_per_batch)
