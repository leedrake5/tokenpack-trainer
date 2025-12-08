# TokenPack Trainer

A lightweight toolkit for token-budgeted batching and microbatching
with Hugging Face Transformers.\
Designed for long-sequence pretraining, seq2seq translation, and mixed-length corpora where fixed batch sizes waste memory. Crucially, most transformers-style syntax can still be used, this is a drop-in patch to improve memory efficiency.

This package provides:

-   `HierarchicalTokenTrainer` --- token-aware, OOM-safe Trainer with
    CPU/GPU microbatching\
-   `T5SpanCorruptionCollatorFast` --- efficient span corruption for
    T5-style pretraining\
-   `CappedSeq2SeqCollator` --- hard-capped dynamic padding for
    seq2seq fine-tuning

------------------------------------------------------------------------

## Installation

From source:

``` bash
pip install -e .
```

From GitHub (example):

``` bash
pip install git+https://github.com/yourname/tokenpack-trainer.git
```

------------------------------------------------------------------------

## Core Concepts

### Why Token-Based Batching?

Traditional training uses:

    batch_size × max_sequence_length

This wastes VRAM when sequences vary widely. Token-based batching
instead enforces:

    sum(tokens_in_batch) ≤ token_budget

This allows:

-   Packing many short samples together
-   Automatically shrinking batches for long samples
-   Stable VRAM usage across wildly variable datasets

------------------------------------------------------------------------

## 1. HierarchicalTokenTrainer

A drop-in replacement for `transformers.Seq2SeqTrainer` that supports:

-   Token-based microbatching\
-   Optional CPU-planned → GPU-executed microbatches\
-   Optional hard truncation caps\
-   OOM-safe checkpointing\
-   Token-aware evaluation\
-   Compatible with:
    -   T5 / mT5 / UMT5
    -   NLLB-style seq2seq models
    -   Flash-Attention models
    -   DeepSpeed & Accelerate

------------------------------------------------------------------------

### Minimal Usage

``` python
from tokenpack_trainer import HierarchicalTokenTrainer

trainer = HierarchicalTokenTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,

    max_tokens_per_batch=4096,
    max_tokens_per_microbatch=512,
    length_column_name="input_length",
    use_cpu_microbatch=True,
)
```

------------------------------------------------------------------------

### Common Advanced Configuration

``` python
trainer = HierarchicalTokenTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,

    max_tokens_per_batch=8192,
    max_tokens_per_microbatch=1024,
    max_eval_tokens_per_microbatch=512,

    max_encoder_len=512,
    max_decoder_len=256,

    length_column_name="input_length",
    use_cpu_microbatch=True,

    log_longest_every=100,
    debug=False,
)
```

------------------------------------------------------------------------

## 2. T5SpanCorruptionCollatorFast

A fast, production-ready data collator for T5-style span corruption
pretraining.

Implements:

-   Random span masking
-   Sentinel token insertion
-   Encoder-target corruption format
-   Dynamic padding
-   Length tracking for token-based trainers

### Usage Example

``` python
from tokenpack_trainer.collators import T5SpanCorruptionCollatorFast

collator = T5SpanCorruptionCollatorFast(
    tokenizer=tokenizer,
    noise_density=0.15,
    mean_noise_span_length=3.0,
    pad_to_multiple_of=8,
    length_column_name="input_length",
)
```

------------------------------------------------------------------------

## 3. CappedSeq2SeqCollator

A dynamic padding collator for translation / instruction tuning
with:

-   Per-batch padding
-   Optional hard caps
-   Optional decoder input control
-   Length tracking for token-based batching

### Usage Example

``` python
from tokenpack_trainer.collators import CappedSeq2SeqCollator

collator = CappedSeq2SeqCollator(
    tokenizer=tokenizer,
    max_encoder_len=512,
    max_decoder_len=256,
    pad_to_multiple_of=8,
    length_column_name="input_length",
)
```

------------------------------------------------------------------------

## Metrics Integration

Use the provided metrics factory:

``` python
from tokenpack_trainer.metrics import make_seq2seq_compute_metrics

compute_metrics = make_seq2seq_compute_metrics(
    tokenizer=tokenizer,
    eval_dataset=eval_dataset,
    source_column="source",
    meteor_only_for_english=True,
)

trainer = HierarchicalTokenTrainer(
    ...,
    compute_metrics=compute_metrics,
)
```

------------------------------------------------------------------------

## Supported Training Types

  Task Type                Supported
  ------------------------ -----------
  T5 Pretraining           ✅
  Translation              ✅
  Instruction Tuning       ✅
  Mixed-Length Corpora     ✅
  Flash-Attention Models   ✅
  DeepSpeed                ✅
  Token Packing            ✅

------------------------------------------------------------------------

## License

MIT License (or update to match your project).
