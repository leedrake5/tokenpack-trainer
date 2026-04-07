# TokenPackTrainer: Throughput, OOM, and Fragmentation Architecture

## Design Philosophy

The trainer navigates three competing pressures:

1. **Throughput**: Larger microbatches saturate the GPU (96% utilization) but risk OOM
2. **Safety**: Smaller microbatches avoid OOM but starve the GPU (15-20% utilization)
3. **Fragmentation**: Near-full VRAM doesn't OOM but makes allocations crawl (200-400s/step)

The system resolves this by learning a **proven stable operating point** (HWM) per regime and always falling back there — not to minimums, not to the edge.

---

## Lessons Learned (Hard-Won)

These lessons cost days of debugging and should not be relearned:

### 1. The autotune safety factor determines everything
Previous values (0.85-0.92) targeted 95-98% total VRAM utilization. This caused CUDA allocator pressure — allocations succeed but crawl through fragmented blocks. Steps go from 2s to 200s with no OOM error. Current values (0.66-0.75) target ~80% total utilization.

### 2. Never autotune from retry attempts
When a step OOMs and the retry succeeds with tiny microbatches, the peak memory is artificially low. Autotuning from this produces garbage `bytes_per_token` (~1000 instead of ~300,000), which makes the system think tokens are free → sets T=max → next step OOMs → B shrinks → death spiral. Autotune only fires on `attempt == 0`.

### 3. B cascades are the main throughput killer
`_regime_on_oom` used to shrink B on every retry (4 per batch). Three failing batches = 12 shrinks: B=457→7. With B=7 and 9760 short examples, that's 1394 microbatches × 0.25s = 349s/step. Fix: B shrinks once per batch (`_b_shrunk_this_batch` flag); subsequent retries shrink T.

### 4. The microbatch cap must bump B, not just T
For regimes with many short examples (regime 1), B is the bottleneck — not T. `25000 examples / B=171 = 151 microbatches` regardless of T. The cap now bumps B too, but conservatively (4× regime B) to avoid OOM from over-aggressive B bumps.

### 5. GPU baseline must use latest value, not max
On checkpoint resume, optimizer states briefly touch GPU (~93GB) then get freed/reorganized. The max-based baseline captured this transient spike and locked it in forever, making the autotune think there's no VRAM headroom. Fix: use latest value (each measurement overwrites the previous).

### 6. Autotuned keys must be cleared on resume
Checkpoint loads `autotuned_keys` — every regime marked as "already tuned." If the safety factor changed between runs, the old (aggressive) T values persist forever because the autotune never re-runs. Fix: always clear `autotuned_keys` on resume so regimes recalibrate.

### 7. Baseline capture must use local step count, not global
`global_step` after resume is e.g. 21537, so `global_step <= 5` is never true. The baseline capture never runs, leaving stale values from the checkpoint. Fix: use `_timing_step_count` which resets each run.

### 8. The watchdog should only defragment, never shrink
The memory pressure watchdog originally called `_regime_on_oom` on slow steps. This caused cascading B shrinks (firing every 3 steps) that killed throughput. The watchdog now only calls `empty_cache()` — defragmentation without regime damage.

### 9. HWM must track both B and T
`hwm_T` existed but `hwm_B` didn't. After OOM, B cascaded to minimums with no reference for recovery. With `hwm_B`, the OOM handler falls back to the proven-stable B, and the fast ramp targets it directly.

### 10. Don't ramp above the proven level
The normal ramp (every 2 steps at 40% B growth) pushes past the proven-stable point, causes OOM, crashes, recovers, then repeats — a "sugar high" cycle. Once a regime reaches its HWM, it stops ramping. The autotune recalibrates T on each first-attempt success; if conditions genuinely improve (more free VRAM), the autotune raises T and that becomes the new HWM naturally. But the ramp never pushes B/T beyond what's been proven to work.

---

## Data Flow

```
Dataset
  │
  ▼
LengthBucketedBatchSampler
  │  Groups by length, packs where sum(lengths) ≤ max_tokens_per_batch
  │
  ▼
Collator (T5SpanCorruption or CappedSeq2Seq)
  │  Tokenizes, truncates, dynamic padding, attaches input_length
  │
  ▼
training_step
  │
  ├─ Baseline capture (steps 0-5, latest value, per-device)
  │
  ├─ Compute lengths on CPU (enc_len, dec_len)
  ├─ Compute regime key: ceil(max(enc + alpha*dec) / 128)
  ├─ Apply regime limits (B, T from learned state)
  ├─ Plan microbatches (greedy pack, compact padding)
  │
  ├─ Microbatch cap (attempt 0 only):
  │    If too many microbatches: bump T and B (B capped at 4× regime B)
  │
  ├─ Pre-flight checks:
  │    1. lm_head estimate (vocab > 64K)
  │    2. Predictive VRAM: bytes_per_token × worst_mb vs total × safety
  │
  ├─ Execute microbatches:
  │    Per-microbatch OOM → mid-batch recovery (keep partial gradients)
  │
  ├─ On success (attempt 0 only): autotune → learn bytes_per_token, set T
  ├─ On success: regime_on_success → update HWM, ramp B/T
  │
  ├─ Step diagnostic (debug=True): print regime state, VRAM, timing
  ├─ Sustained slowdown watchdog: defrag after 120s of slow steps
  │
  └─ On OOM: shrink B (once per batch), then T → retry
```

---

## The Regime System

### What Is a Regime

Regimes group sequences by effective length into 128-token buckets. Each regime independently learns optimal (B, T) limits.

```
effective_length = enc_len + alpha × dec_len
regime_key = ceil(max(effective_length) / 128)
```

### Regime State

| Field | Type | Description |
|-------|------|-------------|
| `B` | int or None | Max examples per microbatch |
| `T` | int | Max tokens per microbatch |
| `stable` | int | Consecutive successes since last OOM |
| `hwm_T` | int or None | Proven-stable T (captured at stable ≥ 3/5) |
| `hwm_B` | int or None | Proven-stable B (captured at stable ≥ 3/5) |
| `bytes_per_token` | float or None | Learned marginal VRAM cost per effective token |

### The Stable Operating Point (HWM)

The HWM pair (hwm_B, hwm_T) represents **where the regime ran reliably**. It's:
- Captured when `stable ≥ 5` (on success) or `stable ≥ 3` (before OOM reset)
- The primary fallback target on OOM (not minimums)
- The ceiling for fast recovery ramp (don't exceed what worked)
- Persisted in checkpoints across runs

---

## Autotune: Learning bytes_per_token

After each successful step on `attempt == 0` (not retries), measures actual VRAM:

```
baseline        = model + optimizer memory (captured during steps 0-5)
available       = total_vram - baseline
batch_mem       = peak_memory - baseline        # marginal batch cost
bytes_per_token = batch_mem / eff_tokens_in_step
target_T        = (safety × available) / bytes_per_token
```

**Guards:**
- Only on `attempt == 0` (retries produce garbage: tiny microbatches → low peak → inflated target)
- Skipped on mid-batch recovery (partial peak data)
- Skipped on tiny post-OOM batches (`eff_tokens < hwm_T × 0.1`)
- Uses `_bottleneck_gpu_stats` for multi-GPU (smallest headroom device)

### Safety Factors

| | ≥80 GB | 48–80 GB | 24–48 GB | <24 GB |
|--|--------|----------|----------|--------|
| Normal | 0.75 | 0.72 | 0.70 | 0.66 |
| Post-OOM | 0.66 | 0.60 | 0.55 | 0.50 |

---

## OOM Handling

### Detection

- `torch.cuda.OutOfMemoryError` exception type
- String patterns: `"CUDA out of memory"`, `"CUBLAS_STATUS_ALLOC_FAILED"`, `"cudaMalloc"`

### The Retry Loop

```
_b_shrunk_this_batch = False

for attempt in 0 .. oom_max_retries (3):
    try:
        Plan → Cap (attempt 0) → Pre-flight → Execute
        Autotune (attempt 0 only)
        return loss
    except OOM:
        cleanup (zero grad, empty cache, gc.collect(0))
        _regime_on_oom(shrink_b = not _b_shrunk_this_batch)
        _b_shrunk_this_batch = True

→ skip batch (return 0 loss) or raise
```

### Three Recovery Levels

**Level 1 — Mid-Batch Recovery:**
OOM after some microbatches completed. Keep accumulated gradients (partial batch is better than zero). Shrink regime. Complete step immediately.

**Level 2 — Full Retry:**
OOM on first microbatch. Zero gradients, shrink regime, re-plan, retry same batch.

**Level 3 — Skip Batch:**
All retries exhausted. Return zero loss, training continues.

### How B and T Shrink (`_regime_on_oom`)

**B shrink (first retry only per batch):**

| Current B vs hwm_B | Action |
|---------------------|--------|
| B is None (uncapped) | Materialize to `_regime_max_B` |
| B > hwm_B | Drop to hwm_B |
| B > hwm_B × 0.85 | Drop to hwm_B × 0.85 |
| B ≤ hwm_B (or no hwm) | Proportional: B × 0.7 |

**T shrink (subsequent retries):**

| Current T vs hwm_T | Action |
|---------------------|--------|
| T > hwm_T | Drop to hwm_T |
| T > hwm_T × 0.85 | Drop to hwm_T × 0.85 |
| T ≤ hwm_T (or no hwm) | Proportional: T × 0.85 |

**T never drops below `_safe_floor_T`:**
```
safe_floor = (0.66 × available_vram) / max(bytes_per_token across regimes)
```
Falls back to `oom_min_tokens` (64) when bytes_per_token is unknown.

### What Gets Invalidated on OOM
- `stable` → 0
- `bytes_per_token` → None (stale calibration can't be trusted)
- Regime removed from `autotuned_keys` (force re-autotune on next success)
- `_regime_oom_count[key]` incremented (triggers conservative safety on re-autotune)

---

## Recovery: HWM + Fast Ramp

After OOM shrinks B/T, the **fast recovery ramp** targets the HWM (proven stable point):

| | Fast Recovery | At/Above HWM |
|--|---|---|
| **Trigger** | B < hwm_B × 0.9 OR T < hwm_T × 0.9 | At proven level |
| **Frequency** | Every 5 successes | No ramp |
| **Growth** | 20% per ramp | Held steady |
| **Ceiling** | hwm (proven level) | Autotune adjusts T naturally |

**Example recovery from B=44 toward hwm_B=64:**
```
stable=5:  B = min(44 * 1.2 + 1, 64) = 53
stable=10: B = min(53 * 1.2 + 1, 64) = 64  ← recovered in ~10 steps
```

---

## Microbatch Cap

When regime limits produce too many microbatches, the cap bumps both B and T for that step:

```
step_T = max(eff_tokens / max_mb + 1, regime_T)
step_B = min(ideal_B, max(regime_B × 4, regime_B + 32))
```

**B is capped at 4× the regime's current B** to prevent OOM from over-aggressive bumps. If the cap'd B still produces too many microbatches, accept the extras — the OOM retry provides safety.

**Only applies on attempt 0.** On retries, the shrunken regime limits are used directly.

---

## Sustained Slowdown Watchdog

Detects CUDA fragmentation (slow allocations without OOM):

- Tracks **baseline** (fastest step ever, never drifts up)
- Accumulates wall-clock time of steps ≥10× baseline AND ≥10s absolute
- After **120 seconds** of accumulated slow time (and 100-step cooldown): calls `empty_cache()`
- **Never shrinks the regime** — only defragments

---

## Pre-Flight Checks

### Predictive VRAM Check (sync-free)

Before executing microbatches, estimates peak memory using cached values:

```
predicted_peak = baseline + bytes_per_token × max(mb_eff_tokens)
safe_limit     = total_vram × autotune_safety
```

If predicted > safe: synthetic OOM → regime shrink before any GPU work.
Skipped when `bytes_per_token` is None (graceful fallback).

### lm_head Check (large-vocab models)

For vocab > 64K, estimates lm_head output tensor size. If it exceeds 85% of free VRAM on the bottleneck GPU, synthetic OOM.

---

## Checkpoint Persistence

### What's Saved (`regime_state.json`)

```json
{
  "regime_limits": {
    "1": {"B": 64, "T": 269632, "stable": 8, "hwm_T": 786432, "hwm_B": 76, "bytes_per_token": 303496.6}
  },
  "autotuned_keys": [],
  "regime_oom_count": {"1": 0},
  "oom_events": 42,
  "oom_skipped_batches": 5,
  "gpu_baseline_mem": 11060000000,
  "gpu_baseline_mem_per_device": {"0": 11060000000}
}
```

### What Happens on Resume

- Regime limits restored (B, T, stable, hwm_T, hwm_B, bytes_per_token)
- `autotuned_keys` **always cleared** → forces re-autotune with current safety factors
- Baseline captured fresh (local step counter, not global)
- OOM history and counters restored

---

## Step Diagnostic (debug=True)

Every step prints:

```
[TokenPackTrainer] Step 7 [SLOW]: 10.8s (baseline=1.1s) |
  regime=1, B=76, T=269632, hwm_T=786432, stable=6, bpt=303496.6 |
  microbatches=33, examples=9767, tokens=49718, eff_tokens=250749 |
  VRAM=13772/97246MB
```

Shows regime key, current limits, HWM, stability count, learned bytes_per_token, microbatch count, token counts, and GPU memory — everything needed to diagnose throughput issues without guessing.

---

## VRAM-Scaled Defaults

| Parameter | ≥80 GB | 48–80 GB | 24–48 GB | <24 GB |
|-----------|--------|----------|----------|--------|
| `ramp_every` | 2 | 3 | 4 | 6 |
| `ramp_B` | 1.40 | 1.30 | 1.25 | 1.15 |
| `ramp_T` | 1.20 | 1.15 | 1.10 | 1.05 |
| `autotune_safety` | 0.75 | 0.72 | 0.70 | 0.66 |
| `autotune_oom_safety` | 0.66 | 0.60 | 0.55 | 0.50 |
| `eval_ramp_every` | 10 | 15 | 20 | 50 |
| `eval_ramp_T` | 1.10 | 1.08 | 1.05 | 1.03 |

---

## Key Parameters

### Batching

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens_per_microbatch` | 400 | GPU memory budget per microbatch |
| `max_tokens_per_batch` | None | Sampler budget |
| `max_examples_per_microbatch` | None | Hard cap on examples per microbatch |
| `max_encoder_len` | None | Hard truncation for encoder |
| `max_decoder_len` | None | Hard truncation for decoder |
| `use_cpu_microbatch` | True | Plan on CPU, prefetch to GPU |
| `padding_aware_budget` | False | max_len × count vs sum(lengths) |
| `max_microbatches_per_step` | 8 | Cap on microbatches per step |

### OOM Recovery

| Parameter | Default | Description |
|-----------|---------|-------------|
| `oom_max_retries` | 3 | Retry attempts per batch |
| `oom_shrink_B` | 0.5 | B shrink factor (global fallback) |
| `oom_shrink_tokens` | 0.85 | T shrink factor (global fallback) |
| `oom_min_B` | 1 | Floor for B |
| `oom_min_tokens` | 64 | Floor for T |
| `oom_skip_batch_on_fail` | True | Skip batch vs raise after exhausting retries |

### Adaptive Regime

| Parameter | Default | Description |
|-----------|---------|-------------|
| `regime_ramp_every` | "auto" | Steps between ramps (VRAM-scaled) |
| `regime_ramp_B` | "auto" | B growth factor (VRAM-scaled) |
| `regime_ramp_T` | "auto" | T growth factor (VRAM-scaled) |
| `autotune_safety` | "auto" | Safety factor for autotune (VRAM-scaled) |
| `autotune_oom_safety` | "auto" | Post-OOM safety factor (VRAM-scaled) |
| `regime_max_B` | None | Ceiling for B (auto: max(1024, token_budget)) |
