# Computational cost - P1-P5 runtime, memory, and complexity

Measured post hoc from the wandb runs of the screening phase (groups
`proposal{1-5}_corrected`): wall-clock from each run's `_runtime` summary,
memory from the wandb system-metric stream (`system.gpu.0.memoryAllocatedBytes`,
`system.proc.memory.rssMB`). No re-runs were needed.

**Setup**: fixed-lambda, 100 epochs (+10 warm-up), the 42 datasets shared by all
five groups (the 7 screening/extension collections). One run per (proposal,
dataset) cell. Host: PSC Bridges-2 GPU nodes (shared).

## Wall-clock per run (seconds, 100 epochs, fixed lambda)

| | P1 residual | P2 uncertainty | P3 consistency | P4 dual-grad | P5 persistence |
|---|--:|--:|--:|--:|--:|
| mean s/run (42 datasets) | 101.4 | 107.4 | 72.1 | 93.5 | 76.2 |
| median s/run | 28.5 | 26.5 | 16.7 | 22.3 | 17.5 |
| ratio vs P1 (per-dataset mean) | 1.00 | 1.06 | 0.68 | 0.89 | 0.99 |
| auto-lambda / fixed ratio | 0.99 | 0.96 | 1.01 | 0.98 | 0.96 |

## vs plain RW-1 (the baseline)

The gate-off control runs from the P2 phase (`RW1m-*`, plain RW-1, 100 ep, same
pipeline) provide a measured baseline on 10 datasets shared with all five
proposal groups (GECCO, CreditCard, OPPORTUNITY x8):

| collection | RW-1 | P1 | P2 | P3 | P4 | P5 |
|---|--:|--:|--:|--:|--:|--:|
| CreditCard | 364 | 667 | 806 | 511 | 676 | 522 |
| GECCO | 206 | 290 | 362 | 209 | 296 | 232 |
| OPPORTUNITY | 62 | 104 | 98 | 69 | 94 | 71 |
| mean s/run (10 datasets) | 106.9 | 178.7 | 195.1 | 127.5 | 172.0 | 132.0 |
| **ratio vs RW-1** | **1.00** | **2.07** | **2.09** | **1.28** | **1.68** | **1.34** |
| **overhead vs RW-1** | - | **+107%** | **+109%** | **+28%** | **+68%** | **+34%** |

**vs DeepAnT**: not measured - the DeepAnT reproduction predates the wandb
tracking, so no runtimes were logged (and none exist in its result CSVs).
Qualitatively, RW-1 is DeepAnT's CNN plus the correction machinery (a
corrected-input forward and one RMSprop step per epoch), so RW-1's wall-clock
is a small constant factor above DeepAnT's; the gate overheads above stack on
top of that. Measuring it exactly would need re-running the DeepAnT
reproduction (`deepant/reproduction/submit_cnn.sh`) with tracking enabled.

So gating costs roughly **1.3x to 2.1x plain RW-1 wall-clock**. The split is
instructive: P1/P2 (the `rw_cegar` base) pay ~2x because that base recomputes
buffer statistics (residual quantiles etc.) every batch, while P3/P5 (the hooks
base) only look up a precomputed epoch-end map in-batch and land at ~1.3x. P4
adds its extra forward+backward on top of a map-free in-batch gate (~1.7x). In
other words the overhead is dominated by the per-batch statistics machinery,
not by the gate math itself, and stays within the same order of magnitude as
RW-1 in every case.

Per-collection mean seconds (fixed lambda, 100 ep):

| collection | P1 | P2 | P3 | P4 | P5 |
|---|--:|--:|--:|--:|--:|
| CreditCard | 667 | 806 | 511 | 676 | 522 |
| GECCO | 290 | 362 | 209 | 296 | 232 |
| MSL | 12 | 14 | 8 | 10 | 14 |
| OPPORTUNITY | 104 | 98 | 69 | 94 | 71 |
| PSM | 484 | 573 | 381 | 495 | 398 |
| SWaT | 624 | 710 | 512 | 628 | 517 |
| TAO | 42 | 27 | 17 | 23 | 18 |

## Peak memory (sampled runs)

Peak over the run, from the wandb system stream (30 s sampling). Values are
per-GPU allocated bytes and process RSS; on shared nodes the GPU figure can
include co-tenants, so read it as an upper bound.

| dataset | | P1 | P2 | P3 | P4 | P5 |
|---|---|--:|--:|--:|--:|--:|
| OPPORTUNITY id1 | GPU MB | 713 | 715 | 765 | 746 | 715 |
| | RSS MB | 1466 | 1469 | 1459 | 1446 | 1454 |
| CreditCard id1 | GPU MB | 683 | 685 | 750 | 688 | 683 |
| | RSS MB | 1559 | 1563 | 1575 | 1565 | 1586 |
| SWaT id1 | GPU MB | 501 | 501 | 501 | 503 | 501 |
| | RSS MB | 1369 | 1375 | 1393 | 1373 | 1373 |

**Memory takeaway: the gates add essentially nothing.** Peaks differ by at most
~70 MB GPU / ~30 MB RSS across proposals on the same dataset; the model, the
correction tensor and the data loader dominate. P3 is the (slightly) heaviest
because it keeps the previous epoch's correction and delta history (two extra
O(T x feats) arrays).

## Where the extra computation actually is (mechanism)

Base cost per epoch: forward + backward over all windows, O(n_windows) batches;
the correction update itself is one RMSprop step per epoch, O(T x feats).

| proposal | extra work per BATCH | extra work per EPOCH end | expected overhead |
|---|---|---|---|
| P1 | robust-z + 2 sigmoids on [B] tensors | - | negligible |
| P2 | M=5 extra MC-dropout FORWARD passes (no grad) | - | largest per-batch add; forwards are cheap vs backward, observed ~+6% |
| P3 | previous-epoch map lookup | d, v, gate map + preserve scaling, O(T) numpy | negligible per batch |
| P4 | 1 extra forward + backward w.r.t. the INPUT | write-back step, O(T x feats) | in principle the priciest per batch (~+30-60% of batch compute); not visible above node noise in practice |
| P5 | ledger scatter-add, O(B) numpy | quantile + convolve + EMA, O(T) | negligible |

## Caveats (read before quoting numbers)

- Wall-clock on a **shared** cluster: runs landed on different Bridges-2 nodes at
  different times, so cross-proposal wall-clock ratios mix gate overhead with
  scheduling and co-tenant noise. That P3/P5 come out FASTER than P1 is partly
  real (their in-batch gate path is a lookup, while P1/P2 recompute buffer
  statistics every batch) and partly node variance. Treat ratios as coarse.
- One run per cell, no seeds; memory sampled at 30 s intervals (short runs can
  miss the true peak); GECCO runs synced without system metrics, hence sampled
  on OPPORTUNITY / CreditCard / SWaT instead.
- The RW-1 baseline runs (`RW1m-*`) come from the P2 phase, so they too landed
  on different nodes than the proposal runs; the 1.3-2.1x ratios carry the same
  scheduling noise as the cross-proposal ratios.
- The reliable conclusions are the coarse ones: **gating costs ~1.3-2.1x plain
  RW-1 wall-clock** (P3/P5 cheapest, P1/P2 priciest via per-batch statistics),
  **all five stay in the same cost class** (same order of magnitude, same
  memory footprint), **auto-lambda is free** (ratio ~1.0), and no proposal
  requires hardware beyond what RW-1 already needs.

## Reproduce

```bash
# extraction script (wandb API, no re-runs):
python experiments/proposals/extract_runtime_stats.py
```
