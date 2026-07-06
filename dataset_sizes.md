# Dataset Sizes & Training Time (TSB-AD-M)

n = 200 datasets. Time = RW-1 training time (l1=0.001, 200 epochs, 1x V100).
Two time columns below: **per-dataset mean** and **collection total** (= n x mean,
i.e. the wall-clock to run that whole collection sequentially on one GPU).

## Overall

- **rows (length)**: min 1,096 / median 46,655 / max 650,000
- **features**: min 2 / median 19 / max 248
- **cells (rows x feats)**: min 30,000 / median 460,800 / max 20,395,869 (**680x spread**)

## Time correlation (Time vs ...)

| variable | Pearson r |
|---|:-:|
| rows | 0.955 |
| feats | -0.253 |
| cells | 0.572 |

**Training time is driven by sequence length (rows), not feature count.**

## Per-collection (sorted by mean cells)

| Collection | n | mean rows | feats | mean cells | mean time/ds | **collection total** |
|---|:-:|--:|:-:|--:|--:|--:|
| TAO | 13 | 10,000 | 3 | 30,000 | 25s (0.4min) | 5.4min |
| MSL | 16 | 3,119 | 55 | 171,565 | 15s (0.3min) | 4.0min |
| SMAP | 27 | 7,855 | 25 | 196,397 | 21s (0.3min) | 9.4min |
| LTDB | 5 | 100,000 | 2 | 220,000 | 226s (3.8min) | 18.8min |
| Genesis | 1 | 16,220 | 18 | 291,960 | 55s (0.9min) | 0.9min |
| Daphnet | 1 | 38,774 | 9 | 348,966 | 90s (1.5min) | 1.5min |
| SVDB | 31 | 207,122 | 2 | 414,245 | 456s (7.6min) | **3.9h** |
| MITDB | 13 | 336,153 | 2 | 672,307 | 781s (13.0min) | **2.8h** |
| SMD | 22 | 25,466 | 38 | 967,721 | 102s (1.7min) | 37.4min |
| Exathlon | 27 | 60,878 | 20 | 1,216,507 | 188s (3.1min) | **1.4h** |
| GECCO | 1 | 138,521 | 9 | 1,246,689 | 318s (5.3min) | 5.3min |
| GHL | 25 | 196,807 | 19 | 3,739,333 | 706s (11.8min) | **4.9h** |
| CATSv2 | 6 | 240,000 | 17 | 4,080,000 | 843s (14.0min) | **1.4h** |
| OPPORTUNITY | 8 | 17,426 | 248 | 4,321,834 | 77s (1.3min) | 10.3min |
| PSM | 1 | 217,624 | 25 | 5,440,600 | 602s (10.0min) | 10.0min |
| CreditCard | 1 | 284,807 | 29 | 8,259,403 | 852s (14.2min) | 14.2min |
| SWaT | 2 | 207,457 | 58 | 10,692,802 | 840s (14.0min) | 28.0min |
| **ALL** | **200** | | | | | **≈ 16.9h** |

> **Whole benchmark (all 200 datasets, RW-1, sequential, 1 GPU) ≈ 60,800s ≈ 16.9h.**
> The long ECG collections dominate: SVDB (3.9h) + MITDB (2.8h) + GHL (4.9h) alone
> ≈ 11.6h, i.e. ~69% of the total. Adding the RW (Algorithm 1) variant and any
> hyperparameter sweeps roughly doubles it — hence the ~1.5–2 day full run.

## Highlighted datasets

| dataset | rows | feats | cells | time |
|---|--:|:-:|--:|--:|
| 173_GECCO_id_1_Sensor | 138,521 | 9 | 1,246,689 | 318s (5.3min) |
| 129_OPPORTUNITY_id_1_HumanActivity | 24,693 | 248 | 6,123,864 | 110s (1.8min) |
| 130_OPPORTUNITY_id_2_HumanActivity | 5,162 | 248 | 1,280,176 | 22s (0.4min) |
| 131_OPPORTUNITY_id_3_HumanActivity | 28,066 | 248 | 6,960,368 | 125s (2.1min) |
| 132_OPPORTUNITY_id_4_HumanActivity | 9,323 | 248 | 2,312,104 | 38s (0.6min) |
| 133_OPPORTUNITY_id_5_HumanActivity | 6,980 | 248 | 1,731,040 | 29s (0.5min) |
| 134_OPPORTUNITY_id_6_HumanActivity | 16,515 | 248 | 4,095,720 | 72s (1.2min) |
| 135_OPPORTUNITY_id_7_HumanActivity | 21,706 | 248 | 5,383,088 | 95s (1.6min) |
| 136_OPPORTUNITY_id_8_HumanActivity | 26,969 | 248 | 6,688,312 | 121s (2.0min) |
| 137_CreditCard_id_1_Finance | 284,807 | 29 | 8,259,403 | 852s (14.2min) |

> **OPPORTUNITY**: most features (248) but short series -> fast (~1-2 min). **CreditCard**: only 29 features but longest (284,807 rows) -> slowest (~14 min). **GECCO**: medium (~138k rows) -> ~5 min. Confirms time proportional to length, not width.

Source: file lengths from `data/TSB-AD-M/`, times from `TSB-AD/eval/metrics/multi_rw1_l1sweep/l1_0.001/CNN_RW/`.
