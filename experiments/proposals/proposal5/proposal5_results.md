# Proposal 5 — Temporal-Persistence Confident-Error CEGAR: Results

**Verdict: P5 BEATS the best-HP/200ep RW-1 reproduction on GECCO — fixed 0.648
(Δ+0.009), auto-λ 0.682 (Δ+0.043) — the only proposal to beat RW-1 on any collection
(1/3). (Single no-seed run; the fixed margin is small — see caveat.)**

## Config note
Corrected config: warm-up = **plain RW-1** (gate OFF) then gate ON; `correction_init
='neg_x'`. Delta config-confounded on the epoch/HP axis (P5 is 100ep/default-HP yet
beats a best-HP/200ep RW-1 — which makes the win stronger, not weaker).

## What Proposal 5 is (docx-faithful gate, amplification-only)
Gate only errors that PERSIST over neighbouring time windows, not isolated spikes.
```
m_t = 1(e_t > τ_e),  τ_e = Q_{q_e}(e)         # residual indicator
p_t = mean(m_{t-h .. t+h})                     # temporal persistence (±h)
g   = σ(k(e_t−τ_e)/mad) · σ(k_p(p_t−τ_p))
```
Persistence computed epoch-end (batches are shuffled) and used next epoch for
amplification (ScaleGrad). Variants `h5` (default, ±5) / `h25` (±25). Cheapest of the
five (no extra forward). Shared hooks base.

## Methodology (collection-level)
Verdict set (10 series), `epochs=100`, `warmup=10`, variant `h5`, fixed HP.
RW-1/DeepAnT = reproduction means.

## Collection-level results (fixed)

| collection | n | DeepAnT AUC-PR* | RW-1 AUC-PR* | P5 AUC-PR | **Δ (P5−RW-1)** | P5 AUC-ROC |
|---|:-:|:--:|:--:|:--:|:--:|:--:|
| OPPORTUNITY | 8 | 0.272 | 0.138 | 0.114 | **−0.024** | 0.683 |
| **GECCO** | 1 | 0.454 | 0.639 | **0.648** | **+0.009** | 0.955 |
| CreditCard | 1 | 0.147 | 0.111 | 0.029 | **−0.082** | 0.630 |

**P5 beats RW-1 on 1/3 (GECCO).** auto-λ widens the GECCO win to **0.682 (Δ+0.043)**.

### Correction diagnostics (thesis §8.4, fixed)
| collection | gate→label AUC | corr@anom/norm | Overlap | AnomalyCoverage |
|---|:--:|:--:|:--:|:--:|
| GECCO | 0.945 | 12.55 | 0.219 | 0.877 |
| CreditCard | 0.899 | 1.75 | 0.008 | 0.222 |
| OPPORTUNITY | 0.684 | 1.09 | 0.118 | 0.134 |

Best gate localization of all five on GECCO (0.945) and the highest correction
concentration (12.5×, covering 88% of anomalies).

## Auto-tuning ablation (auto-λ)
| collection | fixed | auto-λ | RW-1 | beats? |
|---|:--:|:--:|:--:|:--:|
| OPPORTUNITY | 0.114 | 0.111 | 0.138 | no |
| **GECCO** | 0.648 | **0.682** | 0.639 | **YES (Δ+0.043)** |
| CreditCard | 0.029 | 0.032 | 0.111 | no |

auto-λ turns the marginal fixed win (+0.009) into a comfortable one (+0.043).

## Interpretability
GECCO anomalies are long contiguous blocks; the persistence gate filters isolated
residual spikes and locks onto those blocks (gate→label AUC 0.95), concentrating
correction there (12.5×). This is exactly where a temporal-persistence trigger should
win, and it does. **CreditCard (isolated point anomalies) is the opposite regime** —
persistence smoothing cannot lock onto single points, so P5 loses there (0.029), matching
the docx risk that smoothing struggles with point anomalies. OPPORTUNITY is near-tie.

## Caveat (before over-claiming the GECCO win)
GECCO is n=1 and there is no fixed seed; the **fixed** margin (+0.009) is within
plausible run-to-run variance. The **auto-λ** margin (+0.043) is more comfortable, and
P5 wins despite being 100ep/default-HP vs a best-HP/200ep baseline. Still, a
robustness check (repeat GECCO P5 several times) is recommended before stating the win
without qualification.

## Cost
Lowest of the five (moving-average of the residual indicator; no extra forward). Indicative.

## Decision
**P5 is the standout: the only proposal to beat tuned RW-1 (GECCO, 1/3), strongest where
anomalies are contiguous blocks.** Recommend a GECCO robustness re-run and extension to
the block-heavy characterization collections (SMAP/SMD/MITDB) to test whether the
block-anomaly advantage generalizes.

## Reproduce
```bash
sbatch experiments/proposals/submit_rerun_all.sh
python experiments/proposals/aggregate_collection.py --proposal 5
```
