# Update to Luis — Proposals 1 & 2 (Auto-CEGAR × RW-ML)

*Status email; numbers verified against the run logs (RW-1@100 GECCO ≈ 0.674; gate→label
AUC > 0.5 on 17/72 series). Verdict is final; a §8.4-diagnostics re-run may shift the
exact AUC values slightly (no fixed seed), not the 0/3 · 0/3 conclusion.*

---

Hi Luis,

Update on the Auto-CEGAR × RW-ML work. I implemented Proposals 1 and 2 on top of the reproduced RW-1 (Algorithm 2) and evaluated both at the collection level (mean over all series in each collection; RW-1 and DeepAnT per-collection means from our reproduction as references).

Result: neither beats RW-1. On the fail-fast verdict set (opportunity/gecco/creditcard), P1 is 0/3 and P2 is 0/3; on three extra domain-diverse collections (below), P2 is also 0/3. Following the fail-fast plan, I am moving on to Proposal 3.

Why the extra datasets. The original three screening sets span feature count and series length, but they are all sensor/activity/finance, so they do not span domain or anomaly type, which is exactly the axis P2's mechanism hinges on (is an anomaly a confident error, or a high-uncertainty region?). I added three domain-diverse collections, chosen a priori to test the hypothesis, and kept them separate from the fixed verdict set so they do not move the pass/fail goalposts:

- SMAP (satellite telemetry, point anomalies): a noisy, high-uncertainty regime where P2 should help if the hypothesis holds.
- SMD (industrial server): a neutral case.
- MITDB (ECG, periodic): a control where P2 was predicted to hurt, like gecco's smooth periodic signal.

How I diagnose why (interpretability). For every run I record, per timestep, the gate activation and the correction magnitude, and compare both against the ground-truth labels. That gives gate-to-label AUC (does the gate fire at anomalies?), anomaly-vs-normal ratios for gate and correction (does it concentrate there?), and trigger precision/recall, plus a Baldo-Figure-6.1-style plot overlaying the original vs corrected signal around an anomaly. This lets me show the failure mechanism concretely instead of only reporting that AUC-PR dropped.

Why they fail, and P1 and P2 fail differently (the interesting part).

- P1 over-targets. Its gate localizes anomalies well (gate-to-label AUC about 0.90 on GECCO) and amplifies the RW correction exactly there (about 5.5x more correction on anomaly windows). Since the anomaly score is the correction magnitude, the gate ends up "correcting the anomalies away," so the score drops.
- P2's gate is uninformative. The MC-dropout confidence is uniformly weak (trigger fraction about 0 on every collection, gate-to-label AUC above 0.5 on only 17 of 72 series, and even GECCO's 0.53 is barely above random versus P1's 0.90), so P2's gate does not localize anomalies on any collection. GECCO shows the most correction concentration (about 2.8x on anomaly windows), but with a near-random gate this most likely reflects RW-1's own residual-driven correction growth rather than gate-induced targeting. Net: the uncertainty confidence never helps, which is exactly the "MC-dropout uncertainty may be poorly calibrated" risk the proposal doc flagged.
- Characterization, hypothesis not supported. I pre-registered that uncertainty-aware confidence would help on high-uncertainty domains (SMAP) and hurt on periodic signals (MITDB, like gecco). It is not supported: P2 never wins, and contrary to the prediction MITDB is the closest to RW-1 (delta -0.002) while SMAP did not help (delta -0.050). So the uncertainty axis does not explain when P2 helps; only the degree of harm varies (severe on gecco and SMD, near zero on MITDB).

Method notes. RW-1's epoch-wise flow is preserved: the gate's residual signal comes from the previous-epoch correction, and the correction updates once per epoch at epoch end, so there is no look-ahead. I also found and fixed one transcription bug in P2 (a missing square root in the uncertainty-standardized residual, verified against the source equation). Deltas are indicative rather than exact, since the reference RW-1 is best-HP/200ep while P2 is fixed-HP/100ep; but a spot check on GECCO shows RW-1 at 100 epochs (about 0.674) is not below RW-1 at 200 epochs (0.639), so the shorter training does not inflate the baseline.

Everything (code, results, interpretability, reproduction notes) is in the repo:

- P1 results: https://github.com/devOTTO/rwml-autocegar/blob/main/experiments/proposals/proposal1_results.md
- P2 results (with characterization): https://github.com/devOTTO/rwml-autocegar/blob/main/experiments/proposals/proposal2_results.md
- How it is run / methodology: https://github.com/devOTTO/rwml-autocegar/blob/main/experiments/proposals/README.md
- Implementation: https://github.com/devOTTO/rwml-autocegar/blob/main/autocegar/proposals/proposal2.py
- Full experiment folder: https://github.com/devOTTO/rwml-autocegar/tree/main/experiments/proposals

Happy to walk through it whenever. Next: Proposal 3 (RW-Correction-Consistency).

Thanks,
Yoonmee
