# Auto-CEGAR proposals — how to run the experiments

Fail-fast harness for the 5 candidate architectures from
`rw_cegar_research_Proposal_1_to_Proposal_5.docx`. Each proposal is tested first
on **3 selected datasets** (opportunity / gecco / creditcard — topically
disconnected, spanning the RW-vs-DeepAnT performance range). If a proposal beats
the RW-1 baseline on all three it graduates to a full run; otherwise we move on to
the next proposal.

**Implemented:** P1 (Residual-Gated, variants `basic`/`selective`) and P2
(Uncertainty-Aware, MC-dropout, variants `mc5`/`mc10`). P3–P5 are reserved slots.
**Both lost to RW-1 at the collection level** (P1 0/3, P2 0/6) → see
`proposal1_results.md`, `proposal2_results.md`. Next: P3.

**Evaluation unit = whole collection.** `--dataset` accepts a raw TSB-AD-M series
filename, so `submit_pN_coll.sh` runs one array task per series and
`aggregate_collection.py --proposal N` averages them to per-collection means and
compares against the reproduction RW-1 / DeepAnT per-collection means (reference).
The single-`--dataset` / `all` mode remains for quick one-series screens.

- Model code:     `autocegar/proposals/proposalN.py` (+ registry in `__init__.py`)
- Runner:         `run_proposal.py` (repo root)
- Grid + submit:  `experiments/proposals/pN_grid.txt`, `submit_pN_grid.sh`
- Results:        `experiments/proposals/results_pN.csv` (raw, gitignored),
                  `proposalN_results.md` (written up by hand), wandb project
                  `rwml-autocegar`.

## 0. Setup (every session)

```bash
source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
```

wandb credentials come from `.env` (entity `yoonmeeh-cmu`, project
`rwml-autocegar`). To load them into the shell for a manual `wandb` command:
`set -a && source .env && set +a`.

## 1. Run one experiment interactively (quick check)

Grab a GPU, then run the runner directly:

```bash
interact -p GPU-shared --gres=gpu:v100-32:1 -t 1:00:00      # get a GPU node
python run_proposal.py --proposal 1 --dataset opportunity --variant basic \
    --baseline --epochs 100 --warmup 10
# P2 (MC-dropout uncertainty), with a proposal-specific hyperparameter:
python run_proposal.py --proposal 2 --dataset gecco --variant mc5 \
    --baseline --extra tau_u=1.0
```

Key flags (`python run_proposal.py --help` for all):

| flag | meaning | default |
|---|---|---|
| `--proposal N` | which proposal (1–5) | 1 |
| `--dataset` | `opportunity` / `gecco` / `creditcard` / `all` | all |
| `--variant` | proposal variant (P1: `basic`/`selective`; P2: `mc5`/`mc10`) | proposal default |
| `--epochs` / `--warmup` | training length / forecaster-only warm-up | 100 / 10 |
| `--lam` / `--tau` / `--k` | gate strength / robust-z threshold / sharpness | 1.0 / 2.0 / 1.0 |
| `--extra key=val` | proposal-specific kwarg, repeatable (e.g. P2 `--extra tau_u=1.0`) | — |
| `--baseline` | also run plain RW-1 and log the delta | off |
| `--tag` | extra wandb tag (repeatable), e.g. `--tag stage1` | — |
| `--no-wandb` | disable wandb logging | wandb on |

Each run streams per-epoch metrics to wandb (`train/loss`, `train/gate_mean`, …)
and writes final `auc_pr` / `auc_roc` (+ `delta_pr` vs RW-1) to the run summary and
to `results_pN.csv`.

## 2. Run the full sweep grid (batch, recommended)

`pN_grid.txt` has one line of runner args per array task; `submit_pN_grid.sh` runs
them as a slurm array (P1: 11 tasks, `%6` concurrency, wandb **offline** on the
compute node).

```bash
sbatch experiments/proposals/submit_p1_grid.sh
squeue -u $USER                       # watch progress
sacct -j <jobid> --format=JobID,JobName,Elapsed,State   # per-task timing
```

Grid design (P1, mirror it for later proposals):
- **stage1** — basic default HP + `--baseline` on all 3 datasets (baseline only
  here, to avoid recomputing it in every sweep run).
- **stage2/3** — τ and λ sweeps (no baseline; compare against stage1's baseline).
- **stage4** — the alternative variant on all 3.

Edit `pN_grid.txt` to change the sweep; update `--array=1-<N>` in the submit
script to match the line count.

## 3. wandb: sync + find your runs

The array logs offline. After it finishes, sync to the online project from the
login node (which has internet):

```bash
set -a && source .env && set +a
wandb sync --include-offline ./wandb/offline-run-<YYYYMMDD>_*     # scope by date!
```

Then log the aggregate summary run (comparison table + delta bar chart):

```bash
python experiments/proposals/log_proposal_summary_to_wandb.py --proposal 1
```

**Finding the runs among the workspace's many older runs** (they use a different
config schema — old runs have `cegar.*` / `method`, these do not):

- Filter `Group = proposalN`  → all runs from this proposal (incl. baselines).
- Filter `Tags contains PN`   → the proposal runs + summary (not baselines).
- Filter `Tags contains interp` → the batch that recorded interpretability.
- Filter `Tags contains stage2-tau` (etc.) → one sweep stage.
- **Group by** the `dataset` config field → collapse all runs per dataset.
- Column layout: hide the empty `cegar.*` columns; per-epoch metrics live under
  the `train/` group; keep `variant`, `tau`, `lam`, `auc_pr`, `auc_roc`,
  `delta_pr` visible. Save it as a workspace view to reuse for P2–P5.

### Which values to look at

Run types: `P{N}-{variant}-{dataset}-...` (proposal), `RW1-baseline-...` (baseline),
`P{N}-summary` (aggregate), `P{N}-...-example` (figure).

**1. Performance — the verdict (run summary):**
- `auc_pr`, `auc_roc` — proposal performance.
- `rw1_auc_pr`, `rw1_auc_roc` — RW-1 baseline (only on stage-1 runs).
- **`delta_pr`, `delta_roc`** — proposal − RW-1. Positive = improvement. **Main verdict.**

**2. Interpretability — why (run summary, only on the `interp` batch):**
- **`gate/auc_roc_vs_label`** — does the gate localize anomalies? (>0.5 = targets them)
- **`gate/anom_over_norm`** — how many × more the gate fires on anomaly vs normal steps.
- **`corr/anom_over_norm`** — how many × more correction lands on anomalies
  (**high = it is erasing the anomalies → the P1 risk**).
- `corr/anom_mean`, `corr/norm_mean` — correction magnitude, anomaly vs normal.
- `gate/trigger_frac` — fraction of the timeline gated; `gate/trigger_count` — how many steps.
- `gate/trigger_precision` — of gated steps, fraction that are anomalies.
- `gate/trigger_recall` — of anomaly steps, fraction gated.

**3. Training curves — per epoch (charts, `train/` group):**
`train/loss`, `train/l1`, `train/gate_mean`, `train/lam`, `train/tau`, `train/q95`,
`train/phase` (warmup ↔ main).

**4. Config (⚙):** `proposal`, `variant`, `dataset`, `lam`, `tau`, `k`, `conf_mode`,
`conf_q`, `warmup_epochs`, `epochs`, `window_size`, `batch_size`, `l1_weight`,
`scale_normalize`, `correction_init`, `model`, `score`.

**5. Aggregate / figure runs:** `P{N}-summary` has the `all_runs` table +
`delta_pr_by_dataset` chart + `n_datasets_improved`; `P{N}-...-example` has the
`correction_example` image (original vs corrected signal).

**Quick read:** `delta_pr` (did it win?) + `gate/auc_roc_vs_label` (did the gate
target anomalies?) + `corr/anom_over_norm` (did it then erase them?) explains both
whether and why. Note: `gate/*` and `corr/*` exist only on the `interp` batch;
baseline runs have no `gate/*` or `delta_*`.

## 4. Write up the result

Raw rows accumulate in `results_pN.csv` (gitignored). Summarise them in
`proposalN_results.md` — see `proposal1_results.md` for the template: what was
tested (with epochs/warmup), the delta-vs-RW-1 table, sweep tables, timing
(`sacct`), interpretation, and the fail-fast decision. Commit the md (numbers are
embedded, so the gitignored csv is not needed).

## 4b. Dataset anomaly structure (reference)

`dataset_anomaly_structure.md` summarises WHERE and in WHAT SHAPE the anomalies
sit across the **full TSB-AD-M corpus** (all 200 series / 17 collections), classing
each as **point-like** / **block-like** / **mixed**. This is context for the
RW-CEGAR 'erase' failure mode (it bites hardest on long, localizable blocks).
Regenerate with:

```bash
python experiments/proposals/label_stats.py --all-files   # -> .md + per-series .csv
python experiments/proposals/label_stats.py --plot        # timeline PNGs (tested sets)
```

## 5. Fail-fast decision rule

Run stage1 first, read the deltas, then decide:
- **All 3 datasets ≥ RW-1** → promote to a full (all-datasets, 200-epoch) run.
- **Otherwise** → move on to the next proposal. (P1 lost on all 3 → we went to P2.)

## 6. Add a new proposal (P2–P5)

1. Write `autocegar/proposals/proposalN.py` with a `CNN_RW_CEGAR_PN` class
   (subclass `CNN_RW_CEGAR`; expose `PROPOSAL` / `NAME`). Override only the hook
   the proposal needs — most only touch `_compute_signals(window_resid, res_stats)`
   → `(E_t, C_t)`. Proposals that change the forward pass (e.g. P2 MC-dropout,
   P4 input-gradients) add a further hook to the base and override that too.
2. Fill `PROPOSALS[N]` in `autocegar/proposals/__init__.py` (class + variants).
3. Create `pN_grid.txt` + `submit_pN_grid.sh` (copy P1's, adjust args + array size).
4. `sbatch`, then sync + summary as in §3, and write `proposalN_results.md`.

The runner, wandb logging, warm-up, gate, ScaleGrad and controllers are all
inherited — a new proposal is normally just the signal hook + a grid file.
