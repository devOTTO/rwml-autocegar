#!/usr/bin/env python
"""Factor-level gate plot: the E (wrongness) and C (confidence) factors separately.

Backs the P2 claim "a noisy confidence factor destroys the aim the residual
had": trains one proposal, then plots

  (0) whole-series anomaly map with the zoom window,
  (1) the two gate factors E_t and C_t around the largest anomaly block
      (plus their product, the gate), and
  (2) the distribution of the confidence factor on anomaly vs normal
      timesteps over the WHOLE series, annotated with the factor's own
      localization AUC (0.5 = the factor carries no anomaly information).

Uses the per-factor maps recorded by the base trainer in the final epoch
(`wrongness_per_t` / `confidence_per_t`), so it needs one training run and no
labels during training. Defaults target P2 (mc5), where the confidence factor
is sigma(k_c(tau_u - u_t)), the MC-dropout uncertainty gate.

Run on a GPU node:

    python experiments/proposals/plot_confidence_factor.py --proposal 2 \
        --variant mc5 --dataset gecco
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
    "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
})

from autocegar.proposals import build_model
from run_proposal import load_dataset, DATASETS
from plot_correction_example import block_span, shade_anomalies, full_timeline

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def make_figure(proposal, dataset, variant, epochs, warmup, tau, lam, pad, wandb_log):
    data, label = load_dataset(dataset)
    label = np.asarray(label).astype(int)
    print(f"[{dataset}] data {data.shape} | anomalies {int(label.sum())}", flush=True)

    model = build_model(proposal, variant=variant, lam=lam, tau=tau,
                        warmup_epochs=warmup, window_size=50, feats=data.shape[1],
                        epochs=epochs, batch_size=256, l1_weight=0.001)
    model.fit(data)

    E = np.asarray(getattr(model, "wrongness_per_t"))
    C = np.asarray(getattr(model, "confidence_per_t"))
    g = np.asarray(model.gate_per_t)
    n = min(len(E), len(label))
    E, C, g, y = E[:n], C[:n], g[:n], label[:n].astype(bool)

    a, b = block_span(label, pad, 0)
    t = np.arange(a, b)

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(13, 9),
                                        gridspec_kw={"height_ratios": [0.5, 1.6, 1.2]})
    full_timeline(ax0, label, a, b)
    ax0.set_title(f"P{proposal}-{variant} on {dataset}: gate factors E (wrongness) "
                  f"and C (confidence), final epoch")

    ax1.plot(t, E[a:b], color="#1f77b4", lw=1.1, label="E (wrongness factor)")
    ax1.plot(t, C[a:b], color="#d62728", lw=1.1, label="C (confidence factor)")
    ax1.plot(t, g[a:b], color="#2ca02c", lw=1.0, ls="--", label="gate g = E x C")
    shade_anomalies(ax1, label, a, b)
    ax1.set_ylabel("factor value")
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(loc="upper right", fontsize=12)

    # whole-series distribution of C at anomaly vs normal points + its own AUC
    try:
        c_auc = roc_auc_score(y, C) if y.any() and (~y).any() else float("nan")
    except ValueError:
        c_auc = float("nan")
    bins = np.linspace(0.0, 1.0, 41)
    ax2.hist(C[~y], bins=bins, density=True, alpha=0.55, color="#7f7f7f", label="C at normal points")
    ax2.hist(C[y], bins=bins, density=True, alpha=0.55, color="#d62728", label="C at anomaly points")
    ax2.set_xlabel("confidence factor C")
    ax2.set_ylabel("density")
    ax2.legend(loc="upper right", fontsize=12)
    ax2.set_title(f"C alone vs labels: AUC = {c_auc:.3f}  "
                  f"(0.5 = the confidence factor carries no anomaly information)",
                  fontsize=13)
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"P{proposal}_{variant}_{dataset}_confidence_factor.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[{dataset}] saved -> {out} | C-vs-label AUC = {c_auc:.3f}", flush=True)

    if wandb_log:
        import wandb
        run = wandb.init(entity=os.environ.get("WANDB_ENTITY") or None,
                         project=os.environ.get("WANDB_PROJECT", "rwml-autocegar"),
                         mode=os.environ.get("WANDB_MODE", "online"),
                         name=f"P{proposal}-{variant}-{dataset}-conf-factor",
                         group=f"proposal{proposal}", job_type="figure",
                         tags=["figure", f"P{proposal}", dataset], reinit=True)
        run.log({"confidence_factor": wandb.Image(out), "conf_factor_auc": c_auc})
        run.finish()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--proposal", type=int, default=2)
    p.add_argument("--dataset", default="gecco")
    p.add_argument("--variant", default="mc5")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--pad", type=int, default=150)
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        make_figure(args.proposal, ds, args.variant, args.epochs, args.warmup,
                    args.tau, args.lam, args.pad, args.wandb)


if __name__ == "__main__":
    main()
