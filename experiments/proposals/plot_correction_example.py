#!/usr/bin/env python
"""Baldo-Figure-6.1-style example plot: original vs corrected signal at an anomaly.

Trains one proposal on one dataset, then plots a window around an anomaly region
showing (top) the original signal vs the corrected signal (original + RW
correction) for the most-corrected feature, and (bottom) the per-timestep CEGAR
gate activation and |correction| magnitude. This visualises the P1 failure
mechanism: the gate fires at the anomaly and the correction "flattens" it.

Needs a GPU (trains the model). Run inside an interactive GPU session:

    source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
    cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
    python experiments/proposals/plot_correction_example.py --dataset gecco

Saves a PNG under experiments/proposals/figures/ (and logs it to wandb if
--wandb). Follows Baldo's approach: one representative example plot, not a
full-array dump.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autocegar.proposals import build_model
from run_proposal import load_dataset, DATASETS

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def largest_anomaly_span(label, pad):
    """Return (start, end) of a window padded around the longest anomaly block."""
    idx = np.where(np.asarray(label) == 1)[0]
    if len(idx) == 0:
        return 0, min(len(label), 400)
    # split anomaly indices into contiguous blocks, pick the longest
    breaks = np.where(np.diff(idx) > 1)[0]
    blocks = np.split(idx, breaks + 1)
    block = max(blocks, key=len)
    a = max(0, int(block[0]) - pad)
    b = min(len(label), int(block[-1]) + pad + 1)
    return a, b


def shade_anomalies(ax, label, a, b):
    seg = np.asarray(label[a:b])
    t = np.arange(a, b)
    in_span = False
    for i, v in enumerate(seg):
        if v == 1 and not in_span:
            start = t[i]; in_span = True
        elif v == 0 and in_span:
            ax.axvspan(start, t[i], color="red", alpha=0.12, lw=0)
            in_span = False
    if in_span:
        ax.axvspan(start, t[-1], color="red", alpha=0.12, lw=0)


def full_timeline(ax, label, a, b):
    """Whole-series anomaly map (structure) with the zoom window marked."""
    y = np.asarray(label).astype(int)
    idx = np.where(y == 1)[0]
    if len(idx):
        breaks = np.where(np.diff(idx) > 1)[0]
        for blk in np.split(idx, breaks + 1):
            ax.axvspan(blk[0], blk[-1] + 1, color="red", alpha=0.6, lw=0)
    ax.axvspan(a, b, color="#1f77b4", alpha=0.18, lw=0)   # the zoom window
    ax.set_xlim(0, len(y))
    ax.set_yticks([])
    ax.set_ylabel("all\nanomalies", fontsize=8)


def make_figure(proposal, dataset, variant, epochs, warmup, tau, lam, pad, wandb_log):
    data, label = load_dataset(dataset)
    print(f"[{dataset}] data {data.shape} | anomalies {int(label.sum())}", flush=True)

    model = build_model(proposal, variant=variant, lam=lam, tau=tau,
                        warmup_epochs=warmup, window_size=50, feats=data.shape[1],
                        epochs=epochs, batch_size=256, l1_weight=0.001)
    scores = model.fit(data)

    ts = model._normalize(data).T                 # [feats, T] normalized original
    corr = model.correction_full                  # [feats, T]
    corrected = ts + corr                          # changed input
    gate = model.gate_per_t

    f = int(np.argmax(np.abs(corr).sum(axis=1)))   # most-corrected feature
    a, b = largest_anomaly_span(label, pad)
    t = np.arange(a, b)
    print(f"[{dataset}] feature {f} (most corrected), window [{a}:{b}]", flush=True)

    # 3 panels: (0) full-series anomaly structure, (1) orig vs corrected, (2) gate/|corr|
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(11, 7),
                                        gridspec_kw={"height_ratios": [0.5, 2, 1]})
    full_timeline(ax0, label, a, b)
    ax0.set_title(f"P{proposal}-{variant} on {dataset}: anomaly structure (top) + "
                  f"correction at the longest block (blue window)")

    ax1.plot(t, ts[f, a:b], color="#1f77b4", lw=1.1, label="original x")
    ax1.plot(t, corrected[f, a:b], color="#ff7f0e", lw=1.1, label="corrected x + correction")
    shade_anomalies(ax1, label, a, b)
    ax1.set_ylabel(f"feature {f} (normalized)")
    ax1.legend(loc="upper right", fontsize=9)

    ax2.plot(t, gate[a:b], color="#2ca02c", lw=1.1, label="gate activation")
    ax2.plot(t, scores[a:b], color="#9467bd", lw=1.0, label="|correction| (score)")
    shade_anomalies(ax2, label, a, b)
    ax2.set_ylabel("gate / |corr|")
    ax2.set_xlabel("time")
    ax2.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"P{proposal}_{variant}_{dataset}_correction_example.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[{dataset}] saved -> {out}", flush=True)

    if wandb_log:
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(os.path.dirname(FIG_DIR), "..", ".env"))
        except Exception:
            pass
        import wandb
        run = wandb.init(entity=os.environ.get("WANDB_ENTITY") or None,
                         project=os.environ.get("WANDB_PROJECT", "rwml-autocegar"),
                         mode=os.environ.get("WANDB_MODE", "online"),
                         name=f"P{proposal}-{variant}-{dataset}-example",
                         group=f"proposal{proposal}", job_type="figure",
                         tags=["figure", f"P{proposal}", dataset], reinit=True)
        run.log({"correction_example": wandb.Image(out)})
        run.finish()
        print(f"[{dataset}] logged figure to wandb", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--proposal", type=int, default=1)
    p.add_argument("--dataset", default="gecco",
                   help="a registry key, a raw TSB-AD-M filename, or 'all' to loop "
                        "over every tested representative")
    p.add_argument("--variant", default="basic")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--pad", type=int, default=150, help="context padding around the anomaly block")
    p.add_argument("--wandb", action="store_true", help="also log the figure to wandb")
    args = p.parse_args()

    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        make_figure(args.proposal, ds, args.variant, args.epochs, args.warmup,
                    args.tau, args.lam, args.pad, args.wandb)


if __name__ == "__main__":
    main()
