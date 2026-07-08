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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--proposal", type=int, default=1)
    p.add_argument("--dataset", default="gecco", choices=list(DATASETS))
    p.add_argument("--variant", default="basic")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--pad", type=int, default=150, help="context padding around the anomaly block")
    p.add_argument("--wandb", action="store_true", help="also log the figure to wandb")
    args = p.parse_args()

    data, label = load_dataset(args.dataset)
    print(f"data {data.shape} | anomalies {int(label.sum())}", flush=True)

    model = build_model(args.proposal, variant=args.variant, lam=args.lam, tau=args.tau,
                        warmup_epochs=args.warmup, window_size=50, feats=data.shape[1],
                        epochs=args.epochs, batch_size=256, l1_weight=0.001)
    scores = model.fit(data)

    ts = model._normalize(data).T                 # [feats, T] normalized original
    corr = model.correction_full                  # [feats, T]
    corrected = ts + corr                          # changed input
    gate = model.gate_per_t

    f = int(np.argmax(np.abs(corr).sum(axis=1)))   # most-corrected feature
    a, b = largest_anomaly_span(label, args.pad)
    t = np.arange(a, b)
    print(f"feature {f} (most corrected), window [{a}:{b}]", flush=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(11, 6),
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(t, ts[f, a:b], color="#1f77b4", lw=1.1, label="original x")
    ax1.plot(t, corrected[f, a:b], color="#ff7f0e", lw=1.1, label="corrected x + correction")
    shade_anomalies(ax1, label, a, b)
    ax1.set_ylabel(f"feature {f} (normalized)")
    ax1.set_title(f"P{args.proposal}-{args.variant} on {args.dataset}: original vs corrected "
                  f"at anomaly (red span)")
    ax1.legend(loc="upper right", fontsize=9)

    ax2.plot(t, gate[a:b], color="#2ca02c", lw=1.1, label="gate activation")
    ax2.plot(t, scores[a:b], color="#9467bd", lw=1.0, label="|correction| (score)")
    shade_anomalies(ax2, label, a, b)
    ax2.set_ylabel("gate / |corr|")
    ax2.set_xlabel("time")
    ax2.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"P{args.proposal}_{args.variant}_{args.dataset}_correction_example.png")
    fig.savefig(out, dpi=140)
    print(f"saved -> {out}", flush=True)

    if args.wandb:
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(os.path.dirname(FIG_DIR), "..", ".env"))
        except Exception:
            pass
        import wandb
        run = wandb.init(entity=os.environ.get("WANDB_ENTITY") or None,
                         project=os.environ.get("WANDB_PROJECT", "rwml-autocegar"),
                         mode=os.environ.get("WANDB_MODE", "online"),
                         name=f"P{args.proposal}-{args.variant}-{args.dataset}-example",
                         group=f"proposal{args.proposal}", job_type="figure",
                         tags=["figure", f"P{args.proposal}", args.dataset], reinit=True)
        run.log({"correction_example": wandb.Image(out)})
        run.finish()
        print("logged figure to wandb", flush=True)


if __name__ == "__main__":
    main()
