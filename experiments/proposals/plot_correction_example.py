#!/usr/bin/env python
"""Baldo-Figure-6.1-style example plot: original vs corrected signal at an anomaly.

Trains one proposal on one dataset, then plots a window around an anomaly region
showing (top) the original signal vs the corrected signal (original + RW
correction) for the most-corrected feature, and (bottom) the per-timestep CEGAR
gate activation and |correction| magnitude. This visualises the P1 failure
mechanism: the gate fires at the anomaly and the correction "flattens" it.

Extras:
  --blocks K       render the K largest anomaly blocks (one PNG each) from a
                   SINGLE training run (default 1 = the largest block only;
                   legacy filename is kept for block rank 0).
  --rw1-overlay    also train a matched plain RW-1 (same proposal class with
                   lam=0, i.e. the gate multiplies by 1) and overlay its
                   corrected signal / |correction| as gray lines, so the
                   CEGAR-on vs CEGAR-off effect is visible in one figure.

Needs a GPU (trains the model). Run inside an interactive GPU session:

    source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
    cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
    python experiments/proposals/plot_correction_example.py --dataset gecco \
        --blocks 3 --rw1-overlay

Saves PNGs under experiments/proposals/figures/ (and logs them to wandb if
--wandb). Follows Baldo's approach: representative example plots, not a
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

# Larger base fonts so the text stays legible when the PNG is downscaled to fit
# a markdown/README column (the raw figure is rendered at high dpi below).
plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
    "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
})

from autocegar.proposals import build_model
from run_proposal import load_dataset, DATASETS

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def anomaly_blocks(label):
    """Contiguous anomaly blocks, largest first."""
    idx = np.where(np.asarray(label) == 1)[0]
    if len(idx) == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    blocks = np.split(idx, breaks + 1)
    return sorted(blocks, key=len, reverse=True)


def block_span(label, pad, rank):
    """(start, end) padded around the rank-th largest anomaly block (0 = largest)."""
    blocks = anomaly_blocks(label)
    if not blocks:
        return 0, min(len(label), 400)
    block = blocks[min(rank, len(blocks) - 1)]
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
    ax.set_ylabel("all\nanomalies", fontsize=11)


def fit_artifacts(proposal, dataset, variant, epochs, warmup, tau, lam, data):
    """Train once, return everything the plots need."""
    model = build_model(proposal, variant=variant, lam=lam, tau=tau,
                        warmup_epochs=warmup, window_size=50, feats=data.shape[1],
                        epochs=epochs, batch_size=256, l1_weight=0.001)
    scores = model.fit(data)
    return {
        "ts": model._normalize(data).T,            # [feats, T] normalized original
        "corr": model.correction_full,             # [feats, T]
        "gate": model.gate_per_t,
        "scores": scores,
    }


def render_block(proposal, dataset, variant, art, rw1, label, pad, rank, wandb_log):
    ts, corr, gate, scores = art["ts"], art["corr"], art["gate"], art["scores"]
    corrected = ts + corr
    f = int(np.argmax(np.abs(corr).sum(axis=1)))   # most-corrected feature
    a, b = block_span(label, pad, rank)
    t = np.arange(a, b)
    blk = "longest block" if rank == 0 else f"block #{rank + 1}"
    print(f"[{dataset}] feature {f}, {blk}, window [{a}:{b}]", flush=True)

    # 3 panels: (0) full-series anomaly structure, (1) orig vs corrected, (2) gate/|corr|
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(13, 8.5),
                                        gridspec_kw={"height_ratios": [0.5, 2, 1]})
    full_timeline(ax0, label, a, b)
    ax0.set_title(f"P{proposal}-{variant} on {dataset}: anomaly structure (top) + "
                  f"correction at the {blk} (blue window)")

    ax1.plot(t, ts[f, a:b], color="#1f77b4", lw=1.1, label="original x")
    ax1.plot(t, corrected[f, a:b], color="#ff7f0e", lw=1.1, label="corrected x + correction")
    if rw1 is not None:
        rw1_corrected = rw1["ts"] + rw1["corr"]
        ax1.plot(t, rw1_corrected[f, a:b], color="#7f7f7f", lw=1.0, ls="--",
                 label="RW-1 corrected (gate off)")
    shade_anomalies(ax1, label, a, b)
    ax1.set_ylabel(f"feature {f} (normalized)")
    ax1.legend(loc="upper right", fontsize=12)

    ax2.plot(t, gate[a:b], color="#2ca02c", lw=1.1, label="gate activation")
    ax2.plot(t, scores[a:b], color="#9467bd", lw=1.0, label="|correction| (score)")
    if rw1 is not None:
        ax2.plot(t, rw1["scores"][a:b], color="#7f7f7f", lw=1.0, ls="--",
                 label="RW-1 |correction| (gate off)")
    shade_anomalies(ax2, label, a, b)
    ax2.set_ylabel("gate / |corr|")
    ax2.set_xlabel("time")
    ax2.legend(loc="upper right", fontsize=12)
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    suffix = "" if rank == 0 else f"_block{rank + 1}"
    tag = "_vs_rw1" if rw1 is not None else ""
    out = os.path.join(FIG_DIR, f"P{proposal}_{variant}_{dataset}_correction_example{suffix}{tag}.png")
    fig.savefig(out, dpi=200)
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
                         name=f"P{proposal}-{variant}-{dataset}-example{suffix}{tag}",
                         group=f"proposal{proposal}", job_type="figure",
                         tags=["figure", f"P{proposal}", dataset], reinit=True)
        run.log({"correction_example": wandb.Image(out)})
        run.finish()
        print(f"[{dataset}] logged figure to wandb", flush=True)
    return out


def make_figures(proposal, dataset, variant, epochs, warmup, tau, lam, pad,
                 blocks, rw1_overlay, wandb_log):
    data, label = load_dataset(dataset)
    print(f"[{dataset}] data {data.shape} | anomalies {int(label.sum())}", flush=True)

    art = fit_artifacts(proposal, dataset, variant, epochs, warmup, tau, lam, data)
    rw1 = None
    if rw1_overlay:
        # Same class, lam=0: the gate multiplies gradients by exactly 1, so this
        # is plain RW-1 under the identical schedule/config (clean on/off pair).
        # Caveat: exact only for amplification-only proposals (P1/P2/P5, and P4
        # with eta_C=0). P3's preserve write-back (1 - gamma*g) is NOT disabled
        # by lam=0, so for P3 this overlay is amplification-off, not fully gate-off.
        print(f"[{dataset}] training matched gate-off RW-1 (lam=0) for the overlay", flush=True)
        rw1 = fit_artifacts(proposal, dataset, variant, epochs, warmup, tau, 0.0, data)

    n_blocks = min(blocks, max(1, len(anomaly_blocks(label))))
    outs = []
    for rank in range(n_blocks):
        outs.append(render_block(proposal, dataset, variant, art, rw1, label,
                                 pad, rank, wandb_log))
    return outs


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
    p.add_argument("--blocks", type=int, default=1,
                   help="render the K largest anomaly blocks from one training run")
    p.add_argument("--rw1-overlay", action="store_true",
                   help="also train matched gate-off RW-1 (lam=0) and overlay it")
    p.add_argument("--wandb", action="store_true", help="also log the figure to wandb")
    args = p.parse_args()

    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        make_figures(args.proposal, ds, args.variant, args.epochs, args.warmup,
                     args.tau, args.lam, args.pad, args.blocks, args.rw1_overlay,
                     args.wandb)


if __name__ == "__main__":
    main()
