#!/usr/bin/env python
"""Summarize WHERE and in WHAT SHAPE the anomalies sit in each dataset.

For every dataset in ``run_proposal.DATASETS`` it reports length, #anomalies,
anomaly %, number of contiguous anomaly blocks, and block-length stats, so you can
tell point anomalies (many length-1 blocks) from long contiguous blocks (gecco-style).
With ``--plot`` it also saves a per-dataset label-timeline PNG (where the red spans sit).

Run where the data lives (the cluster):
    source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
    cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
    python experiments/proposals/label_stats.py            # all datasets, summary table
    python experiments/proposals/label_stats.py --plot     # + timeline PNGs
    python experiments/proposals/label_stats.py --dataset gecco --plot   # one
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_proposal import load_dataset, DATASETS, DATA_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")


def anomaly_blocks(label):
    """Contiguous label==1 runs -> list of (start, end_inclusive)."""
    y = np.asarray(label).astype(int)
    idx = np.where(y == 1)[0]
    if len(idx) == 0:
        return []
    brk = np.where(np.diff(idx) > 1)[0]
    return [(int(b[0]), int(b[-1])) for b in np.split(idx, brk + 1)]


def shape_of(lens, has_block):
    if not has_block:
        return "none"
    if np.median(lens) <= 2:
        return "point-like"
    if np.median(lens) >= 20:
        return "block-like"
    return "mixed"


def summarize_label(y, name):
    """Structure record for a 0/1 label vector."""
    y = np.asarray(y).astype(int)
    bl = anomaly_blocks(y)
    lens = np.array([e - s + 1 for s, e in bl]) if bl else np.array([0])
    return {
        "dataset": name, "rows": len(y), "anom": int(y.sum()),
        "anom_pct": 100.0 * float(y.mean()), "n_blocks": len(bl),
        "blk_min": int(lens.min()), "blk_med": float(np.median(lens)),
        "blk_max": int(lens.max()), "shape": shape_of(lens, bool(bl)),
        "blocks": bl,
    }


def collection_of(fname):
    """TSB-AD-M filename `NNN_COLLECTION_id_...` -> COLLECTION token."""
    parts = os.path.basename(fname).split("_")
    return parts[1] if len(parts) >= 2 else os.path.basename(fname)


def scan_all_files():
    """Summarize every TSB-AD-M csv on disk (labels only, no GPU)."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    recs, skipped = [], []
    for f in files:
        try:
            # read ONLY the Label column (low_memory=False avoids the usecols+chunk
            # bug; skips the wide feature columns -> much faster on files like the
            # 248-feature OPPORTUNITY series)
            df = pd.read_csv(os.path.join(DATA_DIR, f), usecols=["Label"],
                             low_memory=False, on_bad_lines="skip")
            raw = pd.to_numeric(df["Label"], errors="coerce").dropna()
        except Exception as e:
            skipped.append((f, str(e).splitlines()[0][:80]))
            continue
        nonbin = raw[(raw != 0) & (raw != 1)]
        if len(nonbin):
            print(f"  [warn] {f}: {len(nonbin)} non-binary Label value(s) -> "
                  f"binarized at >0 (e.g. {nonbin.iloc[0]})", flush=True)
        y = (raw.to_numpy() > 0).astype(int)
        r = summarize_label(y, f)
        r["collection"] = collection_of(f)
        recs.append(r)
    if skipped:
        print(f"  [warn] skipped {len(skipped)} unreadable file(s): "
              + ", ".join(s[0] for s in skipped), flush=True)
    return recs


def _anomaly_density(blocks, rows, nbins=400):
    """Per-bin anomaly fraction over [0, rows] -> [nbins] in [0,1]. Built from block
    spans (no full label needed); makes sparse/short anomalies visible on long series."""
    dens = np.zeros(nbins)
    if rows <= 0:
        return dens
    edges = np.linspace(0, rows, nbins + 1)
    width = rows / nbins
    for s, e in blocks:
        e1 = e + 1                                  # end-exclusive
        lo = int(s / rows * nbins)
        hi = min(int((e1 - 1) / rows * nbins), nbins - 1)
        for b in range(lo, hi + 1):
            overlap = max(0.0, min(e1, edges[b + 1]) - max(s, edges[b]))
            dens[b] += overlap / width
    return np.clip(dens, 0.0, 1.0)


def plot_corpus_timelines(recs):
    """One stacked figure: per-collection anomaly-DENSITY strip (where the anomalies
    sit across the series, binned so they stay visible on long series). Drawn from the
    block spans already in ``recs`` (no reload). Representative = the series with the
    most anomaly blocks in that collection (richest distribution)."""
    by_coll = {}
    for r in recs:
        by_coll.setdefault(r["collection"], []).append(r)
    colls = sorted(by_coll, key=lambda c: -len(by_coll[c]))     # big collections first
    reps = {c: max(by_coll[c], key=lambda r: r["n_blocks"]) for c in colls}

    n = len(colls)
    fig, axes = plt.subplots(n, 1, figsize=(11, 0.5 * n + 1))
    if n == 1:
        axes = [axes]
    for ax, c in zip(axes, colls):
        r = reps[c]
        dens = _anomaly_density(r["blocks"], r["rows"])
        ax.imshow(dens[np.newaxis, :], aspect="auto", cmap="Reds", vmin=0, vmax=1,
                  extent=[0, 1, 0, 1])                # normalized time 0..1
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel(c, rotation=0, ha="right", va="center", fontsize=8)
        ax.set_title(f"n={len(by_coll[c])} · {r['anom_pct']:.1f}% anom · {r['n_blocks']} "
                     f"blocks · med len {r['blk_med']:.0f} · {r['shape']}",
                     fontsize=7, loc="left")
    axes[-1].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[-1].set_xlabel("normalized time (0 = start, 1 = end of the representative series)")
    fig.suptitle("TSB-AD-M anomaly distribution — per-collection density (red = anomalous)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "corpus_anomaly_timelines.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def write_corpus_doc(recs):
    """Per-series csv + per-collection markdown summary -> repo."""
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "blocks"} for r in recs])
    csv_out = os.path.join(HERE, "dataset_anomaly_structure.csv")
    df.to_csv(csv_out, index=False)

    # per-collection aggregate
    def agg(g):
        shapes = g["shape"].value_counts()
        return pd.Series({
            "n_series": len(g),
            "rows_total": int(g["rows"].sum()),
            "anom_pct_mean": g["anom_pct"].mean(),
            "anom_pct_min": g["anom_pct"].min(),
            "anom_pct_max": g["anom_pct"].max(),
            "blk_med_med": g["blk_med"].median(),
            "blk_max": int(g["blk_max"].max()),
            "dominant_shape": shapes.index[0],
            "shape_mix": ", ".join(f"{s}:{n}" for s, n in shapes.items()),
        })
    c = df.groupby("collection").apply(agg).sort_values("n_series", ascending=False)

    tested = {DATASETS[k].split("_")[1]: k for k in DATASETS}  # COLLECTION -> registry key
    section = [
        "## Anomaly structure",
        "",
        f"By anomaly **shape** (labels only), all {len(df)} series across {len(c)} "
        "collections. Per-series detail in "
        "`experiments/proposals/dataset_anomaly_structure.csv`.",
        "",
        "`shape`: **point-like** = median anomaly run ≤ 2 (isolated points); "
        "**block-like** = median run ≥ 20 (contiguous segments); **mixed** in between. "
        "Most collections are block-like (only creditcard/TAO are point-like, PSM mixed). "
        "Since P1–P3 show that any CEGAR gating of the correction does not beat RW-1 — and "
        "the point-like creditcard fails too — this block dominance suggests the negative "
        "result generalizes broadly across the corpus, not just the tested three. (No "
        "shape→failure causal claim is made: GECCO's largest drop is confounded with RW-1 "
        "being strongest there.)",
        "",
        "### Per-collection anomaly summary",
        "`★` = the representative series used in the P1/P2/P3 experiments.",
        "",
        "| Collection | ★ | N | total Len | anomaly % (mean / min–max) | median block len | max block | dominant shape |",
        "|---|:-:|:-:|:-:|:-:|:-:|:-:|---|",
    ]
    for coll, r in c.iterrows():
        star = "★" if coll in tested else ""
        section.append(
            f"| {coll} | {star} | {int(r.n_series)} | {int(r.rows_total):,} | "
            f"{r.anom_pct_mean:.2f}% ({r.anom_pct_min:.2f}–{r.anom_pct_max:.2f}) | "
            f"{r.blk_med_med:.0f} | {int(r.blk_max):,} | {r.dominant_shape} "
            f"({r.shape_mix}) |"
        )

    # anomaly-distribution timelines (where the anomalies sit across the series)
    fig_path = plot_corpus_timelines(recs)
    ROOT = os.path.dirname(os.path.dirname(HERE))          # repo root
    fig_rel = os.path.relpath(fig_path, ROOT)
    section += [
        "",
        "### Anomaly distribution (timelines)",
        "Where the anomalies sit across time, per-collection density (binned) for the "
        "most-fragmented representative series (red = anomalous): a few long red bands "
        "(block-like: gecco/SMAP/MITDB) vs scattered ticks (point-like: creditcard/TAO).",
        "",
        f"![anomaly distribution timelines]({fig_rel})",
    ]

    # inject into dataset_sizes.md between the AUTO markers, preserving the
    # hand-maintained size/runtime content (single unified dataset doc)
    doc = os.path.join(ROOT, "dataset_sizes.md")
    START, END = "<!-- ANOMALY-STRUCTURE:AUTO", "<!-- /ANOMALY-STRUCTURE:AUTO -->"
    body = "\n".join(section)
    with open(doc) as fh:
        text = fh.read()
    if START in text and END in text:
        head = text[:text.index("\n", text.index(START)) + 1]   # incl. START marker line
        tail = text[text.index(END):]
        new = head + "\n" + body + "\n\n" + tail
    else:
        new = (text.rstrip() + f"\n\n{START} (regenerated by "
               "experiments/proposals/label_stats.py --all-files) -->\n\n"
               f"{body}\n\n{END}\n")
    with open(doc, "w") as fh:
        fh.write(new)
    return csv_out, doc, c


def summarize(key):
    _, label = load_dataset(key)
    return summarize_label(label, key)


def plot_timeline(key, label, blocks, out):
    y = np.asarray(label).astype(int)
    fig, ax = plt.subplots(figsize=(12, 1.6))
    for s, e in blocks:
        ax.axvspan(s, e + 1, color="red", alpha=0.6, lw=0)
    ax.set_xlim(0, len(y))
    ax.set_yticks([])
    ax.set_xlabel("time")
    ax.set_title(f"{key}: {int(y.sum())} anomalies ({100 * y.mean():.2f}%), {len(blocks)} block(s)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def log_corpus_to_wandb(recs, per_collection):
    """Log the corpus structure to a wandb reference run (per-series + per-collection
    tables, plus shape counts). No training; CPU only."""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"))
    except Exception:
        pass
    import wandb
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "blocks"} for r in recs])
    c = per_collection.reset_index()
    run = wandb.init(entity=os.environ.get("WANDB_ENTITY") or None,
                     project=os.environ.get("WANDB_PROJECT", "rwml-autocegar"),
                     mode=os.environ.get("WANDB_MODE", "online"),
                     name="dataset-anomaly-structure", group="reference",
                     job_type="dataset-structure",
                     tags=["reference", "dataset-structure", "corpus"], reinit=True)
    run.log({"per_series": wandb.Table(dataframe=df),
             "per_collection": wandb.Table(dataframe=c)})
    shape_counts = df["shape"].value_counts().to_dict()
    run.summary.update({"n_series": len(df), "n_collections": len(c),
                        **{f"shape/{k}": int(v) for k, v in shape_counts.items()}})
    run.finish()
    print(f"[wandb] logged corpus structure -> {run.url}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="all", choices=list(DATASETS) + ["all"])
    p.add_argument("--plot", action="store_true", help="also save a label-timeline PNG per dataset")
    p.add_argument("--all-files", action="store_true",
                   help="scan EVERY TSB-AD-M csv on disk and write the corpus doc "
                        "(dataset_anomaly_structure.md/.csv) instead of the registry table")
    p.add_argument("--wandb", action="store_true",
                   help="with --all-files, also log the corpus structure to a wandb "
                        "reference run (per-series + per-collection tables)")
    args = p.parse_args()

    if args.all_files:
        recs = scan_all_files()
        csv_out, md_out, c = write_corpus_doc(recs)
        print(c.to_string())
        print(f"\n[per-series csv -> {csv_out}]\n[per-collection md -> {md_out}]")
        if args.wandb:
            log_corpus_to_wandb(recs, c)
        return

    keys = list(DATASETS) if args.dataset == "all" else [args.dataset]

    header = (f"{'dataset':12s} {'rows':>8} {'anom':>7} {'anom%':>7} {'blocks':>7} "
              f"{'blk_min':>7} {'blk_med':>8} {'blk_max':>8}  shape")
    print(header)
    print("-" * len(header))
    os.makedirs(FIG_DIR, exist_ok=True)
    for k in keys:
        s = summarize(k)
        print(f"{s['dataset']:12s} {s['rows']:>8} {s['anom']:>7} {s['anom_pct']:>6.2f}% "
              f"{s['n_blocks']:>7} {s['blk_min']:>7} {s['blk_med']:>8.1f} {s['blk_max']:>8}  {s['shape']}")
        if args.plot:
            _, label = load_dataset(k)
            out = os.path.join(FIG_DIR, f"labels_{k}.png")
            plot_timeline(k, label, s["blocks"], out)
            print(f"   -> {out}")


if __name__ == "__main__":
    main()
