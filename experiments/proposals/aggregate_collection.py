#!/usr/bin/env python
"""Aggregate collection-level P{N} results to per-collection means and compare
against the reproduced RW-1 (best-HP) and DeepAnT per-collection means.

Emits the markdown table (DeepAnT / RW-1 / P{N} / Δ / AUC-ROC) used in the
proposalN_results.md, and prints a delta summary. Robust to duplicate header
rows produced by concurrent slurm appends.

    python experiments/proposals/aggregate_collection.py --proposal 2
"""
import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def ref_means():
    rw = pd.read_csv(os.path.join(ROOT, "rw/reproduction/summary_rw1_besthp.csv"))
    rw_g = rw.groupby("family").agg(rw_pr=("best_pr", "mean"), rw_roc=("best_roc", "mean"))
    da = pd.read_csv(os.path.join(ROOT, "deepant/reproduction/summary_per_dataset.csv"))
    da_g = da.groupby("family").agg(da_pr=("AUC-PR", "mean"), da_roc=("AUC-ROC", "mean"))
    return rw_g, da_g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", type=int, default=2)
    args = ap.parse_args()

    csv = os.path.join(HERE, f"results_p{args.proposal}.csv")
    df = pd.read_csv(csv)
    # drop duplicate-header rows from concurrent appends
    df = df[pd.to_numeric(df["auc_pr"], errors="coerce").notna()].copy()
    df["auc_pr"] = df["auc_pr"].astype(float)
    df["auc_roc"] = df["auc_roc"].astype(float)
    # VERDICT = default config only (sweep rows share the same CSV): tau=2, lam=1,
    # base variant (basic/mc5), tau_u=0. Filter so sweeps don't pollute the mean.
    for col, val in [("tau", 2.0), ("lam", 1.0)]:
        df = df[pd.to_numeric(df[col], errors="coerce") == val]
    try:                        # keep only each proposal's default variant
        from autocegar.proposals import get_proposal
        dv = get_proposal(args.proposal).get("default_variant")
        if dv:
            df = df[df["variant"] == dv]
    except Exception:
        df = df[df["variant"].isin(["basic", "mc5", "full", "gradnorm", "h5"])]
    if "tau_u" in df.columns:
        df = df[pd.to_numeric(df["tau_u"], errors="coerce").fillna(0.0) == 0.0]
    if "extra" in df.columns:  # exclude sweep/auto rows (they carry --extra)
        df = df[df["extra"].fillna("") == ""]

    g = df.groupby("collection").agg(n=("auc_pr", "size"),
                                     p_pr=("auc_pr", "mean"),
                                     p_roc=("auc_roc", "mean"))
    rw_g, da_g = ref_means()

    label = f"P{args.proposal}"
    lines = [f"| collection | n | DeepAnT AUC-PR* | RW-1 AUC-PR* | {label} AUC-PR | **Δ ({label}−RW-1)** | {label} AUC-ROC |",
             "|---|:-:|:--:|:--:|:--:|:--:|:--:|"]
    improved = 0
    for coll, r in g.iterrows():
        rw_pr = float(rw_g.loc[coll, "rw_pr"]) if coll in rw_g.index else float("nan")
        da_pr = float(da_g.loc[coll, "da_pr"]) if coll in da_g.index else float("nan")
        delta = r.p_pr - rw_pr
        if delta > 0:
            improved += 1
        lines.append(f"| {coll} | {int(r.n)} | {da_pr:.3f} | {rw_pr:.3f} | "
                     f"{r.p_pr:.3f} | **{delta:+.3f}** | {r.p_roc:.3f} |")

    table = "\n".join(lines)
    print(table)
    print(f"\n{label}: beats RW-1 (Δ>0) on {improved}/{len(g)} collections.")
    print("* DeepAnT / RW-1 = reproduction per-collection means (best-HP/200ep, "
          "TSB-AD eval) — reference; P is 100ep collection mean over the series shown (n).")

    # read-docs live in experiments/proposals/results/ (results_pN.csv stays at HERE,
    # written by run_proposal.py)
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"collection_table_p{args.proposal}.md")
    with open(out, "w") as f:
        f.write(table + f"\n\n{label}: beats RW-1 on {improved}/{len(g)} collections.\n")
    print(f"\n[table saved -> {out}]")


if __name__ == "__main__":
    main()
