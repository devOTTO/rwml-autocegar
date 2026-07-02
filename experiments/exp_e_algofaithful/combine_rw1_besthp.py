#!/usr/bin/env python3
"""Per-dataset best-HP reproduction of RW-1: for each dataset take the best
AUC-PR across the L1-weight sweep dirs, aggregate, and compare to the paper
(Table 6.2 AUC-PR / 6.3 AUC-ROC). Run after the 4 sweep jobs finish.
"""
import glob, os, pandas as pd, numpy as np

SWEEP = "/ocean/projects/cis260190p/yhwang2/TSB-AD/eval/metrics/multi_rw1_l1sweep"
LAMS = ["1.0", "0.1", "0.01", "0.001"]
PAPER = {  # family: (AUC-PR RW-1, AUC-ROC RW-1)  from paper Tables 6.2/6.3
 'CATSv2':(0.228,0.755),'CreditCard':(0.173,0.953),'Daphnet':(0.286,0.871),
 'Exathlon':(0.847,0.981),'GECCO':(0.621,0.979),'GHL':(0.013,0.564),
 'Genesis':(0.032,0.954),'LTDB':(0.253,0.611),'MITDB':(0.127,0.639),
 'MSL':(0.086,0.557),'OPPORTUNITY':(0.059,0.430),'PSM':(0.238,0.697),
 'SMAP':(0.095,0.578),'SMD':(0.317,0.790),'SVDB':(0.166,0.628),
 'SWaT':(0.227,0.689),'TAO':(1.000,1.000)}

# load each lambda dir into {filename: row}
per_lam = {}
for lam in LAMS:
    d = os.path.join(SWEEP, f"l1_{lam}", "CNN_RW")
    files = glob.glob(f"{d}/*.csv")
    if not files:
        print(f"[warn] no results yet for l1_{lam} ({d})"); continue
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    for c in ["AUC-PR", "AUC-ROC"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    per_lam[lam] = df.set_index("file")
    print(f"l1_{lam}: {len(df)} datasets")

if not per_lam:
    raise SystemExit("no sweep results found yet")

# union of all files
allf = sorted(set().union(*[set(df.index) for df in per_lam.values()]))
rows = []
for f in allf:
    prs = {lam: per_lam[lam].loc[f, "AUC-PR"] for lam in per_lam if f in per_lam[lam].index}
    best_lam = max(prs, key=prs.get)
    rows.append(dict(file=f, best_lam=best_lam, best_pr=prs[best_lam],
                     best_roc=per_lam[best_lam].loc[f, "AUC-ROC"]))
best = pd.DataFrame(rows)
best["family"] = best["file"].str.extract(r'\d+_([A-Za-z0-9]+)_')

print(f"\n=== per-family best-HP (ours) vs paper ===")
print(f"{'family':<12}{'n':>3}{'PR ours':>9}{'PR paper':>9}{'ROC ours':>10}{'ROC paper':>10}  best-λ dist")
for fam, sub in best.groupby("family"):
    key = fam.upper()
    pp = next((PAPER[k] for k in PAPER if k.upper()==key or key.startswith(k.upper()[:4])), None)
    dist = sub.best_lam.value_counts().to_dict()
    ppr = f"{pp[0]:.3f}" if pp else "  -  "
    proc = f"{pp[1]:.3f}" if pp else "  -  "
    print(f"{fam:<12}{len(sub):>3}{sub.best_pr.mean():>9.3f}{ppr:>9}{sub.best_roc.mean():>10.3f}{proc:>10}  {dist}")

print(f"\n=== OVERALL (best-HP per dataset, n={len(best)}) ===")
print(f"AUC-PR  mean={best.best_pr.mean():.4f}  median={best.best_pr.median():.4f}")
print(f"AUC-ROC mean={best.best_roc.mean():.4f}  median={best.best_roc.median():.4f}")
print(f"best-λ distribution overall: {best.best_lam.value_counts().to_dict()}")
print(f"\nPaper Table 6.1 (RW-1): AUC-PR 0.28, AUC-ROC 0.75")
best.sort_values("best_pr").to_csv(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "rw1_besthp_summary.csv"), index=False)
print("wrote rw1_besthp_summary.csv")
