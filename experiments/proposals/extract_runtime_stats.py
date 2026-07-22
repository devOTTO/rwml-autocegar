#!/usr/bin/env python
"""Extract per-proposal runtime/memory stats from wandb (no re-runs).

Source of docs/computational_cost.md. Pulls `_runtime` for every run in the
`proposal{1-5}_corrected` groups, restricts to the datasets shared by all five
(fixed-lambda, 100 epochs) for a fair comparison, and samples peak
`system.gpu.0.memoryAllocatedBytes` / `system.proc.memory.rssMB` from the
system-metric stream of a few representative runs per proposal.

Usage:
    python experiments/proposals/extract_runtime_stats.py
"""
from collections import defaultdict
from statistics import mean, median

import wandb

ENTITY_PROJECT = "yoonmeeh-cmu/rwml-autocegar"
MEM_SAMPLE_DATASETS = ("opportunity_id1", "creditcard_id1", "swat_id1")


def pull_runtimes(api):
    out = {}
    for p in range(1, 6):
        rows = []
        for r in api.runs(ENTITY_PROJECT, filters={"group": f"proposal{p}_corrected"}, per_page=200):
            rt = r.summary.get("_runtime")
            if rt is None:
                continue
            rows.append({
                "dataset": r.config.get("dataset"),
                "collection": r.config.get("collection"),
                "lam_mode": r.config.get("lam_mode") or "fixed",
                "epochs": r.config.get("epochs"),
                "runtime": float(rt),
                "id": r.id,
            })
        out[p] = rows
    return out


def report(data, api):
    per = {}
    for p, rows in data.items():
        d = defaultdict(list)
        for r in rows:
            if r["lam_mode"] == "fixed" and r["epochs"] == 100 and r["dataset"]:
                d[r["dataset"]].append(r["runtime"])
        per[p] = {k: mean(v) for k, v in d.items()}
    shared = set.intersection(*[set(v) for v in per.values()])
    print(f"shared fixed/100ep datasets: {len(shared)}")

    print("\nmean s/run:", {p: round(mean(per[p][d] for d in shared), 1) for p in per})
    print("median s/run:", {p: round(median(per[p][d] for d in shared), 1) for p in per})
    print("ratio vs P1:", {p: round(mean(per[p][d] / per[1][d] for d in shared), 2) for p in per})

    coll = {r["dataset"]: r["collection"] for r in data[1]}
    by_coll = defaultdict(lambda: defaultdict(list))
    for p in per:
        for d in shared:
            by_coll[coll[d]][p].append(per[p][d])
    for c in sorted(by_coll):
        print(f"{c:14s}" + "".join(f" {mean(by_coll[c][p]):6.0f}" for p in sorted(per)))

    for p, rows in data.items():
        fx, au = defaultdict(list), defaultdict(list)
        for r in rows:
            if r["epochs"] != 100 or not r["dataset"]:
                continue
            (fx if r["lam_mode"] == "fixed" else au)[r["dataset"]].append(r["runtime"])
        common = set(fx) & set(au)
        if common:
            print(f"P{p} auto/fixed ratio: "
                  f"{mean(mean(au[d]) / mean(fx[d]) for d in common):.2f} (n={len(common)})")

    print("\npeak memory samples (GPU MB / RSS MB):")
    for p, rows in data.items():
        for want in MEM_SAMPLE_DATASETS:
            pick = next((r for r in rows if r["dataset"] == want and r["lam_mode"] == "fixed"), None)
            if not pick:
                continue
            ev = list(api.run(f"{ENTITY_PROJECT}/{pick['id']}").history(stream="events", pandas=False))
            if not ev:
                continue
            gpu = max((e.get("system.gpu.0.memoryAllocatedBytes") or 0) for e in ev) / 1e6
            rss = max((e.get("system.proc.memory.rssMB") or 0) for e in ev)
            print(f"  P{p} {want}: gpu={gpu:.0f}MB rss={rss:.0f}MB")


if __name__ == "__main__":
    api = wandb.Api()
    report(pull_runtimes(api), api)
