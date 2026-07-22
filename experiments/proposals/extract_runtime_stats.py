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


def pull_rw1_baseline(api):
    """Plain gate-off RW-1 control runs from the P2 phase (RW1m-*, 100 ep)."""
    rw = {}
    for r in api.runs(ENTITY_PROJECT, filters={"display_name": {"$regex": "^RW1m-"}}, per_page=100):
        rt = r.summary.get("_runtime")
        ds = r.config.get("dataset")
        if rt and ds and r.config.get("epochs") == 100:
            rw.setdefault(ds, []).append(float(rt))
    return {k: mean(v) for k, v in rw.items()}


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


def _per_dataset_means(rows, lam_mode="fixed"):
    d = defaultdict(list)
    for r in rows:
        if r["lam_mode"] == lam_mode and r["epochs"] == 100 and r["dataset"]:
            d[r["dataset"]].append(r["runtime"])
    return {k: mean(v) for k, v in d.items()}


def _report_rw1_baseline(per, shared, api):
    rw = pull_rw1_baseline(api)
    rw_shared = shared & set(rw)
    if not rw_shared:
        return
    print(f"\nvs plain RW-1 baseline (RW1m-*, {len(rw_shared)} shared datasets):")
    print("  RW-1 mean s/run:", round(mean(rw[d] for d in rw_shared), 1))
    print("  ratio vs RW-1:",
          {p: round(mean(per[p][d] / rw[d] for d in rw_shared), 2) for p in per})


def _report_collections(per, shared, data):
    coll = {r["dataset"]: r["collection"] for r in data[1]}
    by_coll = defaultdict(lambda: defaultdict(list))
    for p in per:
        for d in shared:
            by_coll[coll[d]][p].append(per[p][d])
    for c in sorted(by_coll):
        print(f"{c:14s}" + "".join(f" {mean(by_coll[c][p]):6.0f}" for p in sorted(per)))


def _report_auto_ratio(data):
    for p, rows in data.items():
        fx = _per_dataset_means(rows, "fixed")
        au = _per_dataset_means(rows, "auto_tr")
        common = set(fx) & set(au)
        if common:
            print(f"P{p} auto/fixed ratio: "
                  f"{mean(au[d] / fx[d] for d in common):.2f} (n={len(common)})")


def _report_memory(data, api):
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


def report(data, api):
    per = {p: _per_dataset_means(rows) for p, rows in data.items()}
    shared = set.intersection(*[set(v) for v in per.values()])
    print(f"shared fixed/100ep datasets: {len(shared)}")

    _report_rw1_baseline(per, shared, api)

    print("\nmean s/run:", {p: round(mean(per[p][d] for d in shared), 1) for p in per})
    print("median s/run:", {p: round(median(per[p][d] for d in shared), 1) for p in per})
    print("ratio vs P1:", {p: round(mean(per[p][d] / per[1][d] for d in shared), 2) for p in per})

    _report_collections(per, shared, data)
    _report_auto_ratio(data)
    _report_memory(data, api)


if __name__ == "__main__":
    api = wandb.Api()
    report(pull_runtimes(api), api)
