#!/usr/bin/env python
"""Clean-replace the correction-example figures in wandb (option C).

The figures were regenerated locally at higher dpi / larger fonts with wandb
disabled, so the online project still holds the OLD low-res example runs. This
script (1) lists/deletes the old figure runs and (2) re-logs the 41 refreshed
PNGs from figures/ WITHOUT retraining (just uploads the images).

    python experiments/proposals/relog_figures.py --list            # dry-run: show figure runs
    python experiments/proposals/relog_figures.py --delete --apply   # delete old figure runs
    python experiments/proposals/relog_figures.py --relog            # log the 41 new PNGs

A run counts as a "figure run" if job_type == 'figure' or its name ends with
'-example'. Re-logged runs go to group proposalN_corrected, job_type 'figure'.
"""
import argparse
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
NAME_RE = re.compile(r"^P(\d+)_([a-z0-9]+)_(.+)_correction_example\.png$")


def entity_project():
    return (os.environ.get("WANDB_ENTITY"),
            os.environ.get("WANDB_PROJECT", "rwml-autocegar"))


def is_figure_run(run):
    return run.job_type == "figure" or (run.name or "").endswith("-example")


def list_or_delete(delete, apply):
    import wandb
    ent, proj = entity_project()
    api = wandb.Api()
    runs = api.runs(f"{ent}/{proj}")
    figs = [r for r in runs if is_figure_run(r)]
    print(f"found {len(figs)} figure runs in {ent}/{proj}:")
    for r in figs:
        print(f"  {r.id}  {r.group or '-':22s}  {r.name}")
    if delete and apply:
        for r in figs:
            r.delete()
        print(f"[deleted {len(figs)} figure runs]")
    elif delete:
        print("\n[dry-run] pass --apply to actually delete these.")
    return figs


def relog():
    import wandb
    ent, proj = entity_project()
    pngs = sorted(glob.glob(os.path.join(FIG, "*_correction_example.png")))
    print(f"re-logging {len(pngs)} figures to {ent}/{proj} ...")
    n = 0
    for p in pngs:
        m = NAME_RE.match(os.path.basename(p))
        if not m:
            print(f"  skip (unparsed): {os.path.basename(p)}"); continue
        num, variant, dataset = m.group(1), m.group(2), m.group(3)
        run = wandb.init(entity=ent or None, project=proj,
                         mode=os.environ.get("WANDB_MODE", "online"),
                         name=f"P{num}-{variant}-{dataset}-example",
                         group=f"proposal{num}_corrected", job_type="figure",
                         tags=["figure", f"P{num}", dataset, "corrected"],
                         config={"proposal": int(num), "variant": variant,
                                 "dataset": dataset, "correction_init": "neg_x"},
                         reinit=True)
        run.log({"correction_example": wandb.Image(p)})
        run.finish()
        n += 1
        print(f"  [{n}/{len(pngs)}] P{num}-{variant}-{dataset}")
    print(f"[re-logged {n} figures]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list figure runs (dry-run)")
    ap.add_argument("--delete", action="store_true", help="delete figure runs")
    ap.add_argument("--relog", action="store_true", help="log the 41 new PNGs")
    ap.add_argument("--apply", action="store_true", help="actually delete (with --delete)")
    args = ap.parse_args()
    if args.list or args.delete:
        list_or_delete(args.delete, args.apply)
    if args.relog:
        relog()
    if not (args.list or args.delete or args.relog):
        ap.print_help()


if __name__ == "__main__":
    main()
