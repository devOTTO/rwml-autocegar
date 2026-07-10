#!/usr/bin/env python3
"""Move the corrected-config runs to their own wandb group (proposalN_corrected).

The corrected-config runs (warm-up = plain RW-1 then gate, correction_init='neg_x')
share group=proposalN with the OLD-config runs (correction_init='zero'), so grouping
by proposalN mixes them. This re-groups every run with config.correction_init=='neg_x'
to f"proposal{proposal}_corrected" via the wandb API, keeping old and new cleanly apart.

    python experiments/proposals/regroup_corrected.py            # dry-run (prints)
    python experiments/proposals/regroup_corrected.py --apply    # actually update
"""
import argparse
import os

import wandb

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="apply the change (default: dry-run)")
    args = ap.parse_args()
    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT", "rwml-autocegar")
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}" if entity else project)

    n_moved, by_group = 0, {}
    for run in runs:
        if run.config.get("correction_init") != "neg_x":
            continue
        p = run.config.get("proposal")
        if p is None:
            continue
        new_group = f"proposal{int(p)}_corrected"
        by_group[new_group] = by_group.get(new_group, 0) + 1
        if run.group != new_group:
            n_moved += 1
            if args.apply:
                run.group = new_group
                run.update()
    print("corrected-config runs by target group:", by_group)
    print(f"{'moved' if args.apply else 'WOULD move'} {n_moved} runs "
          f"({'applied' if args.apply else 'dry-run — pass --apply'})")


if __name__ == "__main__":
    main()
