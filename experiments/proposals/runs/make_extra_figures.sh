#!/bin/bash
# Extra figures requested at the 2026-07-20 meeting. Run on a GPU node:
#
#   source /ocean/projects/cis260190p/yhwang2/xlstmad_env/bin/activate
#   cd /ocean/projects/cis260190p/yhwang2/rwml-autocegar
#   bash experiments/proposals/runs/make_extra_figures.sh
#
# Produces (under experiments/proposals/figures/):
#   1. Multi-block correction examples with a matched gate-off RW-1 overlay
#      (P1 on GECCO, 3 largest anomaly blocks, one training run + one lam=0 run):
#        P1_basic_gecco_correction_example_vs_rw1.png
#        P1_basic_gecco_correction_example_block2_vs_rw1.png
#        P1_basic_gecco_correction_example_block3_vs_rw1.png
#   2. The same for P5 (the winning gate), for the talk's win story.
#   3. Factor-level plots (E vs C + confidence-factor histogram/AUC) backing the
#      P2 "noisy confidence factor" claim, on GECCO and OPPORTUNITY:
#        P2_mc5_gecco_confidence_factor.png
#        P2_mc5_opportunity_confidence_factor.png
set -e

# 1) P1 GECCO: 3 largest blocks + gate-off RW-1 overlay
python experiments/proposals/plot_correction_example.py \
    --proposal 1 --variant basic --dataset gecco --blocks 3 --rw1-overlay --wandb

# 2) P5 GECCO: same treatment for the winning gate
python experiments/proposals/plot_correction_example.py \
    --proposal 5 --variant h5 --dataset gecco --blocks 3 --rw1-overlay --wandb

# 3) P2 confidence factor (the miscalibration evidence)
python experiments/proposals/plot_confidence_factor.py \
    --proposal 2 --variant mc5 --dataset gecco --wandb
python experiments/proposals/plot_confidence_factor.py \
    --proposal 2 --variant mc5 --dataset opportunity --wandb
