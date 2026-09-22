#!/usr/bin/env python
"""Generates reward_scale_intervention_paired.png (the paired per-seed causal-effect
figure) from the runs produced by run_reward_scale_intervention.sh.

This is the script that SHOULD have produced figures_comparison/
reward_scale_intervention_paired.png in the first place -- that figure exists on disk
but wasn't backed by any committed script (fig_reward_scale_paired, in recovery_time.py,
was apparently called ad hoc). This recreates it exactly (same 4 panels: boost_up's
targeted stimulus + its untouched "control-check" stimulus, damp_down's targeted
stimulus + its untouched "control-check" stimulus) and is written so that adding the
CROSSED conditions later (boost_up applied to 100%->0%, damp_down applied to
0%->100% -- not yet run, see chat) is just two more entries in PANELS below, once
those conditions exist under OUT_ROOT/<condition_name>/model_runs_reversal/.

Usage:
  cd results/transfer/reversal
  python3 code/make_reward_scale_paired_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import seed_groups as SG
except ModuleNotFoundError:
    import importlib
    SG = importlib.import_module("16_06_26_seed_groups")
F = SG.RA.F
import recovery_time as RT

SRC = "action_std_0p15_full/model_runs"          # pre-reversal, shared by every condition
OUT_ROOT = "reward_scale_intervention"
OUT = f"{OUT_ROOT}/figures_comparison"

CONTROL = f"{OUT_ROOT}/control/model_runs_reversal"

# (title, control_runs, treat_runs, stim) -- stim 0 = 0%->100% ("newly valuable"),
# stim 2 = 100%->0% ("newly worthless"). Add the crossed-condition panels here once
# those runs exist, e.g.:
#   ("boost_up (2x)\n100%->0% [targeted]", CONTROL,
#    f"{OUT_ROOT}/boost_100to0/model_runs_reversal", 2),
#   ("damp_down (0.3x)\n0%->100% [targeted]", CONTROL,
#    f"{OUT_ROOT}/damp_0to100/model_runs_reversal", 0),
PANELS = [
    ("boost_up (2x)\n0%->100% [targeted]", CONTROL,
     f"{OUT_ROOT}/boost_up/model_runs_reversal", 0),
    ("boost_up (2x)\n100%->0% [control-check]", CONTROL,
     f"{OUT_ROOT}/boost_up/model_runs_reversal", 2),
    ("damp_down (0.3x)\n100%->0% [targeted]", CONTROL,
     f"{OUT_ROOT}/damp_down/model_runs_reversal", 2),
    ("damp_down (0.3x)\n0%->100% [control-check]", CONTROL,
     f"{OUT_ROOT}/damp_down/model_runs_reversal", 0),
]


def main():
    style = str(F.DEFAULT_STYLE)
    if Path(style).exists():
        plt.style.use(style)
        print(f"style: {style}")
    else:
        print(f"** WARNING: style file not found at {style} -- default styling **")
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    RT.fig_reward_scale_paired(SRC, PANELS, out)
    print("Wrote reward_scale_intervention_paired.png to", out)


if __name__ == "__main__":
    main()
