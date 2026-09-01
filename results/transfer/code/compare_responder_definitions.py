#!/usr/bin/env python
"""Compare responder DEFINITIONS straight from time_resolved/ (no inference, no repo).

  positive   — a unit counts as responsive to a stimulus only if its response is
               significantly ABOVE baseline (excitatory; this is the default used by
               run_inference and every other figure here).
  two_sided  — a unit counts if it has ANY significant difference from baseline,
               including units that are SUPPRESSED below it.

For each definition it saves the two bar plots the request asked for:
  responder_count_per_stim_<def>   — WITHOUT the exclusivity filter (a unit can be a
                                     responder to several stimuli): # responders per stim.
  responder_groups_<def>           — WITH the exclusivity filter + mixed groups.

So you can see how requiring positivity changes the proportions.

Usage (from the bundle root):
  python code/compare_responder_definitions.py \
         --time-resolved figure_data/time_resolved --out figures
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import matplotlib; import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))      # this folder (code/ or scripts/)
try:
    import figures as F                                        # bundle: code/figures.py
except ModuleNotFoundError:
    import importlib
    F = importlib.import_module("16_06_26_figures")            # repo: scripts/16_06_26_figures.py
import build_figure_data_from_timeresolved as bd

# label -> (direction passed to build_data, human title)
DEFS = {"positive":  ("excitatory", "positive-only (excitatory)"),
        "two_sided": ("two_sided",  "two-sided (any difference)")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-resolved", default="figure_data/time_resolved")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--style", default=str(F.DEFAULT_STYLE))
    args = ap.parse_args()
    matplotlib.use("Agg")
    if Path(args.style).exists():
        plt.style.use(args.style)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    for tag, (direction, title) in DEFS.items():
        D, _ = bd.build_data(args.time_resolved, direction=direction)
        # without the exclusivity filter: responders per stimulus
        fig, _ = F.bar_metric(D, "n_responsive", ylabel="# responsive units",
                              title=f"Responders per stimulus — {title}")
        F.save_fig(fig, out, f"responder_count_per_stim_{tag}")
        # with the exclusivity filter + mixed groups
        fig, _ = F.bar_responder_groups(D, title=f"Responder groups — {title}")
        F.save_fig(fig, out, f"responder_groups_{tag}")


if __name__ == "__main__":
    main()
