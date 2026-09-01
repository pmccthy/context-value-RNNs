#!/usr/bin/env python3
"""Build every individual figure panel, ported from
~/Documents/neuronal-representations/results/transfer/figures/make_panels.py.

Each panel category lives in panels/<name>.py (e.g. panels/transfer.py,
panels/reversal.py, panels/cross_model.py -- named to match whatever the
figure list settles on) with a build_all() that calls save_panel() for each
draw_*(ax=...) function in that module. This file just imports each category
module and calls build_all() -- add an import + call here as each category is
filled in.

    python make_panels.py             # tags shown
    SHOW_TAG=0 python make_panels.py  # tags hidden
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent / "panels"))

import _tags  # noqa: E402
_tags.SHOW_TAG = os.environ.get("SHOW_TAG", "1") not in ("0", "false", "False")

# --- add one import per panels/<name>.py category as they're written ---
import transfer as transfer_panels
import reversal_fine as reversal_fine_panels
import reversal_broad as reversal_broad_panels
import vigour_value as vigour_value_panels
import chi2_bars as chi2_bars_panels
import decoding as decoding_panels
import decoding_timeresolved as decoding_timeresolved_panels
import population_timeresolved as population_timeresolved_panels
import responder_stability as responder_stability_panels


def main():
    print("Transfer:");   transfer_panels.build_all()
    print("Reversal (fine Sankey):"); reversal_fine_panels.build_all()
    print("Reversal (broad Sankey):"); reversal_broad_panels.build_all()
    print("Vigour/value vs trials:"); vigour_value_panels.build_all()
    print("Chi2 bars:"); chi2_bars_panels.build_all()
    print("Decoding:"); decoding_panels.build_all()
    print("Decoding (time-resolved):"); decoding_timeresolved_panels.build_all()
    print("Population/tuning (time-resolved):"); population_timeresolved_panels.build_all()
    print("Responder stability:"); responder_stability_panels.build_all()
    print("\nPanels root ->", _tags.FIG_ROOT)


if __name__ == "__main__":
    main()
