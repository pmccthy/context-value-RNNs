"""Old (window-mean paired t-test) vs. temporal-cluster responder-test
proportions as a function of trials-per-stimulus used -- the comparison that
justified adopting the old test for all shipped figures, as an actual panel
rather than a one-off printout. Data: analysis/responder_stability_sweep.py's
output/decoding/responder_stability.json (needs a real cxval-env rerun; see
that script's docstring for the exact command).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "transfer" / "code"))

from _tags import new_panel, save_panel  # noqa: E402
import figures as F  # noqa: E402

STAB_JSON = _HERE.parent / "output" / "decoding" / "responder_stability.json"
MODEL_TYPES = ["rl_only", "classif_rl", "classif_rl_readout_only"]

OLD_COLOUR = "#3377bb"
CLUSTER_COLOUR = "#cc6677"


def _load():
    return json.load(open(STAB_JSON)) if STAB_JSON.exists() else None


def draw_stability(model_type, results=None, ax=None):
    fig, ax, _ = new_panel(ax, figsize=(6.0, 4.5))
    results = results or _load()
    if results is None or model_type not in results:
        ax.text(0.5, 0.5, "no responder_stability.json yet\n(run responder_stability_sweep.py)",
                ha="center", va="center", transform=ax.transAxes)
        return fig
    r = results[model_type]
    n = r["n_trials_grid"]
    ax.errorbar(n, r["old"]["mean"], yerr=r["old"]["sem"], color=OLD_COLOUR,
               marker="o", lw=2, capsize=3, label="window-mean t-test (shipped)")
    ax.errorbar(n, r["cluster"]["mean"], yerr=r["cluster"]["sem"], color=CLUSTER_COLOUR,
               marker="s", lw=2, capsize=3, label="temporal-cluster test")
    ax.set_xscale("log")
    ax.set_xlabel("trials per stimulus used")
    ax.set_ylabel("mean fraction significant\n(across stimuli, seeds)")
    ax.set_title(f"{F.MODELS[model_type]['label']} (n={r['n_seeds']} seeds)")
    ax.legend(frameon=False)
    return fig


def build_all(show_tag=None):
    results = _load()
    if results is None:
        print("  (skip responder stability: run analysis/responder_stability_sweep.py first)")
        return
    for mt in MODEL_TYPES:
        fig = draw_stability(mt, results=results)
        save_panel(fig, "CrossModel/Decoding", f"RESP.stability.{mt}",
                   f"{mt}_responder_stability", show_tag)


if __name__ == "__main__":
    build_all()
