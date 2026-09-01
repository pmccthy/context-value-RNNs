"""Transfer-study panels (3-model comparison: rl_only, classif_rl,
classif_rl_readout_only), converged pre-reversal models.

Tags:
  TRANSFER.popact.<model>   Population activity per stimulus, one bar chart per model
  TRANSFER.respgroups.<model>  Responder-group sizes per model (temporal-cluster t-test)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "style"))
sys.path.insert(0, str(_HERE.parent / "analysis"))
sys.path.insert(0, str(_HERE.parent / "transfer" / "code"))  # figures.py (F.load)

from _tags import new_panel, save_panel  # noqa: E402
import style as S  # noqa: E402
import figures as F  # noqa: E402  (transfer_final/code/figures.py)
import model_group_categories as MG  # noqa: E402

FIGURE_DATA_DIR = _HERE.parent / "transfer" / "figure_data"
TIME_RESOLVED_DIR = FIGURE_DATA_DIR / "time_resolved"
MODEL_TYPES = ["rl_only", "classif_rl", "classif_rl_readout_only"]
STIM_ORDER = ["0", "50", "100"]   # matches figures.py stim index 0,1,2


def _load_D():
    return F.load(str(FIGURE_DATA_DIR))


# --------------------------------------------------------------------------- #
# Population activity per stimulus, one model per figure (bars coloured by
# stimulus, matching neuronal-representations' draw_expert_sizes convention)
# --------------------------------------------------------------------------- #
def draw_population_activity_bar(model_type, D=None, ax=None):
    """One bar per stimulus (coloured by stimulus, matching neuronal-representations'
    draw_expert_sizes/draw_reversal_sizes convention of colouring bars by category
    rather than by model), mean +/- SEM pop_activity across seeds, for ONE model.
    Uses D["scalars"]["pop_activity"][model_index] -> (seed, stim), exactly the array
    figures.py's bar_metric() reads (NaN for seeds not present, handled via nanmean)."""
    fig, ax, _ = new_panel(ax, figsize=(5, 4.5))
    D = D if D is not None else _load_D()
    ti = D["model_types"].index(model_type)
    arr = D["scalars"]["pop_activity"][ti]           # (seed, stim)
    n = int(np.isfinite(arr[:, 0]).sum())
    mean = np.nanmean(arr, axis=0)
    sem = np.nanstd(arr, axis=0) / np.sqrt(max(n, 1))
    colours = [S.STIM_COLOURS[s] for s in STIM_ORDER]
    ax.bar(range(3), mean, yerr=sem, color=colours, capsize=4)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{s}%" for s in STIM_ORDER])
    ax.set_ylabel("Population activity (a.u.)")
    ax.set_xlabel("Stimulus")
    ax.set_title(f"{F.MODELS[model_type]['label']} (n={n} seeds)")
    return fig


def build_population_activity(show_tag=None):
    D = _load_D()
    for mt in MODEL_TYPES:
        try:
            fig = draw_population_activity_bar(mt, D=D)
        except Exception as e:
            print(f"  (skip population activity for {mt}: {e})")
            continue
        save_panel(fig, "Transfer/Population activity", f"TRANSFER.popact.{mt}",
                   f"{mt}_population_activity", show_tag)


# --------------------------------------------------------------------------- #
# Responder-group sizes per model. USES THE EXISTING window-mean paired
# t-test (figure_data.pkl's "responsive" field), NOT the temporal-cluster
# port in model_responders.py -- see chat: the temporal test's group
# composition never stabilises even out to the model's full ~500+
# trials/stimulus (still climbing at n=500 in a trial-count sweep), while
# this window-mean test is flat from ~n=60 trials on AND its group
# composition/chi2-fit to real data is far closer (e.g. classif_rl "all
# three" = 126/3840 here vs. 2469/3840 under the temporal test, against a
# real proportion under 6% of responsive cells). Two taxonomies:
#   FINE  (7 groups, mixed-selectivity powerset)   -- draw_responder_group_bar
#   BROAD (3 groups, winner-take-all preferred stim) -- draw_responder_group_bar_broad
# --------------------------------------------------------------------------- #
def draw_responder_group_bar(model_type, D=None, include_nonresp=True, ax=None):
    fig, ax, _ = new_panel(ax, figsize=(7, 4.8))
    D = D if D is not None else _load_D()
    fine, n_silent = MG.fine_counts_pooled(D, model_type)
    disp_cats = ["0%-only", "50%-only", "100%-only", "0% & 50%", "0% & 100%", "50% & 100%",
                 "all three"] + (["non-responsive"] if include_nonresp else [])
    counts = list(fine.astype(int)) + ([int(n_silent)] if include_nonresp else [])
    colours = [S.GROUP_COLOURS[c] for c in disp_cats]
    n_units = int(fine.sum()) + int(n_silent)
    ax.bar(range(len(disp_cats)), counts, color=colours)
    ax.set_xticks(range(len(disp_cats)))
    ax.set_xticklabels(disp_cats, rotation=35, ha="right")
    ax.set_ylabel("number of units")
    ax.set_title(f"{F.MODELS[model_type]['label']}  ·  window-mean t-test, "
                 f"{n_units} units")
    return fig


def build_responder_groups(show_tag=None):
    D = _load_D()
    for mt in MODEL_TYPES:
        try:
            fig = draw_responder_group_bar(mt, D=D)
        except Exception as e:
            print(f"  (skip responder groups for {mt}: {e})")
            continue
        save_panel(fig, "Transfer/Subgroups", f"TRANSFER.respgroups.{mt}",
                   f"{mt}_responder_groups_fine", show_tag)


# --------------------------------------------------------------------------- #
# BROAD (3-category, winner-take-all preferred-stimulus) responder-group
# sizes, matching neuronal-representations' draw_expert_sizes bar exactly
# (marginal collapse of the same window-mean test, not a different test).
# --------------------------------------------------------------------------- #
def draw_responder_group_bar_broad(model_type, D=None, include_nonresp=True, ax=None):
    fig, ax, _ = new_panel(ax, figsize=(5, 4.5))
    D = D if D is not None else _load_D()
    broad, n_silent = MG.broad_counts_pooled(D, model_type)
    cats = STIM_ORDER + (["non-responsive"] if include_nonresp else [])
    counts = list(broad.astype(int)) + ([int(n_silent)] if include_nonresp else [])
    colours = [S.STIM_COLOURS[s] for s in STIM_ORDER] + (["0.7"] if include_nonresp else [])
    n_units = int(broad.sum()) + int(n_silent)
    ax.bar(range(len(cats)), counts, color=colours)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([f"{s}%" if s in STIM_ORDER else s for s in cats],
                        rotation=30, ha="right")
    ax.set_ylabel("number of units")
    ax.set_title(f"{F.MODELS[model_type]['label']}  ·  preferred stimulus, "
                 f"{n_units} units")
    return fig


def build_responder_groups_broad(show_tag=None):
    D = _load_D()
    for mt in MODEL_TYPES:
        try:
            fig = draw_responder_group_bar_broad(mt, D=D)
        except Exception as e:
            print(f"  (skip broad responder groups for {mt}: {e})")
            continue
        save_panel(fig, "Transfer/Subgroups", f"TRANSFER.respgroups_broad.{mt}",
                   f"{mt}_responder_groups_broad", show_tag)


def build_all(show_tag=None):
    build_population_activity(show_tag)
    build_responder_groups(show_tag)
    build_responder_groups_broad(show_tag)


if __name__ == "__main__":
    build_all()
