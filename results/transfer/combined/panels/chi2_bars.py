"""Group 4: chi-squared (or, for vigour, correlation) model-vs-experiment
comparison bar plots -- one figure per metric, one bar per model type.
Lower chi2 = closer fit; higher correlation = closer fit. All pre-reversal
vs. the real "expert" (non-reversal) data, per the original ask.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "style"))
sys.path.insert(0, str(_HERE.parent / "transfer" / "code"))
sys.path.insert(0, str(_HERE.parent / "analysis"))
sys.path.insert(0, str(_HERE.parent / "cross_model_vs_experiment"))

from _tags import new_panel, save_panel  # noqa: E402
import style as S  # noqa: E402
import figures as F  # noqa: E402
import model_group_categories as MG  # noqa: E402
from chi2_metrics import chi2_shape_fit  # noqa: E402

MODEL_TYPES = ["rl_only", "classif_rl", "classif_rl_readout_only"]
GROUP_COUNTS = _HERE.parent / "cross_model_vs_experiment" / "group_counts"
FINE_ORDER = ["0%-only", "50%-only", "100%-only", "0% & 50%", "0% & 100%", "50% & 100%", "all three"]
BROAD_ORDER = ["0%", "50%", "100%"]


def _load_D():
    return F.load(str(_HERE.parent / "transfer" / "figure_data"))


def _chi2_cat(obs, real_dict, order):
    obs = np.asarray(obs, float)
    real = np.asarray([real_dict[k] for k in order], float)
    exp = (real / real.sum()) * obs.sum()
    return float(np.sum((obs - exp) ** 2 / np.where(exp == 0, np.nan, exp)))


def _broad_from_fine(fine_arr):
    g = lambda k: fine_arr[FINE_ORDER.index(k)]
    return [g("0%-only") + g("0% & 50%") + g("0% & 100%") + g("all three"),
            g("50%-only") + g("0% & 50%") + g("50% & 100%") + g("all three"),
            g("100%-only") + g("0% & 100%") + g("50% & 100%") + g("all three")]


def _bar_by_model(ax, values, ylabel, title, lower_is_better=True):
    types = MODEL_TYPES
    x = np.arange(len(types))
    vals = [values[m] for m in types]
    best = min(types, key=lambda m: values[m]) if lower_is_better else max(types, key=lambda m: values[m])
    colours = [F.MODELS[m]["color"] for m in types]
    edge = ["k" if m == best else "white" for m in types]
    lw = [2.4 if m == best else 1 for m in types]
    ax.bar(x, vals, 0.6, color=colours, edgecolor=edge, linewidth=lw)
    ax.set_xticks(x)
    ax.set_xticklabels([F.MODELS[m]["label"] for m in types], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title + f"\nbest: {F.MODELS[best]['label']}")


def draw_pop_activity_chi2(D=None, ax=None):
    """Pearson correlation of the 3-point (0%,50%,100%) pop.-activity profile
    -- NOT chi2. Same reasoning as draw_vigour_correlation: chi2 goodness-of-
    fit is built for count/multinomial data, and population activity is a
    continuous per-stimulus mean, not a count -- treating it as a "pseudo-
    frequency" (the old chi2_shape_fit route) has no real statistical
    grounding. Correlation measures shape match without that assumption and
    without chi2's large-N-inflates-significance problem. Higher = closer
    match (name kept for the tag/tests below; it's a correlation, not chi2)."""
    fig, ax, _ = new_panel(ax, figsize=(5.2, 4.5))
    D = D if D is not None else _load_D()
    real = json.load(open(GROUP_COUNTS / "pop_activity_expert.json"))
    real_v = np.array([real[s] for s in ["0", "50", "100"]])
    vals = {}
    for mt in MODEL_TYPES:
        ti = D["model_types"].index(mt)
        arr = D["scalars"]["pop_activity"][ti]
        obs = np.array([float(np.nanmean(arr[:, si])) for si in range(3)])
        vals[mt] = float(np.corrcoef(obs, real_v)[0, 1])
    _bar_by_model(ax, vals, "Pearson r (pop. activity profile vs. expert data)",
                  "Population activity profile: model vs. expert data", lower_is_better=False)
    return fig


def draw_vigour_correlation(D=None, ax=None):
    """NOT chi2 -- see chat: vigour's chi2-shape-fit is dominated (~85%) by
    the near-zero 0%-stimulus term (model vigour there is ~0, real mice still
    lick at a low baseline rate), so it mostly just measures "how close to
    zero is this model's baseline vigour", not real shape similarity. Uses
    Pearson correlation of the 3-point (0%,50%,100%) profile instead, which
    isn't sensitive to a near-zero denominator. Higher = closer match."""
    fig, ax, _ = new_panel(ax, figsize=(5.2, 4.5))
    D = D if D is not None else _load_D()
    real = json.load(open(GROUP_COUNTS / "vigour_expert.json"))
    real_v = np.array([real[s] for s in ["0", "50", "100"]])
    vals = {}
    for mt in MODEL_TYPES:
        ti = D["model_types"].index(mt)
        arr = D["scalars"]["vigour"][ti]
        obs = np.array([float(np.nanmean(arr[:, si])) for si in range(3)])
        vals[mt] = float(np.corrcoef(obs, real_v)[0, 1])
    _bar_by_model(ax, vals, "Pearson r (vigour profile vs. expert data)",
                  "Vigour profile: model vs. expert data", lower_is_better=False)
    return fig


def draw_fine_groups_chi2(D=None, ax=None):
    fig, ax, _ = new_panel(ax, figsize=(5.2, 4.5))
    D = D if D is not None else _load_D()
    real = json.load(open(GROUP_COUNTS / "expert.json"))
    vals = {}
    for mt in MODEL_TYPES:
        fine, _ = MG.fine_counts_pooled(D, mt)
        vals[mt] = _chi2_cat(fine, real, FINE_ORDER)
    _bar_by_model(ax, vals, r"$\chi^2$ (fine groups)",
                  "Responder groups (fine, 7-way): model vs. expert data")
    return fig


def draw_broad_groups_chi2(D=None, ax=None):
    fig, ax, _ = new_panel(ax, figsize=(5.2, 4.5))
    D = D if D is not None else _load_D()
    real_fine = json.load(open(GROUP_COUNTS / "expert.json"))
    real_broad = dict(zip(BROAD_ORDER, _broad_from_fine([real_fine[k] for k in FINE_ORDER])))
    vals = {}
    for mt in MODEL_TYPES:
        fine, _ = MG.fine_counts_pooled(D, mt)
        broad = _broad_from_fine(list(fine))
        vals[mt] = _chi2_cat(broad, real_broad, BROAD_ORDER)
    _bar_by_model(ax, vals, r"$\chi^2$ (broad groups)",
                  "Responder groups (broad, 3-way): model vs. expert data")
    return fig


def build_all(show_tag=None):
    D = _load_D()
    for tag, name, fn in [
        ("pop_activity", "population_activity_chi2", draw_pop_activity_chi2),
        ("vigour_corr", "vigour_correlation", draw_vigour_correlation),
        ("groups_fine", "responder_groups_fine_chi2", draw_fine_groups_chi2),
        ("groups_broad", "responder_groups_broad_chi2", draw_broad_groups_chi2),
    ]:
        try:
            fig = fn(D=D)
        except Exception as e:
            print(f"  (skip {tag}: {e})")
            continue
        save_panel(fig, "CrossModel/Chi2", f"CHI2.{tag}", name, show_tag)


if __name__ == "__main__":
    build_all()
