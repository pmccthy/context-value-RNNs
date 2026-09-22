"""Group 5: raw responder-group SIZE distributions -- the underlying category
counts that feed chi2_bars.py's model-vs-real chi2 FIT STATISTIC, shown here
directly as bar charts instead of collapsed into a single chi2 number per
model. One self-contained draw_*(ax=...) panel per (source, method) pair:

    draw_rnn(method, group_mode, ax=None)   RNN models, POOLED (summed) across
                                             the 3 architectures (rl_only,
                                             classif_rl, classif_rl_readout_only)
                                             -- the real side has no per-model
                                             axis to compare against, so a
                                             single pooled distribution is the
                                             apples-to-apples counterpart of
                                             the real data's single distribution.
    draw_real(method, group_mode, ax=None)  Real experimental data (expert /
                                             non-reversal), via chi2_bars.py's
                                             own _real_groups() pre-extracted
                                             group-count json.

method: "temporal" or "time_averaged" (matches chi2_bars.py / _method.py).
group_mode: "broad" (3-way, winner-take-all preferred stimulus) or "fine"
(7-way, mixed-selectivity powerset).

Reuses chi2_bars.py's own data-loading (_load_D, _real_groups,
_broad_from_fine, FINE_ORDER, BROAD_ORDER) rather than re-implementing it, so
these counts are guaranteed identical to what feeds the chi2 fit in
chi2_bars.py / figure2().
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "style"))
sys.path.insert(0, str(_HERE.parent / "transfer" / "code"))
sys.path.insert(0, str(_HERE.parent / "analysis"))
sys.path.insert(0, str(_HERE.parent / "cross_model_vs_experiment"))

from _tags import new_panel  # noqa: E402
import style as S  # noqa: E402
import model_group_categories as MG  # noqa: E402
import model_responders as MR  # noqa: E402
import _method  # noqa: E402
import chi2_bars as C2  # noqa: E402  (_load_D, _real_groups, _broad_from_fine, FINE_ORDER, BROAD_ORDER)
from scipy.stats import chi2_contingency  # noqa: E402

MODEL_TYPES = ["rl_only", "classif_rl", "classif_rl_readout_only"]


def _rnn_counts(method, group_mode):
    """Group-size counts pooled (summed) across all 3 RNN architectures,
    in C2.FINE_ORDER / C2.BROAD_ORDER order. Also returns total silent
    (non-responsive) unit count across the 3 architectures."""
    D = C2._load_D(method)
    fn = MG.fine_counts_pooled if group_mode == "fine" else MG.broad_counts_pooled
    n = 7 if group_mode == "fine" else 3
    totals = np.zeros(n)
    n_silent = 0
    for mt in MODEL_TYPES:
        counts, silent = fn(D, mt)
        totals += counts
        n_silent += silent
    return totals, int(n_silent)


def _real_counts(method, group_mode):
    """Real (expert/non-reversal) group-size counts, in the same category
    order, via chi2_bars.py's pre-extracted group-count json."""
    real_fine = C2._real_groups("expert", method)
    fine = np.array([real_fine[k] for k in C2.FINE_ORDER], dtype=float)
    if group_mode == "fine":
        return fine
    return np.array(C2._broad_from_fine(list(fine)), dtype=float)


def _colours(group_mode):
    if group_mode == "fine":
        return [S.GROUP_COLOURS[c] for c in C2.FINE_ORDER]
    return [S.STIM_COLOURS[c.rstrip("%")] for c in C2.BROAD_ORDER]


def _draw_bar(ax, counts, group_mode, title):
    order = C2.FINE_ORDER if group_mode == "fine" else C2.BROAD_ORDER
    x = np.arange(len(order))
    ax.bar(x, counts, color=_colours(group_mode))
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=35 if group_mode == "fine" else 0,
                        ha="right" if group_mode == "fine" else "center")
    ax.set_ylabel("count")
    ax.set_title(title, fontsize=10)


def draw_rnn(method, group_mode, ax=None):
    fig, ax, _ = new_panel(ax, figsize=(5.4, 4.6) if group_mode == "fine" else (4.2, 4.6))
    counts, n_silent = _rnn_counts(method, group_mode)
    n = int(counts.sum()) + n_silent
    _draw_bar(ax, counts, group_mode,
              f"RNN models (pooled, {'fine' if group_mode == 'fine' else 'broad'})\n"
              f"{_method.METHOD_LABEL[method]}  ·  n={n} units, {n_silent} silent")
    return fig


def draw_real(method, group_mode, ax=None):
    fig, ax, _ = new_panel(ax, figsize=(5.4, 4.6) if group_mode == "fine" else (4.2, 4.6))
    counts = _real_counts(method, group_mode)
    n = int(counts.sum())
    _draw_bar(ax, counts, group_mode,
              f"Experimental data ({'fine' if group_mode == 'fine' else 'broad'})\n"
              f"{_method.METHOD_LABEL[method]}  ·  n={n} responsive cells")
    return fig


# --------------------------------------------------------------------------- #
# Pairwise chi2-test-of-homogeneity between all 4 group-size distributions
# above (RNN pooled / experimental x temporal / time_averaged) -- a
# symmetric NxN matrix per granularity, as opposed to the one-directional
# "model vs. real, real treated as expected" fit statistic in chi2_bars.py's
# draw_fine_groups_chi2/draw_broad_groups_chi2. Any pair is comparable here
# (including RNN-vs-RNN or real-vs-real across method), not just model-vs-real.
# --------------------------------------------------------------------------- #
ENTITY_SPECS = [
    ("RNN", "temporal"), ("RNN", "time_averaged"),
    ("experimental", "temporal"), ("experimental", "time_averaged"),
]
ENTITY_LABELS = ["RNN\n(temporal-cluster)", "RNN\n(time-averaged)",
                  "Experimental\n(temporal-cluster)", "Experimental\n(time-averaged)"]


def _entity_counts(source, method, group_mode):
    if source == "RNN":
        counts, _ = _rnn_counts(method, group_mode)
    else:
        counts = _real_counts(method, group_mode)
    return counts


def pairwise_chi2_matrix(group_mode):
    """Returns (chi2, pval), each a 4x4 symmetric ndarray (NaN on the
    diagonal -- self-comparison is undefined/trivial) over ENTITY_SPECS
    order. Each off-diagonal entry is a standard 2xK chi2 test of
    homogeneity (scipy.stats.chi2_contingency on the two entities' raw
    group-size counts, K=3 for broad / 7 for fine) -- tests whether the two
    distributions' SHAPE differs, independent of their different total unit/
    cell counts. Higher chi2 (lower p) = more different."""
    vectors = [_entity_counts(source, method, group_mode) for source, method in ENTITY_SPECS]
    n = len(vectors)
    chi2 = np.full((n, n), np.nan)
    pval = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            table = np.vstack([vectors[i], vectors[j]])
            stat, p, _, _ = chi2_contingency(table)
            chi2[i, j] = chi2[j, i] = stat
            pval[i, j] = pval[j, i] = p
    return chi2, pval
