"""Group 6: chi2 goodness-of-fit DIAGNOSTICS -- visualizes what chi2_bars.py's
draw_fine_groups_chi2/draw_broad_groups_chi2 actually compute, one number,
so it doesn't have to be taken on faith. For one (model_type, method,
group_mode) at a time:

  draw_obs_vs_expected  grouped bars: the model's own observed group-size
                        counts vs. the REAL data's counts rescaled to the
                        model's total N (exactly the "expected" side of the
                        chi2 sum: exp = (real / real.sum()) * obs.sum()) --
                        the rescale factor is shown explicitly in the legend,
                        and the two distributions' actual SHAPES sit side by
                        side so you can see where they diverge.
  draw_residuals        signed standardized residual (obs - exp) / sqrt(exp)
                        per category -- the per-term contribution to the
                        chi2 statistic (chi2 = sum of these squared), so you
                        can see which categories drive the overall number and
                        in which direction the model over/under-represents
                        that group relative to the (rescaled) real data.

Both panels also report chi2, degrees of freedom, the goodness-of-fit
p-value (scipy.stats.chi2.sf), and Cohen's w = sqrt(chi2 / N) -- a sample-
size-independent effect size for one-sample chi2 GOF tests (rule of thumb:
~0.1 small, ~0.3 medium, ~0.5 large). This matters because the raw chi2
value alone conflates "how different the two shapes are" with "how many
units/cells there are" -- with N in the thousands (as here), even a tiny,
practically-irrelevant proportional mismatch will produce a huge chi2 and a
vanishing p-value. Cohen's w strips out the N-dependence so it's at least
roughly comparable across panels/models with different total unit counts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_dist

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "style"))
sys.path.insert(0, str(_HERE.parent / "transfer" / "code"))
sys.path.insert(0, str(_HERE.parent / "analysis"))
sys.path.insert(0, str(_HERE.parent / "cross_model_vs_experiment"))

from _tags import new_panel  # noqa: E402
import style as S  # noqa: E402
import figures as F  # noqa: E402
import model_group_categories as MG  # noqa: E402
import model_responders as MR  # noqa: E402
import _method  # noqa: E402
import chi2_bars as C2  # noqa: E402  (_load_D, _real_groups, _broad_from_fine, FINE_ORDER, BROAD_ORDER)


def _obs_and_exp(model_type, method, group_mode):
    """Returns (obs, exp, real, order) -- obs/exp/real all in `order`'s
    category order, exp already rescaled to obs.sum() exactly as
    chi2_bars._chi2_cat computes it (so these numbers are guaranteed
    identical to what the chi2 bar charts are actually testing).

    IMPORTANT (bug fixed here): for group_mode="broad", the model side MUST
    be built the same way chi2_bars.draw_broad_groups_chi2 builds it --
    fine_counts_pooled() collapsed via C2._broad_from_fine(), which lets a
    mixed-selectivity unit (e.g. "0% & 50%") add into BOTH broad bins it
    touches, exactly mirroring how the real-data side is already double-
    counted (see subgroups.py / _broad_from_fine's docstring). This file
    previously called MG.broad_counts_pooled() instead, which is a totally
    different, non-double-counting WINNER-TAKE-ALL assignment (each unit
    argmax'd into exactly one broad bin) -- a real, distinct statistic, but
    NOT the one CHI2GRID/draw_broad_groups_chi2 computes. Using it here
    silently produced different obs counts (and therefore different chi2
    values and even a different best-fit model ranking) than the bar chart
    this figure is supposed to be diagnosing. Fine mode was never affected:
    both this file and chi2_bars call fine_counts_pooled() directly."""
    D = C2._load_D(method)
    real_fine = C2._real_groups("expert", method)
    if group_mode == "fine":
        obs, _ = MG.fine_counts_pooled(D, model_type)
        order = C2.FINE_ORDER
        real = np.array([real_fine[k] for k in order], dtype=float)
    else:
        fine, _ = MG.fine_counts_pooled(D, model_type)
        obs = C2._broad_from_fine(list(fine))
        order = C2.BROAD_ORDER
        real = np.array(C2._broad_from_fine([real_fine[k] for k in C2.FINE_ORDER]), dtype=float)
    obs = np.asarray(obs, dtype=float)
    exp = (real / real.sum()) * obs.sum()
    return obs, exp, real, order


def _stats(obs, exp):
    chi2_val = float(np.sum((obs - exp) ** 2 / np.where(exp == 0, np.nan, exp)))
    dof = len(obs) - 1
    p = float(chi2_dist.sf(chi2_val, dof))
    n = float(obs.sum())
    w = float(np.sqrt(chi2_val / n)) if n > 0 else float("nan")
    return chi2_val, dof, p, w


def _colours(order, group_mode):
    if group_mode == "fine":
        return [S.GROUP_COLOURS[c] for c in order]
    return [S.STIM_COLOURS[c.rstrip("%")] for c in order]


def draw_obs_vs_expected(model_type, method, group_mode, ax=None):
    fig, ax, _ = new_panel(ax, figsize=(6.4, 4.6) if group_mode == "fine" else (4.8, 4.6))
    obs, exp, real, order = _obs_and_exp(model_type, method, group_mode)
    chi2_val, dof, p, w = _stats(obs, exp)
    x = np.arange(len(order))
    width = 0.38
    colours = _colours(order, group_mode)
    scale = obs.sum() / max(real.sum(), 1e-9)
    # Static labels (not the per-model scale factor -- that varies per row/
    # panel, so embedding it in the label text would give _combine_legends
    # several near-duplicate entries once the composite merges legends
    # across panels; the factor is reported in the text annotation instead).
    ax.bar(x - width / 2, obs, width, color=colours, label="Model (observed)")
    ax.bar(x + width / 2, exp, width, color=colours, alpha=0.45, hatch="//",
           edgecolor="0.3", linewidth=0.6, label="Real (rescaled to match model N)")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=35 if group_mode == "fine" else 0,
                        ha="right" if group_mode == "fine" else "center")
    ax.set_ylabel("count")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(0.98, 0.98,
            rf"$\chi^2$={chi2_val:.0f}, dof={dof}" + "\n" + rf"p={p:.2g}, Cohen's w={w:.2f}"
            + "\n" + rf"rescale ×{scale:.2g}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.set_title(f"{F.MODELS[model_type]['label']}", fontsize=11)
    return fig


def draw_residuals(model_type, method, group_mode, ax=None):
    fig, ax, _ = new_panel(ax, figsize=(6.4, 4.2) if group_mode == "fine" else (4.8, 4.2))
    obs, exp, real, order = _obs_and_exp(model_type, method, group_mode)
    resid = (obs - exp) / np.sqrt(np.where(exp == 0, np.nan, exp))
    x = np.arange(len(order))
    colours = ["#c0392b" if r > 0 else "#2166ac" for r in resid]
    ax.bar(x, resid, color=colours)
    ax.axhline(0, color="0.3", lw=1)
    for lim in (2, -2):
        ax.axhline(lim, color="0.6", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=35 if group_mode == "fine" else 0,
                        ha="right" if group_mode == "fine" else "center")
    ax.set_ylabel("standardized residual\n" + r"(obs $-$ exp) / $\sqrt{\mathrm{exp}}$")
    ax.set_title(f"{F.MODELS[model_type]['label']}: per-category error", fontsize=11)
    # Make the residual -> chi2 relationship explicit and checkable on the
    # figure itself: chi2 = sum(residual_i^2) exactly, term by term, since
    # residual_i^2 = (obs_i-exp_i)^2/exp_i is precisely the i-th summand of
    # the chi2 sum computed in _stats(). Recomputing chi2 here from `resid`
    # (rather than re-calling _stats) is itself a live verification that the
    # two panels agree to floating-point precision.
    chi2_from_resid = float(np.nansum(resid ** 2))
    ax.text(0.02, 0.02,
            r"$\sum_i \mathrm{resid}_i^2$" + f" = {chi2_from_resid:.0f}"
            + r"  $\equiv\ \chi^2$" + " (see panel a)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5,
            style="italic", color="0.25")
    return fig
