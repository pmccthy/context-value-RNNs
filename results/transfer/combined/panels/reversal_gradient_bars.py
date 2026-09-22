"""Post-reversal learning-rate (gradient) bar panel -- RNN models.

RNN analog of ../../../neuronal-representations/results/transfer/figures/
panels/reversal_gradient.py's draw_reversal_gradient_bars: summarizes the
post-reversal vigour-vs-trials curve (see vigour_value.py's
draw_vigour_vs_trials / figure1_bottom) as ONE number per stimulus direction
instead of the full trial-resolved curve.

Unlike the experimental side (fixed trial window, since real sessions are
short/noisy and never settle onto a clean per-session asymptote), the RNN
side uses a per-SEED RECOVERY-POINT gradient: each recovered seed's own
first-crossing of a mirror-swapped vigour target (see analysis/
reversal_gradients.py's rnn_recovery_gradient / _recovery_index for the
exact definition -- absolute threshold = rel_thr * pre-reversal "100%"
vigour, used for both directions), then the OLS slope of vigour vs. trials
from the reversal point (t=0) through that seed's own recovery trial. This
tighter, seed-specific window is more appropriate here because RNN training
curves are densely probed and far less noisy than real behavioural sessions
(see chat).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "style"))
sys.path.insert(0, str(_HERE.parent / "transfer" / "code"))
sys.path.insert(0, str(_HERE.parent / "analysis"))

from _tags import new_panel, save_panel  # noqa: E402
import figures as F  # noqa: E402
import reversal_gradients as RG  # noqa: E402

MODEL_TYPES = RG.MODEL_TYPES

# Same stimulus order/pretty-names as the experimental panel, so the two
# line up column-for-column once combined into one composite.
GRADIENT_STIMS = ("0_to_100", "100_to_0")
GRADIENT_PRETTY = {"100_to_0": "100%→0%", "0_to_100": "0%→100%"}


def _draw_gradient_bars(ax, items, title, ylabel, rng_seed=0):
    """items: [(label, colour, values_array, count_text)], one bar each.
    Verbatim copy of the experimental panel's helper (see reversal_gradient.
    py) -- deliberately duplicated, not imported, per the cross-repo
    no-cross-import convention."""
    rng = np.random.default_rng(rng_seed)
    for i, (label, colour, vals, count_text) in enumerate(items):
        if len(vals):
            m = float(np.mean(vals))
            se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            ax.bar(i, m, width=0.6, color=colour, yerr=se, capsize=4, zorder=2)
            jitter = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(i + jitter, vals, color="0.25", s=15, alpha=0.6, zorder=3, linewidth=0)
    ax.axhline(0, color="0.5", lw=1, zorder=1)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([f"{lab}\n({ct})" for lab, _, _, ct in items], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


# -- real-mouse gradient, vendored (not imported) from neuronal-representations
# results/transfer/figures/panels/reversal_gradient.py, per the cross-repo
# no-cross-import convention (see that module for the "why a fixed window
# instead of a recovery point" rationale) -- read directly from its own
# per-trial CSVs via NEURONAL_REPO, exactly like extract_experiment_metric_
# values.py in ../cross_model_vs_experiment/ already does for other metrics.
_NEURONAL_REPO = Path(os.environ.get(
    "NEURONAL_REPO", str(Path.home() / "Documents" / "neuronal-representations")))
_EXP_DATA = _NEURONAL_REPO / "results" / "transfer" / "data"
_EXP_REV_CSV = _EXP_DATA / "lick" / "lick_rates_per_trial.csv"
_EXP_META = _EXP_DATA / "single_trial" / "single_trial_metadata.csv"
# sat_plot_colours.STIM_STEM_COLOURS, vendored (same reasoning as above).
_EXP_STIM_COLOURS = {"100_to_0": "#4b2362", "50": "#c24167", "0_to_100": "#edb081"}


def _exp_reversal_points():
    """Verbatim copy of reversal_gradient.py's _reversal_points() -- vendored,
    not imported (see module note above)."""
    if not _EXP_META.exists():
        return {}
    m = pd.read_csv(_EXP_META)
    out = {}
    for sess, g in m.groupby("session"):
        pre = g[g.condition.str.contains("pre-Rev")].session_trial_number
        post = g[g.condition.str.contains("post-Rev")].session_trial_number
        if len(pre) and len(post):
            out[sess] = 0.5 * (pre.max() + post.min())
    return out


def _exp_fixed_window_gradient(window=100, stims=GRADIENT_STIMS, min_trials=5):
    """Verbatim copy of reversal_gradient.py's fixed_window_gradient() --
    vendored, not imported (see module note above)."""
    if not _EXP_REV_CSV.exists():
        return pd.DataFrame(columns=["session", "stimulus", "n_trials", "slope"])
    df = pd.read_csv(_EXP_REV_CSV)
    revs = _exp_reversal_points()
    df = df[df.session.isin(revs)].copy()
    df["rel"] = df.trial_number - df.session.map(revs)
    rows = []
    for sess, sdf in df.groupby("session"):
        for stim in stims:
            s = sdf[(sdf.stimulus_label.astype(str) == stim) & (sdf.rel >= 0) & (sdf.rel < window)]
            if len(s) < min_trials:
                continue
            slope = float(np.polyfit(s.rel.values, s.anticipatory_lick_rate.values, 1)[0])
            rows.append({"session": sess, "stimulus": stim, "n_trials": len(s), "slope": slope})
    return pd.DataFrame(rows)


def draw_experimental_gradient_bars(window=100, ax=None):
    """Real-mouse counterpart of draw_rnn_gradient_bars, redrawn here (not
    imported -- see module note above) so figure_gradient_all_sources() in
    ../compose.py can place it alongside the 3 RNN model panels in one row.
    Output is identical to reversal_gradient.draw_reversal_gradient_bars()."""
    fig, ax, _ = new_panel(ax, figsize=(4.6, 4.6))
    df = _exp_fixed_window_gradient(window=window)
    items = []
    for stim in GRADIENT_STIMS:
        vals = df[df.stimulus == stim].slope.values if len(df) else np.array([])
        items.append((GRADIENT_PRETTY[stim], _EXP_STIM_COLOURS[stim], vals, f"n={len(vals)} sessions"))
    _draw_gradient_bars(ax, items,
                        f"Real mice\nreversal lick-rate gradient (first {window} trials)",
                        "slope (Hz / trial)")
    return fig


def draw_rnn_gradient_bars(model_type, rel_thr=0.10, smooth_window=3, ax=None):
    """Bar plot of the per-seed recovery-point vigour gradient, one bar per
    value-reversing stimulus direction, mean +/- SEM across recovered seeds
    that also cross the vigour-recovery threshold, with each seed's own
    slope jittered on top. Colour matches this model's colour everywhere
    else in the pipeline (figures.MODELS); label text and axis styling
    otherwise match the experimental panel exactly for visual consistency."""
    fig, ax, _ = new_panel(ax, figsize=(4.6, 4.6))
    colour = F.MODELS[model_type]["color"]
    label = F.MODELS[model_type]["label"]
    n_total = len(RG.recovered_seeds(model_type))
    items = []
    for stim in GRADIENT_STIMS:
        df = RG.rnn_recovery_gradient(model_type, stim, rel_thr=rel_thr, smooth_window=smooth_window)
        vals = df.slope.values
        items.append((GRADIENT_PRETTY[stim], colour, vals, f"n={len(vals)}/{n_total} recovered"))
    _draw_gradient_bars(ax, items,
                        f"{label}\nrecovery-point vigour gradient",
                        "slope (vigour / trial)")
    return fig


def draw_rnn_fit_diagnostic_grid(model_type, rel_thr=0.10, smooth_window=3, ncols=6):
    """Supplementary (record of every seed, per the chat): for every seed
    with a valid recovery-gradient fit in EITHER stimulus direction, one
    panel showing that seed's full post-reversal vigour trajectory (light,
    behind), the actual [0, recovery_trial] window of points RG.
    rnn_recovery_gradient()'s OLS fit was computed from (highlighted, one
    colour per direction), the fit line, and the recovery point itself
    (diamond marker) -- the RNN-side analogue of reversal_gradient.
    draw_fit_diagnostic_grid() on the experimental side."""
    fits = {stim: {d["seed"]: d for d in RG.rnn_recovery_gradient_with_points(
        model_type, stim, rel_thr=rel_thr, smooth_window=smooth_window)}
        for stim in GRADIENT_STIMS}
    seeds = sorted(set(fits[GRADIENT_STIMS[0]]) | set(fits[GRADIENT_STIMS[1]]))
    n = len(seeds)
    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.5 * nrows), squeeze=False)
    for i, seed in enumerate(seeds):
        ax = axes[i // ncols][i % ncols]
        for stim in GRADIENT_STIMS:
            d = fits[stim].get(seed)
            if d is None:
                continue
            colour = _EXP_STIM_COLOURS[stim]
            ax.plot(d["t"], d["v"], color=colour, lw=0.8, alpha=0.3)
            ax.scatter(d["t"][: d["idx"] + 1], d["v"][: d["idx"] + 1], s=10,
                       color=colour, alpha=0.8, linewidth=0, zorder=3)
            xx = np.array([d["t"][0], d["t"][d["idx"]]])
            ax.plot(xx, d["intercept"] + d["slope"] * xx, color=colour, lw=1.8, zorder=4)
            ax.scatter([d["t"][d["idx"]]], [d["v"][d["idx"]]], color=colour, s=30,
                       marker="D", edgecolor="k", lw=0.5, zorder=5)
        ax.set_title(f"seed {seed}", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"{F.MODELS[model_type]['label']}: recovery-point vigour-gradient fits "
                f"-- every seed", fontsize=11)
    fig.text(0.5, 0.01, "trials since reversal", ha="center", fontsize=9)
    fig.text(0.01, 0.5, "vigour", va="center", rotation=90, fontsize=9)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.95))
    return fig


def build_all(show_tag=None):
    for mt in MODEL_TYPES:
        save_panel(draw_rnn_gradient_bars(mt), "Vigour/Reversal",
                   f"VV.REV.gradient.{mt}", f"rnn_gradient_bars_{mt}", show_tag)
        save_panel(draw_rnn_fit_diagnostic_grid(mt), "Vigour/Reversal",
                   f"VV.REV.gradient.{mt}.fitdiag", f"rnn_gradient_fit_diagnostic_{mt}", show_tag)


if __name__ == "__main__":
    build_all()
