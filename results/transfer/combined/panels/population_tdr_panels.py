"""Time-resolved TDR state-space panels -- RNN models.

Draws the TDR analogs of population_pca_panels.py's two figure types,
mirroring the real-data tdr_phase_space.png / tdr_combined_projections.png
figures in neuronal-representations/results/transfer/figures/plot_tdr.py.
The actual regression fit + per-seed aggregation lives in ../analysis/
population_tdr.py -- this module only draws.

  draw_tdr_phase_space(model_type, preds=(0,1), ax=None)
      2D trajectory through TDR-axis space, one line per stimulus, solid =
      pre-reversal, dashed = post-reversal, same start/end marker convention
      as population_pca_panels.draw_pca_phase_space.

  draw_tdr_timecourses(model_type)
      One stacked subplot per predictor (stim / context / value), full
      trial timecourse, pre solid / post dashed, dotted ITI|stim and
      stim|outcome boundary lines -- same layout as draw_pca_timecourses.
"""
from __future__ import annotations

import random

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "style"))
sys.path.insert(0, str(_HERE.parent / "transfer" / "code"))
sys.path.insert(0, str(_HERE.parent / "analysis"))

from _tags import new_panel, save_panel  # noqa: E402
import style as S  # noqa: E402
import figures as F  # noqa: E402
import population_tdr as PTDR  # noqa: E402

MODEL_TYPES = PTDR.MODEL_TYPES
STIM_ORDER = ["0", "50", "100"]
STIM_PRETTY = {"0": "0%", "50": "50%", "100": "100%"}
PRED_NAMES = PTDR.PRED_NAMES
PRED_PRETTY = {"stim": "stim axis", "context": "context axis", "value": "value axis"}


def _traj_marker_handles(scale=1.0):
    """Proxy Line2D legend handles explaining the trajectory markers shared
    by every 3D phase-space plot in this module: an unfilled dot at
    stimulus onset (t=0s), an edged triangle at stimulus offset (t=2s --
    stimuli are shown for exactly 2s), and an edged square at the end of
    the plotted window (on request, so marker shape as well as line colour
    is explained in the legend)."""
    return [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="0.3",
                   markeredgecolor="none", linestyle="None", markersize=6 * scale,
                   label="stimulus onset (t=0s)"),
        plt.Line2D([0], [0], marker="^", color="none", markerfacecolor="0.3",
                   markeredgecolor="k", linestyle="None", markersize=7 * scale,
                   label="stimulus offset (t=2s)"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="0.3",
                   markeredgecolor="k", linestyle="None", markersize=7 * scale,
                   label="window end"),
    ]


def draw_tdr_phase_space(model_type, preds=(0, 1), ax=None):
    fig, ax, _ = new_panel(ax, figsize=(5.2, 4.8))
    D = PTDR.population_tdr_trajectories(model_type)
    t = D["t"]
    m = t >= 0
    xi, yi = preds
    for si, stim in enumerate(STIM_ORDER):
        colour = S.STIM_COLOURS[stim]
        for mean, ls, lw in ((D["pre_mean"], "-", 1.8), (D["post_mean"], "--", 1.8)):
            x, y = mean[m, si, xi], mean[m, si, yi]
            ax.plot(x, y, color=colour, ls=ls, lw=lw)
            ax.scatter(x[-1], y[-1], color=colour, s=32, marker="s",
                       edgecolor="k", lw=0.5, zorder=3)
        ax.scatter(D["pre_mean"][m, si, xi][0], D["pre_mean"][m, si, yi][0],
                   color=colour, s=24, zorder=3)
    ax.set_xlabel(PRED_PRETTY[PRED_NAMES[xi]])
    ax.set_ylabel(PRED_PRETTY[PRED_NAMES[yi]])
    handles = [plt.Line2D([0], [0], color="0.2", ls="-", lw=1.8, label="pre-reversal"),
              plt.Line2D([0], [0], color="0.2", ls="--", lw=1.8, label="post-reversal")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="best")
    ax.set_title(f"{F.MODELS[model_type]['label']} TDR phase space\n"
                 f"(n={D['n_pre']} pre, {D['n_post']}/{D['n_post_total']} recovered post)",
                 fontsize=10)
    return fig


def draw_tdr_timecourses(model_type):
    D = PTDR.population_tdr_trajectories(model_type)
    t = D["t"]
    n_iti, stim_ts = D["period"]["n_iti_pre"], D["period"]["stim_ts"]
    n_pred = D["pre_mean"].shape[-1]
    fig, axes = plt.subplots(n_pred, 1, figsize=(6.2, 2.6 * n_pred), sharex=True)
    axes = np.atleast_1d(axes)
    pk = D["peak_times"]
    for pi, ax in enumerate(axes):
        ax.axvline(-0.5, color="0.6", lw=0.8, ls=":")
        ax.axvline(stim_ts - 0.5, color="0.6", lw=0.8, ls=":")
        ax.axvline(pk[pi], color="0.4", lw=1.0, ls="-.", alpha=0.6)
        for si, stim in enumerate(STIM_ORDER):
            colour = S.STIM_COLOURS[stim]
            pm, ps = D["pre_mean"][:, si, pi], D["pre_sem"][:, si, pi]
            qm, qs = D["post_mean"][:, si, pi], D["post_sem"][:, si, pi]
            ax.fill_between(t, pm - ps, pm + ps, color=colour, alpha=0.15, lw=0)
            ax.plot(t, pm, color=colour, ls="-", lw=1.8,
                    label=f"{STIM_PRETTY[stim]}, pre" if pi == 0 else None)
            if np.isfinite(qm).all():
                ax.fill_between(t, qm - qs, qm + qs, color=colour, alpha=0.15, lw=0)
                ax.plot(t, qm, color=colour, ls="--", lw=1.8,
                        label=f"{STIM_PRETTY[stim]}, post" if pi == 0 else None)
        ax.set_ylabel(f"{PRED_PRETTY[PRED_NAMES[pi]]}\n(peak @ t={pk[pi]:.1f})")
        ax.set_xlim(t[0], t[-1])
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time from stim onset (bins)")
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left",
                   bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"{F.MODELS[model_type]['label']} TDR component time courses "
                 f"(n={D['n_pre']} pre, {D['n_post']}/{D['n_post_total']} recovered post)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def draw_tdr_phase_space_grid_stacked(model_types=None, preds=(0, 1), ncols=6, n_seeds=10, rng_seed=0):
    """TDR analogue of population_pca_panels.draw_pca_phase_space_grid_stacked
    -- one combined figure, all model types' individual-seed grids stacked
    vertically as row-bands (model label at left of each, vertically
    centred), instead of 3 separate files."""
    model_types = model_types if model_types is not None else MODEL_TYPES
    xi, yi = preds
    sections = []
    for mt in model_types:
        D = PTDR.population_tdr_per_seed_trajectories(mt)
        all_seeds = D["seeds"]
        seeds = all_seeds
        if n_seeds is not None and n_seeds < len(seeds):
            seeds = random.Random(rng_seed).sample(seeds, n_seeds)
            seeds = sorted(seeds, key=lambda d: d["seed"])
        sections.append((mt, D["t"], seeds, len(all_seeds)))
    ncols_eff = min(ncols, max((len(s[2]) for s in sections), default=1))
    rows_per_section = [int(np.ceil(len(s[2]) / ncols_eff)) if s[2] else 1 for s in sections]
    nrows = sum(rows_per_section)
    fig, axes = plt.subplots(nrows, ncols_eff, figsize=(2.6 * ncols_eff, 2.4 * nrows), squeeze=False)
    row_starts = []
    row0 = 0
    for (mt, t, seeds, n_all), n_rows_sec in zip(sections, rows_per_section):
        row_starts.append(row0)
        m = t >= 0
        n = len(seeds)
        for i, d in enumerate(seeds):
            ax = axes[row0 + i // ncols_eff][i % ncols_eff]
            for si, stim in enumerate(STIM_ORDER):
                colour = S.STIM_COLOURS[stim]
                x, y = d["scores_pre"][m, si, xi], d["scores_pre"][m, si, yi]
                ax.plot(x, y, color=colour, ls="-", lw=1.0)
                ax.scatter(x[-1], y[-1], color=colour, s=18, marker="s", edgecolor="k", lw=0.3, zorder=3)
                if d["has_post"]:
                    x2, y2 = d["scores_post"][m, si, xi], d["scores_post"][m, si, yi]
                    ax.plot(x2, y2, color=colour, ls="--", lw=1.0)
                    ax.scatter(x2[-1], y2[-1], color=colour, s=18, marker="s", edgecolor="k", lw=0.3, zorder=3)
            ax.set_title(f"seed {d['seed']}" + ("" if d["has_post"] else " (no post)"), fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(PRED_PRETTY[PRED_NAMES[xi]], fontsize=11)
            ax.set_ylabel(PRED_PRETTY[PRED_NAMES[yi]], fontsize=11)
            ax.spines[["top", "right"]].set_visible(False)
        for j in range(n, n_rows_sec * ncols_eff):
            axes[row0 + j // ncols_eff][j % ncols_eff].axis("off")
        row0 += n_rows_sec
    # No suptitle and no per-row seed-count bracket, on request -- the seed
    # counts/selection method are documented in the special-subset captions
    # file instead of cluttering the image itself.
    fig.tight_layout(rect=(0.03, 0, 1, 1))
    for (mt, t, seeds, n_all), n_rows_sec, row0 in zip(sections, rows_per_section, row_starts):
        top_bbox = axes[row0][0].get_position()
        bot_bbox = axes[row0 + n_rows_sec - 1][0].get_position()
        yc = (top_bbox.y1 + bot_bbox.y0) / 2
        fig.text(0.005, yc, F.MODELS[mt]['label'], ha="left", va="center",
                 fontsize=13, fontweight="normal", rotation=90)
    return fig


def draw_tdr_phase_space_grid_stacked_3d(model_types=None, preds=(0, 1, 2), ncols=5,
                                          n_seeds=10, rng_seed=0, elev=22, azim=-60):
    """3D analogue of draw_tdr_phase_space_grid_stacked() -- same stacked
    per-model row-band layout and shared rng_seed convention (picks the
    same 10 seeds per model as the 2D version), but each panel plots all
    3 TDR predictor axes (stim/context/value) jointly instead of a flat
    2-axis projection. See draw_tdr_phase_space_3d() for the single-panel
    version this mirrors, including the z-label placement workaround."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
    model_types = model_types if model_types is not None else MODEL_TYPES
    xi, yi, zi = preds
    sections = []
    for mt in model_types:
        D = PTDR.population_tdr_per_seed_trajectories(mt)
        all_seeds = D["seeds"]
        seeds = all_seeds
        if n_seeds is not None and n_seeds < len(seeds):
            seeds = random.Random(rng_seed).sample(seeds, n_seeds)
            seeds = sorted(seeds, key=lambda d: d["seed"])
        sections.append((mt, D["t"], seeds, len(all_seeds)))
    ncols_eff = min(ncols, max((len(s[2]) for s in sections), default=1))
    rows_per_section = [int(np.ceil(len(s[2]) / ncols_eff)) if s[2] else 1 for s in sections]
    nrows = sum(rows_per_section)
    fig, axes = plt.subplots(nrows, ncols_eff, figsize=(3.1 * ncols_eff, 2.9 * nrows),
                             squeeze=False, subplot_kw={"projection": "3d"})
    row_starts = []
    row0 = 0
    for (mt, t, seeds, n_all), n_rows_sec in zip(sections, rows_per_section):
        row_starts.append(row0)
        m = t >= 0
        off_idx = int(np.argmin(np.abs(t[m] - 2.0)))
        n = len(seeds)
        for i, d in enumerate(seeds):
            ax = axes[row0 + i // ncols_eff][i % ncols_eff]
            for si, stim in enumerate(STIM_ORDER):
                colour = S.STIM_COLOURS[stim]
                x, y, z = d["scores_pre"][m, si, xi], d["scores_pre"][m, si, yi], d["scores_pre"][m, si, zi]
                ax.plot(x, y, z, color=colour, ls="-", lw=1.0)
                ax.scatter(x[0], y[0], z[0], color=colour, s=10, zorder=3)
                ax.scatter(x[off_idx], y[off_idx], z[off_idx], color=colour, s=15, marker="^",
                          edgecolor="k", lw=0.3, zorder=3)
                ax.scatter(x[-1], y[-1], z[-1], color=colour, s=16, marker="s", edgecolor="k", lw=0.3, zorder=3)
                if d["has_post"]:
                    x2 = d["scores_post"][m, si, xi]
                    y2 = d["scores_post"][m, si, yi]
                    z2 = d["scores_post"][m, si, zi]
                    ax.plot(x2, y2, z2, color=colour, ls="--", lw=1.0)
                    ax.scatter(x2[off_idx], y2[off_idx], z2[off_idx], color=colour, s=15, marker="^",
                              edgecolor="k", lw=0.3, zorder=3)
                    ax.scatter(x2[-1], y2[-1], z2[-1], color=colour, s=16, marker="s", edgecolor="k", lw=0.3, zorder=3)
            ax.set_title(f"seed {d['seed']}" + ("" if d["has_post"] else " (no post)"), fontsize=7, pad=0)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_xlabel(PRED_PRETTY[PRED_NAMES[xi]], fontsize=10, labelpad=-6)
            ax.set_ylabel(PRED_PRETTY[PRED_NAMES[yi]], fontsize=10, labelpad=-6)
            ax.text2D(1.0, 0.5, PRED_PRETTY[PRED_NAMES[zi]], transform=ax.transAxes, rotation=90,
                      ha="center", va="center", fontsize=10)
            ax.view_init(elev=elev, azim=azim)
        for j in range(n, n_rows_sec * ncols_eff):
            axes[row0 + j // ncols_eff][j % ncols_eff].axis("off")
        row0 += n_rows_sec
    stim_handles = [plt.Line2D([0], [0], color=S.STIM_COLOURS[s], lw=2.2, label=f"{STIM_PRETTY[s]} stim")
                   for s in STIM_ORDER]
    style_handles = [plt.Line2D([0], [0], color="0.2", ls="-", lw=1.8, label="pre-reversal"),
                     plt.Line2D([0], [0], color="0.2", ls="--", lw=1.8, label="post-reversal")]
    handles = stim_handles + style_handles + _traj_marker_handles()
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="upper right", ncol=4)
    # No suptitle and no per-row seed-count bracket, on request -- the seed
    # counts/selection method are documented in the special-subset captions
    # file instead of cluttering the image itself.
    fig.tight_layout(rect=(0.03, 0, 1, 0.96))
    for (mt, t, seeds, n_all), n_rows_sec, row0 in zip(sections, rows_per_section, row_starts):
        top_bbox = axes[row0][0].get_position()
        bot_bbox = axes[row0 + n_rows_sec - 1][0].get_position()
        yc = (top_bbox.y1 + bot_bbox.y0) / 2
        fig.text(0.005, yc, F.MODELS[mt]['label'], ha="left", va="center",
                 fontsize=13, fontweight="normal", rotation=90)
    return fig


def draw_tdr_phase_space_grid(preds=(0, 1), model_type=None, ncols=6, n_seeds=None, rng_seed=0):
    """Supplementary (record of every seed, per the chat): one panel per
    seed, its own per-seed joint-basis TDR phase-space trajectory (pre
    solid; post dashed only if this seed is in the recovered-seeds set).
    Same start-marker/end-marker convention as draw_tdr_phase_space.

    n_seeds: if given, plot a reproducible random subset of this many seeds
    (fixed rng_seed) instead of every seed."""
    D = PTDR.population_tdr_per_seed_trajectories(model_type)
    t = D["t"]
    m = t >= 0
    xi, yi = preds
    all_seeds = D["seeds"]
    seeds = all_seeds
    if n_seeds is not None and n_seeds < len(seeds):
        seeds = random.Random(rng_seed).sample(seeds, n_seeds)
        seeds = sorted(seeds, key=lambda d: d["seed"])
    n = len(seeds)
    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.4 * nrows), squeeze=False)
    for i, d in enumerate(seeds):
        ax = axes[i // ncols][i % ncols]
        for si, stim in enumerate(STIM_ORDER):
            colour = S.STIM_COLOURS[stim]
            x, y = d["scores_pre"][m, si, xi], d["scores_pre"][m, si, yi]
            ax.plot(x, y, color=colour, ls="-", lw=1.0)
            ax.scatter(x[-1], y[-1], color=colour, s=18, marker="s", edgecolor="k", lw=0.3, zorder=3)
            if d["has_post"]:
                x2, y2 = d["scores_post"][m, si, xi], d["scores_post"][m, si, yi]
                ax.plot(x2, y2, color=colour, ls="--", lw=1.0)
                ax.scatter(x2[-1], y2[-1], color=colour, s=18, marker="s", edgecolor="k", lw=0.3, zorder=3)
        ax.set_title(f"seed {d['seed']}" + ("" if d["has_post"] else " (no post)"), fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(PRED_PRETTY[PRED_NAMES[xi]], fontsize=7)
        ax.set_ylabel(PRED_PRETTY[PRED_NAMES[yi]], fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    cov = "every seed" if len(seeds) == len(all_seeds) else f"{len(seeds)} of {len(all_seeds)} seeds"
    fig.suptitle(f"{F.MODELS[model_type]['label']} TDR phase space "
                f"({PRED_NAMES[xi]} vs {PRED_NAMES[yi]}) -- {cov}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def draw_tdr_phase_space_3d(model_type, preds=(0, 1, 2), ax=None, elev=22, azim=-60):
    """3D analogue of draw_tdr_phase_space -- since TDR has exactly 3
    predictor axes (stim/context/value), the default preds=(0,1,2) shows
    ALL of them jointly (not a projection dropping one, unlike PCA's 2D
    default which only ever showed 2 of many components) -- same
    trajectories/marker convention as draw_tdr_phase_space. Requires
    mpl_toolkits.mplot3d. If ax is given it must already be a 3D axes
    (projection="3d"); when ax is None one is created here."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
    if ax is None:
        fig = plt.figure(figsize=(8.6, 6.2))
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0.02, right=0.78, top=0.95, bottom=0.08)
    else:
        fig = ax.figure
    D = PTDR.population_tdr_trajectories(model_type)
    t = D["t"]
    m = t >= 0
    off_idx = int(np.argmin(np.abs(t[m] - 2.0)))
    xi, yi, zi = preds
    for si, stim in enumerate(STIM_ORDER):
        colour = S.STIM_COLOURS[stim]
        for mean, ls, lw in ((D["pre_mean"], "-", 1.8), (D["post_mean"], "--", 1.8)):
            x, y, z = mean[m, si, xi], mean[m, si, yi], mean[m, si, zi]
            ax.plot(x, y, z, color=colour, ls=ls, lw=lw)
            ax.scatter(x[off_idx], y[off_idx], z[off_idx], color=colour, s=30, marker="^",
                       edgecolor="k", lw=0.5, zorder=3)
            ax.scatter(x[-1], y[-1], z[-1], color=colour, s=32, marker="s",
                       edgecolor="k", lw=0.5, zorder=3)
        ax.scatter(D["pre_mean"][m, si, xi][0], D["pre_mean"][m, si, yi][0],
                   D["pre_mean"][m, si, zi][0], color=colour, s=24, zorder=3)
    ax.set_xlabel(PRED_PRETTY[PRED_NAMES[xi]])
    ax.set_ylabel(PRED_PRETTY[PRED_NAMES[yi]])
    ax.tick_params(axis="both", pad=2)
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    # ax.set_zlabel() places its label at a hardcoded, view-angle-independent
    # y=0 display pixel in this matplotlib version (a known mplot3d bug --
    # verified empirically across every elev/azim combination), which
    # renders it clipped off the bottom of the canvas every time. Placed
    # manually instead, in axes-fraction coordinates along the right edge,
    # which is immune to that bug. Font size matched to the x/y labels' OWN
    # actual size (read back after set_xlabel, rather than a hardcoded
    # number) so the z label never silently drifts out of sync with them
    # again if the mplstyle's axes.labelsize ever changes.
    z_fontsize = ax.xaxis.label.get_fontsize()
    ax.text2D(1.05, 0.5, PRED_PRETTY[PRED_NAMES[zi]], transform=ax.transAxes,
              rotation=90, ha="center", va="center", fontsize=z_fontsize)
    ax.view_init(elev=elev, azim=azim)
    stim_handles = [plt.Line2D([0], [0], color=S.STIM_COLOURS[s], lw=2.2, label=f"{STIM_PRETTY[s]} stim")
                   for s in STIM_ORDER]
    style_handles = [plt.Line2D([0], [0], color="0.25", ls="-", lw=1.8, label="pre-reversal"),
                     plt.Line2D([0], [0], color="0.25", ls="--", lw=1.8, label="post-reversal")]
    handles = stim_handles + style_handles + _traj_marker_handles()
    ax.legend(handles=handles, frameon=False, fontsize=6.5, loc="upper left", labelspacing=0.35)
    ax.set_title(f"{F.MODELS[model_type]['label']} TDR phase space (3D)\n"
                 f"(n={D['n_pre']} pre, {D['n_post']}/{D['n_post_total']} recovered post)",
                 fontsize=10)
    return fig


def build_all(show_tag=None):
    for mt in MODEL_TYPES:
        try:
            save_panel(draw_tdr_phase_space_grid(model_type=mt), "Dimensionality/TDR",
                       f"DIM.TDR.grid.{mt}", f"{mt}_tdr_phase_grid_all_seeds", show_tag)
        except Exception as e:
            print(f"  (skip TDR seed grid for {mt}: {e})")
        try:
            save_panel(draw_tdr_phase_space(mt), "Dimensionality/TDR",
                       f"DIM.TDR.phase.{mt}", f"{mt}_tdr_phase_space", show_tag)
        except Exception as e:
            print(f"  (skip TDR phase space for {mt}: {e})")
        try:
            save_panel(draw_tdr_phase_space_3d(mt), "Dimensionality/TDR",
                       f"DIM.TDR.phase3d.{mt}", f"{mt}_tdr_phase_space_3d", show_tag)
        except Exception as e:
            print(f"  (skip TDR phase space 3D for {mt}: {e})")
        try:
            save_panel(draw_tdr_timecourses(mt), "Dimensionality/TDR",
                       f"DIM.TDR.tc.{mt}", f"{mt}_tdr_timecourses", show_tag)
        except Exception as e:
            print(f"  (skip TDR timecourses for {mt}: {e})")


if __name__ == "__main__":
    build_all()
