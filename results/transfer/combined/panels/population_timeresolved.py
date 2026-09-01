"""Time-resolved population-activity + pooled tuning-heatmap panels -- the
model-side analogue of neuronal-representations' panels/population.py
(draw_expert_mean/draw_reversal_mean) and the pooled half of panels/
subgroups.py (draw_pooled_heatmap), both driven entirely by the already-
computed figure_data{,_reversal}.pkl aligned_mean/tuning arrays -- no live
decoding, no sklearn needed, safe to import anywhere make_panels.py runs.

Seed selection matches transfer.py's draw_population_activity_bar (ALL seeds
present in the loaded figure_data, via np.nanmean/isfinite) -- NOT filtered
to recovered-only, since that filter only matters for post-reversal
trajectory panels where a flat failure trace would distort a learning curve
(see panels/vigour_value.py); a population mean / tuning heatmap is a
straight pooled summary the same way the real repo pools every session.

  draw_population_mean_trace(model_type, D, ax, phase_label)  per-stimulus
      mean hidden-unit activation over time (pool over seed & unit), +/- SEM
      across seeds -- the analogue of draw_expert_mean/draw_reversal_mean.
  draw_pooled_tuning_heatmap(model_type, D, ax, phase_label)  ALL seeds'
      units pooled into one unit x stimulus tuning heatmap, sorted by
      preferred stimulus -- the analogue of draw_pooled_heatmap.
  draw_metric_bar(model_type, key, D, ax, ...)  single per-stimulus bar of
      any D['scalars'][key] metric (used for the vigour / 'lick rate' bar
      analogue of draw_expert_lick_bar).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "style"))
sys.path.insert(0, str(_HERE.parent / "transfer" / "code"))

from _tags import new_panel, save_panel  # noqa: E402
import style as S  # noqa: E402
import figures as F  # noqa: E402

MODEL_TYPES = ["rl_only", "classif_rl", "classif_rl_readout_only"]
STIM_ORDER = ["0", "50", "100"]
REV_TAG = os.environ.get("REV_TAG", "")
HORIZON_LABEL = f" [{REV_TAG.lstrip('_')} horizon]" if REV_TAG else ""

FIGURE_DATA_DIR = _HERE.parent / "transfer" / "figure_data"
FIGURE_DATA_REV_DIR = _HERE.parent / "transfer" / f"figure_data_reversal{REV_TAG}"


def draw_population_mean_trace(model_type, D, ax=None, phase_label="pre"):
    fig, ax, _ = new_panel(ax, figsize=(6, 4.5))
    ti = D["model_types"].index(model_type)
    am = D["aligned_mean"]["data"][ti]             # (seed, unit, time, stim)
    n_seeds = int(np.isfinite(am[:, 0, 0, 0]).sum())
    per_seed_mean = np.nanmean(am, axis=1)          # (seed, time, stim) -- pool over units
    mean = np.nanmean(per_seed_mean, axis=0)        # (time, stim)
    sem = np.nanstd(per_seed_mean, axis=0) / np.sqrt(max(n_seeds, 1))
    n_iti = D["period"]["n_iti_pre"]
    bounds = D["period"]["segment_bounds"]
    t = np.arange(mean.shape[0]) - n_iti   # 0 = stimulus onset (matches the
                                            # real repo's "time from stimulus onset" axis)
    for si, stim in enumerate(STIM_ORDER):
        color = S.STIM_COLOURS[stim]
        ax.plot(t, mean[:, si], color=color, lw=2, label=f"{stim}%")
        ax.fill_between(t, mean[:, si] - sem[:, si], mean[:, si] + sem[:, si],
                        color=color, alpha=0.2, lw=0)
    for b in bounds[:-1]:
        ax.axvline(b[1] - n_iti - 0.5, color="0.6", ls=":", lw=1)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time from stim onset")
    ax.set_ylabel("Population activity (a.u.)")
    tag = HORIZON_LABEL if phase_label == "post" else ""
    ax.set_title(f"{F.MODELS[model_type]['label']} population mean, {phase_label} "
                 f"(n={n_seeds} seeds){tag}", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    return fig


def draw_pooled_tuning_heatmap(model_type, D, ax=None, phase_label="pre", colorbar=True):
    fig, ax, _ = new_panel(ax, figsize=(3.8, 5.2))
    ti = D["model_types"].index(model_type)
    tuning = D["tuning"]["data"][ti]     # (seed, stim, unit), stim order = STIM_ORDER (0,50,100)
    # Heatmap columns go 100% -> 0% (descending), matching the real repo's own
    # EXP_ORDER/REV_ORDER convention (subgroups.py: ["100_to_0","50","0_to_100"])
    # -- reverse of the ascending order used elsewhere (bars, traces).
    col_order = [2, 1, 0]
    tuning = tuning[:, col_order, :]
    heat_labels = [STIM_ORDER[i] for i in col_order]
    mat = tuning.transpose(0, 2, 1).reshape(-1, tuning.shape[1])   # (seed*unit, stim)
    finite_rows = np.isfinite(mat).all(axis=1)
    mat = mat[finite_rows]
    peak = mat.max(axis=1); pref = mat.argmax(axis=1)
    order = np.lexsort((-peak, pref))
    vmax = float(np.nanpercentile(mat, 99)) if mat.size else 1.0
    im = ax.imshow(mat[order], aspect="auto", cmap="hot", vmin=0, vmax=vmax, interpolation="nearest")
    # White dotted separators at preferred-stimulus group boundaries -- same
    # convention as the real repo's _draw_matrix (subgroups.py).
    for b in np.where(np.diff(pref[order]) != 0)[0]:
        ax.axhline(b + 0.5, color="white", lw=0.8, ls=(0, (1, 1.3)), alpha=0.9)
    ax.set_xticks(range(len(heat_labels))); ax.set_xticklabels([f"{s}%" for s in heat_labels])
    ax.set_yticks([])
    tag = HORIZON_LABEL if phase_label == "post" else ""
    ax.set_title(f"{F.MODELS[model_type]['label']}\nall units pooled, {phase_label} "
                 f"(n={mat.shape[0]}){tag}", fontsize=10)
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, label="mean activation")
    return fig


def draw_metric_bar(model_type, key, D, ax=None, ylabel=None, phase_label="pre"):
    fig, ax, _ = new_panel(ax, figsize=(5, 4.5))
    ti = D["model_types"].index(model_type)
    arr = D["scalars"][key][ti]     # (seed, stim)
    n = int(np.isfinite(arr[:, 0]).sum())
    mean = np.nanmean(arr, axis=0); sem = np.nanstd(arr, axis=0) / np.sqrt(max(n, 1))
    colours = [S.STIM_COLOURS[s] for s in STIM_ORDER]
    ax.bar(range(3), mean, yerr=sem, color=colours, capsize=4)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{s}%" for s in STIM_ORDER])
    ax.set_ylabel(ylabel or key)
    ax.set_xlabel("stimulus")
    tag = HORIZON_LABEL if phase_label == "post" else ""
    ax.set_title(f"{F.MODELS[model_type]['label']} {key}, {phase_label} (n={n} seeds){tag}", fontsize=10)
    return fig


def build_all(show_tag=None):
    try:
        Dpre = F.load(str(FIGURE_DATA_DIR))
    except Exception as e:
        print(f"  (skip population/tuning time-resolved panels: pre figure_data not found: {e})")
        return
    try:
        Dpost = F.load(str(FIGURE_DATA_REV_DIR))
    except Exception as e:
        Dpost = None
        print(f"  (post-reversal figure_data{REV_TAG} not found -- pre-only panels this run: {e})")

    for mt in MODEL_TYPES:
        try:
            save_panel(draw_population_mean_trace(mt, Dpre, phase_label="pre"),
                       "Transfer/Population activity", f"POP.mean_tr.pre.{mt}",
                       f"{mt}_population_mean_trace_pre", show_tag)
        except Exception as e:
            print(f"  (skip population mean trace (pre) for {mt}: {e})")
        if Dpost is not None:
            try:
                save_panel(draw_population_mean_trace(mt, Dpost, phase_label="post"),
                           "Transfer/Population activity", f"POP.mean_tr.post{REV_TAG}.{mt}",
                           f"{mt}_population_mean_trace_post{REV_TAG}", show_tag)
            except Exception as e:
                print(f"  (skip population mean trace (post) for {mt}: {e})")

        try:
            save_panel(draw_pooled_tuning_heatmap(mt, Dpre, phase_label="pre"),
                       "Transfer/Subgroups", f"SUB.heatmappool.pre.{mt}",
                       f"{mt}_pooled_heatmap_pre", show_tag)
        except Exception as e:
            print(f"  (skip pooled heatmap (pre) for {mt}: {e})")
        if Dpost is not None:
            try:
                save_panel(draw_pooled_tuning_heatmap(mt, Dpost, phase_label="post"),
                           "Transfer/Subgroups", f"SUB.heatmappool.post{REV_TAG}.{mt}",
                           f"{mt}_pooled_heatmap_post{REV_TAG}", show_tag)
            except Exception as e:
                print(f"  (skip pooled heatmap (post) for {mt}: {e})")

        try:
            save_panel(draw_metric_bar(mt, "vigour", Dpre, ylabel="vigour", phase_label="pre"),
                       "Transfer/Vigour+Value", f"BEH.vigourbar.pre.{mt}",
                       f"{mt}_vigour_bar_pre", show_tag)
        except Exception as e:
            print(f"  (skip vigour bar for {mt}: {e})")


if __name__ == "__main__":
    build_all()
