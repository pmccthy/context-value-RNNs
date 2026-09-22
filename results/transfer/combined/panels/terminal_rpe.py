"""terminal_rpe.py

Per-stimulus reward-prediction-error (RPE / TD-error) "vs trials", mean+SEM
over seeds -- for the MAIN dataset. The main model_runs_ckpt / reversal_*_ckpt
datasets have NO live-logged RPE (track_gradients-style probing was never
turned on for them -- see combined/analysis/checkpoint_weights.py's docstring
for the parallel gradient-norm gap). What IS available without any retraining:
results/transfer/reversal/terminal_rpe/ already holds a POST-HOC "terminal"
RPE estimate for exactly this dataset's 5000-trial reversal horizon (REV_TAG=
"_5k") -- repeatedly evaluating the FROZEN final pre-reversal and final
post-reversal models (model.pt) on fresh trial draws (see
results/transfer/reversal/code/terminal_rpe.py). This is NOT a training
trajectory: every "block" comes from the same frozen weights, so point-to-
point spread across blocks is pure sampling noise, not learning. Rendered
here in the same "two-point scatter+errorbar, PLACEHOLDER-labelled" style
vigour_value.py already uses for its own terminal-value fallback branch, for
visual consistency within the same composite figure.

REV_TAG must be "_5k" for this data to be found (results/transfer/reversal/
terminal_rpe/ was computed against the reversal_5000_ckpt horizon only; the
2500-trial horizon has no terminal RPE yet).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "style"))
sys.path.insert(0, str(_HERE.parent / "transfer" / "code"))
sys.path.insert(0, str(_HERE.parent / "analysis"))

from _tags import new_panel, save_panel  # noqa: E402
import style as S  # noqa: E402
import figure_config as FC  # noqa: E402
import vigour_value as VV  # noqa: E402

MODEL_TYPES = VV.MODEL_TYPES
STIM_ORDER = VV.STIM_ORDER

TERMINAL_RPE_PRE = VV.TERMINAL_RPE_PRE
TERMINAL_RPE_POST = VV.TERMINAL_RPE_POST


def draw_rpe_vs_trials(model_type, ax=None):
    fig, ax, _ = new_panel(ax, figsize=(6.5, 4.5))
    keep = VV.recovered_seeds(model_type)
    n_total = len(VV._load_seed_histories(VV.MODEL_RUNS_POST, model_type, "vigour"))
    reversal_x = VV._reversal_x(model_type, keep=keep, units="trials")

    pre_hist = VV._load_seed_histories(TERMINAL_RPE_PRE, model_type, "rpe", keep=keep)
    post_hist = VV._load_seed_histories(TERMINAL_RPE_POST, model_type, "rpe", keep=keep)
    if not pre_hist and not post_hist:
        ax.text(0.5, 0.5, "no terminal probe_rpe data\n(recovered seeds)",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    span = max(reversal_x * 0.06, 20.0)
    rng = np.random.default_rng(0)
    n_post_seeds = len(post_hist)
    for label, hist, xc in [("pre", pre_hist, reversal_x - span * 1.5),
                             ("post", post_hist, reversal_x + span * 1.5)]:
        if not hist:
            continue
        all_y = np.concatenate([y for _, y in hist.values()], axis=0)  # (n_seed*n_block, 3)
        mean = np.nanmean(all_y, axis=0)
        sem = np.nanstd(all_y, axis=0) / np.sqrt(max(all_y.shape[0], 1))
        for si, s in enumerate(STIM_ORDER):
            colour = S.STIM_COLOURS[s]
            jitter = rng.uniform(-span * 0.4, span * 0.4, size=all_y.shape[0])
            ax.scatter(xc + jitter, all_y[:, si], color=colour, s=8, alpha=0.35, zorder=2)
            ax.errorbar([xc], [mean[si]], yerr=[sem[si]], fmt="o", color=colour,
                        markersize=9, capsize=4, zorder=3,
                        label=f"{s}%" if label == "pre" else None)

    ax.axhline(0, color="0.3", lw=0.8, zorder=1)
    ax.set_xlim(reversal_x - span * 4, reversal_x + span * 4)
    ax.set_xticks([reversal_x - span * 1.5, reversal_x + span * 1.5])
    ax.set_xticklabels(["pre", "post"])
    ax.set_ylabel("mean RPE (TD error)")
    ax.legend(frameon=False, fontsize=8)
    label_mt = FC.MODELS.get(model_type, {}).get("label", model_type)
    ax.set_title(f"{label_mt} -- terminal RPE (n={n_post_seeds}/{n_total} recovered)\n"
                 f"PLACEHOLDER: frozen-model repeated evals, not a training trajectory "
                 f"-- rerun with live RPE probing for the real curve")
    return fig


def build_all(show_tag=None):
    for mt in MODEL_TYPES:
        try:
            fig = draw_rpe_vs_trials(mt)
        except Exception as e:
            print(f"  (skip terminal RPE vs trials for {mt}: {e})")
            continue
        save_panel(fig, "Mechanistic/TerminalRPE", f"MECH.RPE.{mt}",
                   f"{mt}_terminal_rpe_vs_trials", show_tag)


if __name__ == "__main__":
    build_all()
