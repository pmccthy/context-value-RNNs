"""gradient_loss_panels.py

New composite rows enabled by the model_runs_instrumented /
reversal_5000_instrumented rerun (see checkpoint_weights.
full_weight_change_trajectory's docstring for the weight-side equivalent):
per-UPDATE gradient norms and individual loss terms, logged directly in
history.json (track_gradients=True) -- no checkpoint/torch loading needed,
these are already scalars-per-update.

Each seed's own per-update array is mean-binned to ~N_BINS points per phase
before averaging across seeds, purely so the composite panel isn't a solid
smear of 850-1700+ points per seed; the underlying data is real per-update
resolution, this is display-only downsampling (like the weight-diff rows'
per-checkpoint resolution, just coarser since checkpoints are already
sparser than updates).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "style"))
sys.path.insert(0, str(_HERE.parent / "transfer" / "code"))
sys.path.insert(0, str(_HERE.parent / "analysis"))

from _tags import new_panel, save_panel  # noqa: E402
import figure_config as FC  # noqa: E402
import checkpoint_weights as CW  # noqa: E402
import vigour_value as VV  # noqa: E402

MODEL_TYPES = CW.MODEL_TYPES
CW_REV_TAG = "_5k"
N_BINS = 60  # per phase, display-only downsampling -- see module docstring


def _load_history(rdir):
    p = Path(rdir) / "history.json"
    if not p.exists():
        return None
    return json.load(open(p))


def _binned(trials, values, n_bins=N_BINS):
    """Mean-bin (trials, values) into n_bins contiguous chunks (last chunk
    may be shorter) -- returns (bin_center_trials, bin_mean_values)."""
    n = len(values)
    if n == 0:
        return np.array([]), np.array([])
    n_bins = max(1, min(n_bins, n))
    idx_chunks = np.array_split(np.arange(n), n_bins)
    values = np.asarray(values, dtype=float)
    t_out = np.array([trials[c].mean() for c in idx_chunks])
    v_out = np.array([np.nanmean(values[c]) for c in idx_chunks])
    return t_out, v_out


def _phase_series(model_type, seed, key, rev_tag=CW_REV_TAG):
    """(t_pre, v_pre, t_post, v_post), each mean-binned, for one seed/key,
    read directly from the two phases' history.json. Returns None if either
    phase is missing or the key is absent/empty there (e.g. aux_loss for
    rl_only, grad_norm_stim_head for rl_only, nonneg_loss which is always
    empty in this rerun)."""
    pre_rdir = CW.run_dir(model_type, seed)
    post_rdir = CW._COMBINED_ROOT / f"model_runs_reversal{rev_tag}" / model_type / f"seed{seed}"

    def _one(rdir):
        h = _load_history(rdir)
        if h is None:
            return None
        vals = h.get(key)
        if not vals:
            return None
        n_trials, total_updates = h.get("n_trials"), h.get("total_updates")
        if not n_trials or not total_updates:
            return None
        u = np.arange(1, len(vals) + 1)
        trials = u * (n_trials / total_updates)
        return _binned(trials, vals)

    pre = _one(pre_rdir)
    post = _one(post_rdir)
    if pre is None or post is None or len(pre[0]) == 0 or len(post[0]) == 0:
        return None
    return pre[0], pre[1], post[0], post[1]


def _collect(model_type, seeds, key, rev_tag=CW_REV_TAG):
    """Same seed-averaging convention as weight_change._collect_seed_series
    -- interpolates onto the first successful seed's own bin-center grid per
    phase, so a seed whose bin count/edges differ slightly isn't dropped."""
    ref_pre = ref_post = None
    rows_pre, rows_post = [], []
    for seed in seeds:
        got = _phase_series(model_type, seed, key, rev_tag=rev_tag)
        if got is None:
            continue
        t_pre, v_pre, t_post, v_post = got
        if ref_pre is None:
            ref_pre, ref_post = t_pre, t_post
        rows_pre.append(np.interp(ref_pre, t_pre, v_pre))
        rows_post.append(np.interp(ref_post, t_post, v_post))
    if ref_pre is None:
        return None
    return ref_pre, ref_post, np.asarray(rows_pre), np.asarray(rows_post)


def _mean_sem(arr):
    n = arr.shape[0]
    mean = np.nanmean(arr, axis=0)
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return mean, sem


def _draw_multi_series(ax, model_type, seeds, series, ylabel, title_kind):
    reversal_x = VV._reversal_x(model_type, keep=set(seeds), units="trials")
    n_seeds_used = 0
    x_min, x_max = None, None
    for key, label, colour in series:
        result = _collect(model_type, seeds, key)
        if result is None:
            continue
        t_pre, t_post, arr_pre, arr_post = result
        # Drop the final bin of each phase: a reproducible end-of-phase
        # logging artifact (a sharp terminal spike right at the pre- and
        # post-reversal phase boundaries, present in every seed's raw
        # per-update log) that otherwise distorts both the plotted trend
        # and the x-axis extent near each boundary.
        t_pre, arr_pre = t_pre[:-1], arr_pre[:, :-1]
        t_post, arr_post = t_post[:-1], arr_post[:, :-1]
        n_seeds_used = max(n_seeds_used, arr_pre.shape[0])
        mean_pre, sem_pre = _mean_sem(arr_pre)
        mean_post, sem_post = _mean_sem(arr_post)
        t_full = np.concatenate([t_pre, t_post + reversal_x])
        mean = np.concatenate([mean_pre, mean_post])
        sem = np.concatenate([sem_pre, sem_post])
        ax.plot(t_full, mean, color=colour, lw=1.8, label=label)
        ax.fill_between(t_full, mean - sem, mean + sem, color=colour, alpha=0.15, lw=0)
        x_min = t_full[0] if x_min is None else min(x_min, t_full[0])
        x_max = t_full[-1] if x_max is None else max(x_max, t_full[-1])

    if n_seeds_used == 0:
        raise FileNotFoundError(f"no {title_kind} logs found for {model_type}")

    ax.axvline(reversal_x, color="0.2", linestyle=":", lw=1.5)
    ax.set_xlabel("trials")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    label = FC.MODELS.get(model_type, {}).get("label", model_type)
    ax.set_title(f"{label}: {title_kind}, full pre+reversal trajectory [n={n_seeds_used} seeds]")
    if x_min is not None:
        ax.set_xlim(x_min, x_max)
    return n_seeds_used


GRAD_SERIES = [
    ("grad_norm", "total", "#333333"),
    ("grad_norm_backbone", "backbone", "#3377bb"),
    ("grad_norm_vigour_head", "vigour head (actor)", "#aa3377"),
    ("grad_norm_value_head", "value head (critic)", "#229977"),
    ("grad_norm_stim_head", "stim head", "#ee9933"),
]


def draw_gradient_norms(model_type, ax=None, seeds=None):
    """Per-update gradient-norm magnitude, full pre+reversal trajectory,
    mean +/- SEM across recovered seeds -- total grad_norm plus each
    module's own norm (backbone, vigour_head, value_head, stim_head where
    present -- rl_only has no stim_head, so only 4 lines instead of 5).
    New: enabled by track_gradients in the model_runs_instrumented /
    reversal_5000_instrumented rerun."""
    fig, ax, _ = new_panel(ax, figsize=(7.0, 4.8))
    keep = VV.recovered_seeds(model_type)
    seeds = seeds if seeds is not None else (sorted(keep) if keep else CW.SEEDS)
    _draw_multi_series(ax, model_type, seeds, GRAD_SERIES, "gradient norm", "gradient norms")
    return fig


LOSS_SERIES = [
    ("policy_loss", "policy loss", "#aa3377"),
    ("value_loss", "value loss", "#229977"),
    ("activity_loss", "activity loss (regularizer)", "#3377bb"),
    ("aux_loss", "aux (SSL) loss", "#ee9933"),
]


def draw_loss_terms(model_type, ax=None, seeds=None):
    """Per-update individual loss terms, full pre+reversal trajectory, mean
    +/- SEM across recovered seeds -- policy_loss, value_loss, and the
    regularizer/auxiliary terms (activity_loss always logged though
    typically small unless the activity penalty is active; aux_loss only
    nonzero for the SSL models, empty/skipped for rl_only; nonneg_loss
    omitted -- always empty in this rerun, i.e. that penalty wasn't active).
    New: enabled by the same instrumented rerun as draw_gradient_norms."""
    fig, ax, _ = new_panel(ax, figsize=(7.0, 4.8))
    keep = VV.recovered_seeds(model_type)
    seeds = seeds if seeds is not None else (sorted(keep) if keep else CW.SEEDS)
    _draw_multi_series(ax, model_type, seeds, LOSS_SERIES, "loss", "loss terms")
    return fig


def build_all(show_tag=None):
    for mt in MODEL_TYPES:
        try:
            fig = draw_gradient_norms(mt)
        except Exception as e:
            print(f"  (skip gradient norms for {mt}: {e})")
        else:
            save_panel(fig, "Mechanistic/GradientsLosses", f"MECH.GRAD.{mt}",
                       f"{mt}_gradient_norms", show_tag)

        try:
            fig = draw_loss_terms(mt)
        except Exception as e:
            print(f"  (skip loss terms for {mt}: {e})")
        else:
            save_panel(fig, "Mechanistic/GradientsLosses", f"MECH.LOSS.{mt}",
                       f"{mt}_loss_terms", show_tag)


if __name__ == "__main__":
    build_all()
