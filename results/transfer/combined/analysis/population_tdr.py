"""Time-resolved TDR (targeted dimensionality reduction) trajectories -- RNN
models.

RNN analog of neuronal-representations' canonical TDR pipeline (analysis/
run_tdr_v2.py's fit_tdr_peak_norm + figures/plot_tdr.py), built on the same
figure_data{,_reversal}.pkl aligned_mean array population_pca.py already
uses -- see that module's docstring for the general "no live model
inference, joint pre+post basis, pre-anchored sign convention, pre-
all-seeds/post-recovered-only aggregation" design points, all reused
unchanged here. This module only differs in HOW the axes are chosen: not by
unsupervised PCA, but by regressing each unit's response against a design
matrix of task variables, then orthogonalizing the resulting weight vectors
-- i.e. TDR proper (Mante et al. 2013), same technique the real repo's own
run_tdr_v2.py uses.

Design matrix (ported verbatim from run_tdr_v2.py's "stim_scalar_context_
value" variant -- the one the real pipeline's own headline "peak" TDR
projection uses, i.e. exactly the 3 axes tdr_phase_space.png plots):
  stim     physical stimulus identity, unchanged by reversal: -1 (0%-cue),
           0 (50%-cue), +1 (100%-cue)
  context  reversal phase: -1 (pre), +1 (post) -- "context" in this pipeline
           IS reversal phase, per the chat.
  value    the stimulus's CURRENT reward probability given phase: 0/.5/1 for
           the 0/50/100 cue pre-reversal, MIRRORED (1/.5/0) post-reversal --
           the same mirror-swap already used for the RNN recovery-gradient
           definition in reversal_gradients.py (MIRROR = {0:2,1:1,2:0}).
One regression per seed, per timepoint, across all 6 (stim x phase)
conditions at once -- exactly the "one joint basis spanning pre AND post"
design population_pca.py's own docstring justifies, here it's *required*
by construction (the "context"/"value" predictors are literally undefined
without both phases in the same design matrix), not just a choice.

Deliberate simplification vs. the real pipeline: run_tdr_v2.py adds a PCA-
denoising step (project betas through the top N_PCS singular vectors of the
raw per-trial data) before the peak-time / QR step, because real ΔF/F
traces are single-trial-noisy. aligned_mean is already a trial-averaged
estimate per unit (see figures.py's own docstring), so that denoising step
is skipped here (equivalent to using the identity denoising matrix) -- there
is much less to denoise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import population_pca as PPCA  # noqa: E402 -- reuses _D(), figure_data cache

MODEL_TYPES = PPCA.MODEL_TYPES
RG = PPCA.RG
STIM_ORDER = PPCA.STIM_ORDER            # ["0", "50", "100"]
PRED_NAMES = ["stim", "context", "value"]
STIM_HIGH_IDX = PPCA.STIM_HIGH_IDX      # "100%" -- pre-reversal sign anchor
STIM_LOW_IDX = PPCA.STIM_LOW_IDX        # "0%"

_STIM_SCALAR = {0: -1.0, 1: 0.0, 2: 1.0}
_MIRROR = {0: 2, 1: 1, 2: 0}             # same mapping as reversal_gradients.MIRROR


def _design_matrix():
    """(6, 3) rows = [pre stim0, pre stim1, pre stim2, post stim0, post
    stim1, post stim2], columns = [stim, context, value] (see module
    docstring). Fixed -- doesn't depend on data, so built once."""
    rows = []
    for phase, ctx in (("pre", -1.0), ("post", 1.0)):
        for stim_idx in range(3):
            stim = _STIM_SCALAR[stim_idx]
            value_idx = stim_idx if phase == "pre" else _MIRROR[stim_idx]
            value = _STIM_SCALAR[value_idx] * 0.5 + 0.5   # -1/0/1 -> 0/.5/1
            rows.append([stim, ctx, value])
    return np.asarray(rows, dtype=float)


X_DESIGN = _design_matrix()


def _seed_joint_tdr(aligned_pre, aligned_post, n_iti, stim_ts, smooth_window=3):
    """aligned_{pre,post}: (unit, time, stim) for ONE seed. Returns
    (scores_pre, scores_post, peak_times) -- scores_* shaped (time, stim,
    n_pred), peak_times shaped (n_pred,) in stim-epoch-relative bins."""
    u, t_pre, s = aligned_pre.shape
    t_post = aligned_post.shape[1]
    assert t_pre == t_post, "pre/post time axes must match for a joint TDR fit"
    n_time = t_pre
    # (6, unit, time): conditions 0-2 = pre stim 0/1/2, 3-5 = post stim 0/1/2
    Y = np.concatenate([aligned_pre.transpose(2, 0, 1), aligned_post.transpose(2, 0, 1)], axis=0)

    betas_time = np.zeros((3, u, n_time))
    for t in range(n_time):
        mdl = LinearRegression(fit_intercept=False)
        mdl.fit(X_DESIGN, Y[:, :, t])
        betas_time[:, :, t] = mdl.coef_.T

    stim_epoch = slice(n_iti, n_iti + stim_ts)
    peak_times = np.zeros(3, dtype=int)
    betas_peak = np.zeros((3, u))
    for i in range(3):
        norms = np.linalg.norm(betas_time[i, :, stim_epoch], axis=0)
        peak_times[i] = int(np.argmax(norms)) + n_iti
        betas_peak[i] = betas_time[i, :, peak_times[i]]

    Q, _ = np.linalg.qr(betas_peak.T)     # (u, 3)
    tdr_axes = Q.T                        # (3, u), orthonormal rows

    # project the FULL trajectory (all timepoints, both phases) onto the axes
    scores = np.einsum("pu,cut->cpt", tdr_axes, Y)   # (6, 3, time)
    scores_pre = scores[:3].transpose(2, 0, 1)        # (time, stim, pred)
    scores_post = scores[3:].transpose(2, 0, 1)

    diff = (scores_pre[stim_epoch, STIM_HIGH_IDX, :].mean(axis=0)
            - scores_pre[stim_epoch, STIM_LOW_IDX, :].mean(axis=0))
    flip = np.where(diff < 0, -1.0, 1.0)
    scores_pre = scores_pre * flip
    scores_post = scores_post * flip
    if smooth_window and smooth_window > 1:
        # Display-only denoising of the projected trajectory (does NOT touch
        # the TDR axis fit -- peak_times/betas_peak above are computed from
        # the unsmoothed betas_time) -- same convention as population_pca.
        # _seed_joint_pca's own smoothing.
        for arr in (scores_pre, scores_post):
            for si in range(arr.shape[1]):
                for ci in range(arr.shape[2]):
                    arr[:, si, ci] = RG._smooth(arr[:, si, ci], smooth_window)
    return scores_pre, scores_post, peak_times - n_iti


def population_tdr_per_seed_trajectories(model_type, smooth_window=3):
    """Per-SEED (not cross-seed-averaged) version of population_tdr_
    trajectories() above -- for population_tdr_panels.draw_tdr_phase_space_
    grid(), which needs every seed's own trajectory (record of every seed,
    per the chat). Same per-seed joint-TDR fit and seed-inclusion convention
    as the aggregate version. Returns dict(t, period, seeds=[{seed,
    has_post, scores_pre, scores_post, peak_times}])."""
    Dpre, Dpost = PPCA._D("pre"), PPCA._D("post")
    if Dpre["seeds"] != Dpost["seeds"]:
        raise ValueError("pre/post figure_data seed ordering mismatch")
    ti = Dpre["model_types"].index(model_type)
    aligned_pre = Dpre["aligned_mean"]["data"][ti]
    aligned_post = Dpost["aligned_mean"]["data"][ti]
    period = Dpre["period"]
    n_iti, stim_ts = period["n_iti_pre"], period["stim_ts"]
    seeds = Dpre["seeds"]
    recovered = RG.recovered_seeds(model_type)
    out_seeds = []
    for si, seed in enumerate(seeds):
        if not (np.isfinite(aligned_pre[si]).all() and np.isfinite(aligned_post[si]).all()):
            continue
        sp, sq, pk = _seed_joint_tdr(aligned_pre[si], aligned_post[si], n_iti, stim_ts,
                                     smooth_window=smooth_window)
        out_seeds.append(dict(seed=seed, has_post=seed in recovered, scores_pre=sp,
                               scores_post=sq, peak_times=pk))
    return dict(t=np.arange(period["n_align_ts"]) - n_iti, period=period, seeds=out_seeds)


_traj_cache = {}


def population_tdr_trajectories(model_type, smooth_window=3):
    """Per-model-type aggregate, same structure/seed-inclusion convention as
    population_pca.population_pca_trajectories -- see that function's
    docstring. Returns a dict with the same keys, `var_explained` replaced
    by `peak_times` ((3,) mean per-seed peak bin, stim-onset-relative, per
    predictor)."""
    key = (model_type, smooth_window)
    if key in _traj_cache:
        return _traj_cache[key]
    Dpre, Dpost = PPCA._D("pre"), PPCA._D("post")
    if Dpre["seeds"] != Dpost["seeds"]:
        raise ValueError("pre/post figure_data seed ordering mismatch")
    ti = Dpre["model_types"].index(model_type)
    aligned_pre = Dpre["aligned_mean"]["data"][ti]
    aligned_post = Dpost["aligned_mean"]["data"][ti]
    period = Dpre["period"]
    n_iti, stim_ts = period["n_iti_pre"], period["stim_ts"]
    seeds = Dpre["seeds"]
    recovered = RG.recovered_seeds(model_type)

    pre_list, post_list, peak_list = [], [], []
    post_keep_mask = []
    for si, seed in enumerate(seeds):
        if not (np.isfinite(aligned_pre[si]).all() and np.isfinite(aligned_post[si]).all()):
            continue
        sp, sq, pk = _seed_joint_tdr(aligned_pre[si], aligned_post[si], n_iti, stim_ts,
                                     smooth_window=smooth_window)
        pre_list.append(sp)
        post_list.append(sq)
        peak_list.append(pk)
        post_keep_mask.append(seed in recovered)

    pre_arr = np.stack(pre_list, axis=0)
    post_arr_all = np.stack(post_list, axis=0)
    post_keep_mask = np.array(post_keep_mask, dtype=bool)
    post_arr = post_arr_all[post_keep_mask]
    n_pre = pre_arr.shape[0]
    n_post = int(post_keep_mask.sum())

    def _mean_sem(arr):
        n = arr.shape[0]
        m = arr.mean(axis=0)
        sem = arr.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(m)
        return m, sem

    pre_mean, pre_sem = _mean_sem(pre_arr)
    post_mean, post_sem = _mean_sem(post_arr) if n_post else (np.full_like(pre_mean, np.nan),
                                                               np.full_like(pre_mean, np.nan))
    out = dict(
        t=np.arange(period["n_align_ts"]) - n_iti,
        period=period,
        pre_mean=pre_mean, pre_sem=pre_sem,
        post_mean=post_mean, post_sem=post_sem,
        n_pre=n_pre, n_post=n_post, n_post_total=len(recovered),
        peak_times=np.mean(peak_list, axis=0),
    )
    _traj_cache[key] = out
    return out
