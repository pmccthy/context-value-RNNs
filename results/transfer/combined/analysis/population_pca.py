"""Time-resolved PCA state-space trajectories -- RNN models.

RNN analog of neuronal-representations' PCA state-space pipeline (figures/
plot_pca.py + figures/panels/dimensionality.py's draw_pca_projections),
built on the ALREADY-COMPUTED per-timepoint population activity in
figure_data{,_reversal}.pkl's aligned_mean array -- no live model inference,
no torch, safe to import anywhere make_panels.py runs (see population_
timeresolved.py's own docstring for the same design point).

Methodology (deliberately adapted, not a literal port, because the RNN task
has far fewer distinct conditions than a real session -- 3 stimuli vs. many
real trial types -- see below):

  * The real pipeline's "pca_time_avg" variant fits PCA on the time-AVERAGED
    per-condition response (conditions x units). With only 3 stimuli here
    that matrix would have just 3 rows -- PCA on 3 points can carry at most
    2 non-trivial components after centering, which is too degenerate to
    give 3 usable PCs. Instead this module mirrors the real pipeline's OTHER
    variant, "pca_all_time" ("PCA fit on the full time course", per extract_
    pca.py's own docstring): the fitted matrix is (time x stimulus) rows by
    unit columns -- i.e. every timepoint of every stimulus's trial-averaged
    trace is its own observation. With n_align_ts=11 timepoints x 3 stimuli
    = 33 rows (per phase), or 66 once pre+post are pooled (see next point),
    this comfortably supports the top 3 components this module keeps.

  * PRE and POST reversal are fit as ONE JOINT basis per seed (pool the pre-
    phase and post-phase (time x stimulus) rows together before fitting),
    not two independent bases. This matches what the real repo's own pre/
    post overlay plots actually show (e.g. tdr_phase_space.png draws solid
    pre-reversal and dashed post-reversal trajectories AS DIRECTLY
    COMPARABLE curves in one shared 2D space) -- only possible if both
    phases were projected through the same fixed axes. Fitting the two
    phases with independent PCA bases would make "same PC1" pre and post
    mean two different things and the overlay would be uninterpretable.

  * PCA sign is arbitrary per fit (and per seed), so before pooling across
    seeds every component's sign is flipped, if needed, so that (100% stim
    - 0% stim) during the PRE-reversal stimulus epoch is positive on every
    PC. Pre-reversal is used as the anchor (not post) because pre-reversal's
    "100%" stimulus is unambiguously the highest-value condition for every
    seed; post-reversal's stim/value mapping is exactly what's flipped by
    the reversal, so it can't serve as a stable reference.

  * Seed inclusion mirrors the rest of this pipeline's own convention
    (vigour_value.py, reversal_gradients.py): the PRE-reversal average pools
    every seed present in the data (no reward-based filtering, matching
    population_timeresolved.py's own population-mean-trace convention);
    the POST-reversal average is restricted to RECOVERED seeds only
    (recovered_fraction >= THR, via reversal_gradients.recovered_seeds --
    see that module's docstring for why). Both curves are still projected
    through the SAME per-seed joint basis, so the asymmetric seed sets
    don't break comparability -- this is the same asymmetry already present
    throughout every other pre-vs-post panel in this pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "transfer" / "code"))
sys.path.insert(0, str(_HERE))

import figures as F  # noqa: E402
import reversal_gradients as RG  # noqa: E402

MODEL_TYPES = RG.MODEL_TYPES
REV_TAG = RG.REV_TAG
FIGURE_DATA_DIR = _HERE.parent / "transfer" / "figure_data"
FIGURE_DATA_REV_DIR = _HERE.parent / "transfer" / f"figure_data_reversal{REV_TAG}"

N_COMPONENTS = 3
STIM_ORDER = ["0", "50", "100"]     # index into the stim axis of aligned_mean
STIM_HIGH_IDX = 2                    # "100%" -- the pre-reversal sign anchor's high end
STIM_LOW_IDX = 0                     # "0%"   -- the pre-reversal sign anchor's low end

_D_CACHE = {}


def _D(phase):
    """Cached figure_data.pkl load for "pre" or "post"."""
    if phase not in _D_CACHE:
        path = FIGURE_DATA_DIR if phase == "pre" else FIGURE_DATA_REV_DIR
        _D_CACHE[phase] = F.load(str(path))
    return _D_CACHE[phase]


def _seed_joint_pca(aligned_pre, aligned_post, n_iti, stim_ts, n_components, smooth_window=3):
    """aligned_{pre,post}: (unit, time, stim) for ONE seed. Fits one PCA on
    the pooled (pre-rows + post-rows) matrix (see module docstring), then
    sign-aligns every component using the PRE-reversal stimulus epoch, and
    returns (scores_pre, scores_post, var_explained) with scores_* shaped
    (time, stim, n_components)."""
    u, t_pre, s = aligned_pre.shape
    t_post = aligned_post.shape[1]
    X_pre = aligned_pre.transpose(1, 2, 0).reshape(t_pre * s, u)
    X_post = aligned_post.transpose(1, 2, 0).reshape(t_post * s, u)
    X = np.concatenate([X_pre, X_post], axis=0)
    mean = X.mean(axis=0)
    k = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=k, random_state=0)
    scores = pca.fit_transform(X - mean)
    scores_pre = scores[: t_pre * s].reshape(t_pre, s, k)
    scores_post = scores[t_pre * s:].reshape(t_post, s, k)
    stim_epoch = slice(n_iti, n_iti + stim_ts)
    diff = (scores_pre[stim_epoch, STIM_HIGH_IDX, :].mean(axis=0)
            - scores_pre[stim_epoch, STIM_LOW_IDX, :].mean(axis=0))
    flip = np.where(diff < 0, -1.0, 1.0)
    scores_pre = scores_pre * flip
    scores_post = scores_post * flip
    if smooth_window and smooth_window > 1:
        # Display-only denoising of the projected trajectory (does NOT touch
        # the PCA fit itself, and commutes with the sign-flip above since
        # smoothing is linear and the sign is a constant per component) --
        # centred moving average, same helper/convention as reversal_
        # gradients.py's own smoothing of the raw vigour trace.
        for arr in (scores_pre, scores_post):
            for si in range(arr.shape[1]):
                for ci in range(arr.shape[2]):
                    arr[:, si, ci] = RG._smooth(arr[:, si, ci], smooth_window)
    return scores_pre, scores_post, pca.explained_variance_ratio_


def population_pca_per_seed_trajectories(model_type, n_components=N_COMPONENTS, smooth_window=3):
    """Per-SEED (not cross-seed-averaged) version of population_pca_
    trajectories() above -- for population_pca_panels.draw_pca_phase_space_
    grid(), which needs every seed's own trajectory (record of every seed,
    per the chat), not just the aggregate mean. Same per-seed joint-PCA fit
    and seed-inclusion convention (see module docstring) as the aggregate
    version, just not collapsed across seeds. Returns dict(t, period,
    seeds=[{seed, has_post, scores_pre, scores_post, var_explained}]) --
    has_post marks whether this seed is in the recovered-seeds set (i.e.
    whether scores_post is meaningful to draw)."""
    Dpre, Dpost = _D("pre"), _D("post")
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
        sp, sq, ve = _seed_joint_pca(aligned_pre[si], aligned_post[si], n_iti, stim_ts, n_components,
                                     smooth_window=smooth_window)
        out_seeds.append(dict(seed=seed, has_post=seed in recovered, scores_pre=sp,
                               scores_post=sq, var_explained=ve))
    return dict(t=np.arange(period["n_align_ts"]) - n_iti, period=period, seeds=out_seeds)


_traj_cache = {}


def population_pca_trajectories(model_type, n_components=N_COMPONENTS, smooth_window=3):
    """Per-model-type aggregate: fits the joint per-seed PCA (see module
    docstring) for every seed present in BOTH phases, then averages the
    (sign-aligned) projected trajectories -- PRE over all present seeds,
    POST over recovered-only seeds. Returns a dict:
      t                (n_align_ts,) time-from-stim-onset, in period bins
      period           D["period"] (segments / segment_bounds, shared pre/post)
      pre_mean/pre_sem     (time, stim, k)
      post_mean/post_sem   (time, stim, k)
      n_pre, n_post, n_post_total   seed counts (post_total = len(recovered_seeds))
      var_explained    (k,) mean per-seed explained-variance ratio (of the
                       seeds actually pooled into pre_mean)
    """
    key = (model_type, n_components, smooth_window)
    if key in _traj_cache:
        return _traj_cache[key]
    Dpre, Dpost = _D("pre"), _D("post")
    if Dpre["seeds"] != Dpost["seeds"]:
        raise ValueError("pre/post figure_data seed ordering mismatch")
    ti = Dpre["model_types"].index(model_type)
    aligned_pre = Dpre["aligned_mean"]["data"][ti]      # (seed, unit, time, stim)
    aligned_post = Dpost["aligned_mean"]["data"][ti]
    period = Dpre["period"]
    n_iti, stim_ts = period["n_iti_pre"], period["stim_ts"]
    seeds = Dpre["seeds"]
    recovered = RG.recovered_seeds(model_type)

    pre_list, post_list, var_list = [], [], []
    post_keep_mask = []
    for si, seed in enumerate(seeds):
        if not (np.isfinite(aligned_pre[si]).all() and np.isfinite(aligned_post[si]).all()):
            continue
        sp, sq, ve = _seed_joint_pca(aligned_pre[si], aligned_post[si], n_iti, stim_ts, n_components,
                                     smooth_window=smooth_window)
        pre_list.append(sp)
        post_list.append(sq)
        var_list.append(ve)
        post_keep_mask.append(seed in recovered)

    pre_arr = np.stack(pre_list, axis=0)                # (n_seeds, time, stim, k)
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
        var_explained=np.mean(var_list, axis=0),
    )
    _traj_cache[key] = out
    return out
