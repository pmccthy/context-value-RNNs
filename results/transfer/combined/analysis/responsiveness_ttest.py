"""Temporal t-test responsiveness (contiguous-significant-timepoints criterion).

Ported VERBATIM from
~/Documents/neuronal-representations/analysis/responsiveness_ttest.py (itself
ported from analysis/26_08_26_responsiveness_t_test.ipynb) so model
responder-significance uses the exact same statistical method as the
experimental figures: per neuron, per stimulus, an independent t-test at each
timepoint compares the stimulus window against the pre-stimulus baseline; the
neuron is a significant responder when its longest contiguous run of
significant timepoints exceeds n_timepoints / 3.

Only numpy + scipy dependencies -- no changes made from the source file
(model_responders.py in this directory adapts it to context-value-RNNs' own
per-trial aligned-activity arrays; nothing below is model-specific).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ttest_ind


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _resample_axis(arr: np.ndarray, n_target: int, axis: int = 1) -> np.ndarray:
    """Linearly resample ``arr`` along ``axis`` to ``n_target`` samples -- works
    for either upsampling (n_target > n_src) or downsampling (n_target < n_src),
    same fractional-position (0..1) interpolation either way.

    Stand-in for the notebook's ``utils.upsample_nd`` (only ever used to equalise
    the two windows' timepoint counts before the timepoint-wise t-test). Kept
    the original name as an alias below since the real-data pipeline's default
    usage (equal-length windows -> never hits this branch) is unaffected either
    way.
    """
    n_src = arr.shape[axis]
    if n_src == n_target:
        return arr
    src = np.linspace(0.0, 1.0, n_src)
    tgt = np.linspace(0.0, 1.0, n_target)
    moved = np.moveaxis(arr, axis, 0)                    # (time, ...)
    flat = moved.reshape(n_src, -1)
    out = np.empty((n_target, flat.shape[1]), dtype=flat.dtype)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(tgt, src, flat[:, j])
    out = out.reshape((n_target,) + moved.shape[1:])
    return np.moveaxis(out, 0, axis)


_upsample_axis = _resample_axis  # backward-compat alias (name no longer accurate for downsample use)


def t_test_temporal(cond_A, cond_B, timepoints=(), multiple_test=None, resize="upsample"):
    """Independent t-test at each timepoint between condition A and B, per neuron.

    ``cond_A``/``cond_B`` are ``(n_trials, n_timepoints, n_neurons)`` (a 2-D
    ``(n_trials, n_timepoints)`` array is treated as a single neuron). Returns
    ``t_vals, p_vals`` each shaped ``(n_timepoints, n_neurons)``.

    If the two inputs differ in timepoint count, ``resize`` picks which one is
    linearly resampled to match the other (same fractional-position interp
    either way -- see _resample_axis):
      "upsample" (default, matches the original notebook): the SHORTER window
        is stretched up to the LONGER one's length -- this is what the
        real-data pipeline uses (irrelevant there in practice since both
        windows are already equal-length/equal-fs), and remains the default
        so nothing that doesn't pass ``resize`` explicitly changes behaviour.
      "downsample": the LONGER window is compressed down to the SHORTER one's
        native length instead -- e.g. on the model side, where the ITI/
        baseline window often has fewer native timepoints than the stimulus
        window, this avoids inventing interpolated baseline samples and
        instead subsamples (via linear interpolation) the stimulus window
        down to the baseline's real timepoint count -- see model_responders.py.

    ``multiple_test`` mirrors the notebook argument; the default (``None``)
    leaves p-values uncorrected across timepoints (the notebook's main run),
    any other value applies a Benjamini-Hochberg FDR per neuron.
    """
    cond_A = np.asarray(cond_A)
    cond_B = np.asarray(cond_B)
    if cond_A.shape[1] != cond_B.shape[1]:
        lens = [cond_A.shape[1], cond_B.shape[1]]
        if resize == "upsample":
            target_idx = int(np.argmin(lens))    # the shorter one gets resized...
            target_len = int(np.max(lens))        # ...up to the longer one's length
        elif resize == "downsample":
            target_idx = int(np.argmax(lens))    # the longer one gets resized...
            target_len = int(np.min(lens))         # ...down to the shorter one's length
        else:
            raise ValueError(f"bad resize: {resize!r} (want 'upsample' or 'downsample')")
        resized = _resample_axis([cond_A, cond_B][target_idx], target_len, axis=1)
        if target_idx == 0:
            cond_A = resized
        else:
            cond_B = resized
    assert cond_A.shape[1] == cond_B.shape[1], (
        f"timepoint mismatch: {cond_A.shape[1]} != {cond_B.shape[1]}")

    n_timepoints = cond_A.shape[1]
    n_neurons = cond_A.shape[-1] if cond_A.ndim == 3 else 1
    A = cond_A.reshape(cond_A.shape[0], n_timepoints, n_neurons)
    B = cond_B.reshape(cond_B.shape[0], n_timepoints, n_neurons)

    t_vals, p_vals = ttest_ind(A, B, axis=0)             # (n_timepoints, n_neurons)
    t_vals = np.nan_to_num(np.asarray(t_vals), nan=0.0)
    p_vals = np.asarray(p_vals)
    p_vals = np.where(np.isnan(p_vals), 1.0, p_vals)     # degenerate timepoints -> ns

    if multiple_test is not None:
        from statsmodels.stats.multitest import multipletests
        for n in range(n_neurons):
            try:
                _, p_corr, _, _ = multipletests(p_vals[:, n], alpha=0.05,
                                                method=multiple_test)
                p_vals[:, n] = p_corr
            except Exception:
                pass
    return t_vals, p_vals


def length_of_continuous_subarray(arr, max_int=1):
    """Longest tolerated-contiguous run of significant timepoints.

    Verbatim port of the notebook: ``arr`` is the sequence of significant
    timepoint indices (e.g. ``[1,2,3,4,6,7,9]``); ``max_int`` tolerates small
    lapses in the run (gaps up to ``max_int``).
    """
    def longest_false_streak(a):
        max_len = current = 0
        for val in a:
            if not val:
                current += 1
                max_len = max(max_len, current)
            else:
                current = 0
        return max_len

    arr = np.asarray(arr)
    if arr.size < 2:
        return 0
    arr_intervals = np.diff(arr)
    arr_intervals_continuous = arr_intervals <= max_int
    return longest_false_streak(np.diff(arr_intervals_continuous))


def significant_responders(p_vals, alpha=0.05, max_int=3, frac=1.0 / 3.0):
    """Boolean ``(n_neurons,)`` mask of significant responders.

    A neuron is significant when its longest tolerated-contiguous run of
    timepoints with ``p < alpha`` exceeds ``frac * n_timepoints`` (``> n/3`` in
    the notebook). ``p_vals`` is ``(n_timepoints, n_neurons)``.
    """
    p_vals = np.asarray(p_vals)
    n_timepoints, n_neurons = p_vals.shape
    thresh = frac * n_timepoints
    out = np.zeros(n_neurons, dtype=bool)
    for n in range(n_neurons):
        sig_idx = np.where(p_vals[:, n] < alpha)[0]
        run = length_of_continuous_subarray(sig_idx, max_int=max_int)
        out[n] = run > thresh
    return out


# --------------------------------------------------------------------------- #
# end-to-end convenience
# --------------------------------------------------------------------------- #
def scale_per_neuron(y_window):
    """Z-score each neuron across the pooled (trial x time) samples of a window.

    ``y_window`` is ``(n_trials, n_timepoints, n_neurons)``. Mirrors the
    notebook's per-neuron scaling so effect sizes are comparable across neurons;
    the t-test statistic itself is invariant to this affine scaling.
    """
    y = np.asarray(y_window, dtype=float)
    flat = y.reshape(-1, y.shape[-1])
    mu = np.nanmean(flat, axis=0)
    sd = np.nanstd(flat, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (y - mu) / sd


def responders_from_windows(baseline, stim, alpha=0.05, max_int=3,
                            frac=1.0 / 3.0, scale=True, multiple_test=None,
                            resize="upsample"):
    """Per-neuron responsiveness from baseline & stimulus windows.

    ``baseline`` and ``stim`` are ``(n_trials, n_timepoints, n_neurons)`` arrays
    (same trials, sliced to the two windows). Returns a dict with:
      ``sig``     (n_neurons,) bool   significant responder
      ``effect``  (n_neurons,) float  mean(stim) - mean(baseline), trial-averaged
      ``p_vals``  (n_timepoints, n_neurons)
      ``run``     (n_neurons,) int     longest tolerated-contiguous sig run

    When ``scale`` is True both windows are z-scored per neuron together (so the
    reported effect size is in comparable, ~z units). NOTE: scaling pools
    baseline+stim BEFORE any resize below, so it's computed over each window's
    own native samples regardless of ``resize``.

    ``resize`` is passed straight through to t_test_temporal -- "upsample"
    (default, original behaviour) or "downsample"; see that function's
    docstring. Only matters when baseline/stim have different native
    timepoint counts (never happens for the real imaging windows; routinely
    happens for the model's ITI-vs-stim windows -- see model_responders.py).
    """
    baseline = np.asarray(baseline, dtype=float)
    stim = np.asarray(stim, dtype=float)
    if scale:
        nb = baseline.shape[1]
        pooled = np.concatenate([baseline, stim], axis=1)
        pooled = scale_per_neuron(pooled)
        baseline, stim = pooled[:, :nb], pooled[:, nb:]

    _, p_vals = t_test_temporal(baseline, stim, multiple_test=multiple_test, resize=resize)
    n_timepoints, n_neurons = p_vals.shape
    thresh = frac * n_timepoints
    sig = np.zeros(n_neurons, dtype=bool)
    run = np.zeros(n_neurons, dtype=int)
    for n in range(n_neurons):
        sig_idx = np.where(p_vals[:, n] < alpha)[0]
        run[n] = length_of_continuous_subarray(sig_idx, max_int=max_int)
        sig[n] = run[n] > thresh
    effect = (stim.mean(axis=1) - baseline.mean(axis=1)).mean(axis=0)  # (n_neurons,)
    return {"sig": sig, "effect": np.asarray(effect), "p_vals": p_vals, "run": run}
