"""Responder significance for context-value-RNNs model activations, using the
EXACT same temporal-cluster t-test method as the experimental figures (see
responsiveness_ttest.py in this directory, ported verbatim from
~/Documents/neuronal-representations/analysis/responsiveness_ttest.py).

This REPLACES the model repo's existing responder definition
(transfer_final/code/build_figure_data_from_timeresolved.py's per_model(): a
single paired t-test of stim-window-mean minus baseline-window-mean, one test
per unit per stimulus) with the finer per-timepoint independent t-test +
contiguous-run criterion used on the real neural data. The two will not agree
exactly -- this module exists so we can compare them, not just replace one
with the other silently.

Data source: transfer_final/figure_data/time_resolved/<model_type>_seed<NN>.npz
(and the *_reversal / *_reversal_5k variants), each holding a per-trial
aligned-activity array `aligned` (n_trials, n_timepoints, n_units), a
`stimulus` id per trial, and `bounds` = (n_iti, n_stim, n_outcome) timepoints
per trial segment. Baseline window = the n_iti ITI timepoints; stimulus
window = the n_stim timepoints right after ITI. Unlike the real data (many
imaging frames per window), the model has only a handful of timepoints per
window (bounds are typically (3, 5, 3)) -- responsiveness_ttest.t_test_temporal
already handles unequal-length windows by linearly upsampling the shorter one
(see its docstring), so this is applied unmodified, not adapted.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np

from responsiveness_ttest import responders_from_windows

# Must match figures.py's GROUP_ORDER / _GROUP_CODES bit-encoding exactly, so
# group counts here are directly comparable to the existing (non-temporal)
# responder_group_counts().
GROUP_LABELS = [
    "0%-only", "50%-only", "100%-only",
    "0% & 50%", "0% & 100%", "50% & 100%",
    "all three",
]
_GROUP_ORDER = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
_GROUP_CODES = [sum(1 << i for i in g) for g in _GROUP_ORDER]
_BITS = np.array([1, 2, 4])


def responders_temporal_one_seed(npz_path, alpha=0.05, max_int=3, frac=1.0 / 3.0):
    """Run the temporal-cluster t-test for one (model_type, seed) time_resolved
    file. Returns (H, 3) bool array: resp[:, s] = significant responders to
    stimulus s, matching the shape/semantics of build_figure_data_from_timeresolved's
    `resp` array (but from the temporal-cluster test, not the paired-window test)."""
    z = np.load(npz_path, allow_pickle=True)
    aligned = np.asarray(z["aligned"], dtype=np.float32)   # (trial, time, unit)
    stim = np.asarray(z["stimulus"])
    n_iti, stim_ts, _rew_ts = (int(b) for b in z["bounds"])
    H = aligned.shape[-1]
    resp = np.zeros((H, 3), dtype=bool)
    diag = {}
    for s in range(3):
        idx = stim == s
        baseline = aligned[idx, 0:n_iti, :]                # (n_trials_s, n_iti, H)
        stim_win = aligned[idx, n_iti:n_iti + stim_ts, :]   # (n_trials_s, stim_ts, H)
        out = responders_from_windows(baseline, stim_win, alpha=alpha,
                                       max_int=max_int, frac=frac)
        resp[:, s] = out["sig"]
        diag[s] = dict(n_trials=int(idx.sum()), n_sig=int(out["sig"].sum()))
    return resp, diag


def _classify(bitmask):
    for label, code in zip(GROUP_LABELS, _GROUP_CODES):
        if bitmask == code:
            return label
    return None  # not significant for any stimulus


def pooled_group_counts_temporal(time_resolved_dir, model_type, alpha=0.05,
                                  max_int=3, frac=1.0 / 3.0, seeds=None):
    """Pooled (summed over seeds) responder-group counts for one model type,
    using the temporal-cluster test. GROUP_LABELS order. Mirrors
    reversal_study/code/population_similarity.py's group_counts_pooled(), but
    against the new responder definition instead of the existing `responsive`
    field in figure_data.pkl."""
    pattern = str(Path(time_resolved_dir) / f"{model_type}_seed*.npz")
    files = sorted(glob.glob(pattern))
    if seeds is not None:
        wanted = set(seeds)
        files = [f for f in files
                 if int(re.search(r"_seed(\d+)\.npz$", f).group(1)) in wanted]
    counts = {label: 0 for label in GROUP_LABELS}
    n_nonresp = 0
    n_units_total = 0
    per_seed = {}
    for f in files:
        seed = int(re.search(r"_seed(\d+)\.npz$", f).group(1))
        resp, diag = responders_temporal_one_seed(f, alpha=alpha, max_int=max_int, frac=frac)
        codes = resp.astype(int) @ _BITS
        for code in codes:
            label = _classify(int(code))
            if label is None:
                n_nonresp += 1
            else:
                counts[label] += 1
        n_units_total += resp.shape[0]
        per_seed[seed] = diag
    return dict(counts=counts, n_units_total=n_units_total, n_nonresp=n_nonresp,
                n_seeds=len(files), per_seed=per_seed)


# --------------------------------------------------------------------------- #
# Native-timepoint variant (no upsampling): per user's explicit request, this
# does NOT stretch the 3-timepoint ITI/baseline window to match the 5-timepoint
# stim window via interpolation (unlike responsiveness_ttest.t_test_temporal,
# which upsamples for the real imaging use case). Instead the baseline window's
# native timepoints are POOLED into one reference sample (trials x n_iti
# flattened), and each of the STIM window's native timepoints is compared
# against that pooled baseline via its own independent two-sample t-test --
# no interpolated/synthetic data at any point, only timepoints actually present
# in the rollout. The contiguous-run criterion is then applied over the
# native n_stim_ts-length p-value sequence (frac defaults to 1/3, as before).
# --------------------------------------------------------------------------- #
from scipy.stats import ttest_ind as _ttest_ind          # noqa: E402
from responsiveness_ttest import length_of_continuous_subarray  # noqa: E402


REAL_TRIALS_PER_STIM = {0: 60, 1: 80, 2: 60}
"""Median per-session, per-stimulus trial counts in the real imaging data
(lick_rates_per_trial_expert.csv, n=43 sessions: 0%% median 60 [40-90], 50%%
median 80 [40-120], 100%% median 60 [40-90]). Passed as n_trials_target to
subsample the model's ~500+ trials/stimulus down to comparable statistical
power -- see chat: removing upsampling alone did not fix the "all three"
responder-group over-dominance, because the real driver is trial count, not
interpolation."""


def responders_native_one_seed(npz_path, alpha=0.05, max_int=3, frac=1.0 / 3.0,
                                n_trials_target=None, rng_seed=0):
    """Like responders_temporal_one_seed, but comparing each native stim
    timepoint against the POOLED (not upsampled) baseline distribution.

    n_trials_target: None (use all trials, as before), an int (same cap for
    all 3 stimuli), or a {0,1,2: int} dict (per-stimulus cap, e.g.
    REAL_TRIALS_PER_STIM) -- when the stimulus has more trials than the cap,
    a random subset (rng_seed-seeded, so reproducible) is drawn WITHOUT
    replacement and used for both the baseline and stim windows, so the
    test's statistical power roughly matches a real imaging session's."""
    z = np.load(npz_path, allow_pickle=True)
    aligned = np.asarray(z["aligned"], dtype=np.float32)   # (trial, time, unit)
    stim = np.asarray(z["stimulus"])
    n_iti, stim_ts, _rew_ts = (int(b) for b in z["bounds"])
    H = aligned.shape[-1]
    resp = np.zeros((H, 3), dtype=bool)
    diag = {}
    rng = np.random.default_rng(rng_seed)
    for s in range(3):
        idx = np.where(stim == s)[0]
        if n_trials_target is not None:
            target = n_trials_target[s] if isinstance(n_trials_target, dict) else n_trials_target
            if len(idx) > target:
                idx = np.sort(rng.choice(idx, size=target, replace=False))
        baseline = aligned[idx, 0:n_iti, :]                       # (n_trials_s, n_iti, H)
        baseline_pooled = baseline.reshape(-1, H)                 # (n_trials_s*n_iti, H)
        stim_win = aligned[idx, n_iti:n_iti + stim_ts, :]          # (n_trials_s, stim_ts, H)
        p_vals = np.zeros((stim_ts, H))
        for t in range(stim_ts):
            _, p = _ttest_ind(stim_win[:, t, :], baseline_pooled, axis=0)
            p_vals[t] = np.where(np.isnan(p), 1.0, p)
        thresh = frac * stim_ts
        sig = np.zeros(H, dtype=bool)
        for h in range(H):
            sig_idx = np.where(p_vals[:, h] < alpha)[0]
            run = length_of_continuous_subarray(sig_idx, max_int=max_int)
            sig[h] = run > thresh
        resp[:, s] = sig
        diag[s] = dict(n_trials=int(idx.sum()), n_sig=int(sig.sum()))
    return resp, diag


def pooled_group_counts_native(time_resolved_dir, model_type, alpha=0.05,
                                max_int=3, frac=1.0 / 3.0, seeds=None,
                                n_trials_target=None, rng_seed=0):
    """Native-timepoint-test analogue of pooled_group_counts_temporal(). See
    responders_native_one_seed for n_trials_target/rng_seed (trial-count
    subsampling to match real statistical power)."""
    pattern = str(Path(time_resolved_dir) / f"{model_type}_seed*.npz")
    files = sorted(glob.glob(pattern))
    if seeds is not None:
        wanted = set(seeds)
        files = [f for f in files
                 if int(re.search(r"_seed(\d+)\.npz$", f).group(1)) in wanted]
    counts = {label: 0 for label in GROUP_LABELS}
    n_nonresp = 0
    n_units_total = 0
    per_seed = {}
    for f in files:
        seed = int(re.search(r"_seed(\d+)\.npz$", f).group(1))
        resp, diag = responders_native_one_seed(f, alpha=alpha, max_int=max_int, frac=frac,
                                                n_trials_target=n_trials_target, rng_seed=rng_seed)
        codes = resp.astype(int) @ _BITS
        for code in codes:
            label = _classify(int(code))
            if label is None:
                n_nonresp += 1
            else:
                counts[label] += 1
        n_units_total += resp.shape[0]
        per_seed[seed] = diag
    return dict(counts=counts, n_units_total=n_units_total, n_nonresp=n_nonresp,
                n_seeds=len(files), per_seed=per_seed)
