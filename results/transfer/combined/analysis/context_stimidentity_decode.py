"""New decoders for the model-side FIG2 equivalent, built to match
cxval.analysis's own conventions (StandardScaler + LinearSVC + k-fold CV,
balanced downsampling) exactly -- NOT added to cxval/analysis.py itself
(that's the shared reproducibility library; these are figure-specific
derived analyses, kept here instead).

Three decoders, each the model-side analogue of one decoding_pooled.py bar:

  context_decode_per_stim   draw_reversal_context_bar's analogue: per
                             (identity-anchored) stimulus, can you tell PRE
                             from POST trials apart from that stimulus's own
                             activity? (chance = 0.5; low = the code for that
                             cue is context-invariant / value-driven only.)

  value_decode_pooled       draw_reversal_value_bar's analogue ("value_xor":
                             0% vs 100% cue, decoded by CURRENT reward value,
                             i.e. binary_value_decode_within averaged over
                             pre and post -- each phase's low/high labels
                             use THAT phase's own value_matrix column, so a
                             swap cue is correctly relabelled post-reversal).

  stimidentity_decode_pooled draw_reversal_stimidentity_bar's analogue: pool
                             PRE+POST trials together (context ignored) and
                             decode 0%-cue vs 100%-cue by IDENTITY -- tests
                             whether there's a genuinely context-invariant
                             per-cue signature, independent of value.

All operate on the same act_dict convention as cxval.analysis (stim_hidden/
reward_hidden/baseline_hidden, stimulus, context; see act_dict_adapter.py).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def _fit_score(X_tr, y_tr, X_te, y_te):
    sc = StandardScaler().fit(X_tr)
    clf = LinearSVC(max_iter=2000, dual="auto").fit(sc.transform(X_tr), y_tr)
    return clf.score(sc.transform(X_te), y_te)


def context_decode_per_stim(act_dict, period, pooling, n_folds=5, random_state=42):
    """Decode context (pre=0/post=1) from ONE stimulus's own pooled (both
    phases) trials, per stimulus. Returns acc (n_stim,); NaN where too few
    trials. Chance = 0.5."""
    ctx_arr = act_dict["context"]
    stim_arr = act_dict["stimulus"]
    hidden = act_dict["stim_hidden" if period == "stim" else "reward_hidden"]
    n_stim = int(stim_arr.max()) + 1
    ts, H = hidden.shape[1], hidden.shape[2]
    acc = np.full(n_stim, np.nan)
    rng = np.random.default_rng(random_state)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for s in range(n_stim):
        mask = stim_arr == s
        h_pre = hidden[mask & (ctx_arr == 0)]
        h_post = hidden[mask & (ctx_arr == 1)]
        n_min = min(len(h_pre), len(h_post))
        if n_min < 2 or 2 * n_min < n_folds * 2:
            continue
        h_pre = h_pre[rng.choice(len(h_pre), n_min, replace=False)]
        h_post = h_post[rng.choice(len(h_post), n_min, replace=False)]
        h_all = np.concatenate([h_pre, h_post], axis=0)
        y_trial = np.array([0] * n_min + [1] * n_min)

        fold_accs = []
        for train_idx, test_idx in skf.split(np.arange(2 * n_min), y_trial):
            if pooling == "average":
                X_tr, y_tr = h_all[train_idx].mean(1), y_trial[train_idx]
                X_te, y_te = h_all[test_idx].mean(1), y_trial[test_idx]
            else:
                X_tr = h_all[train_idx].reshape(-1, H)
                X_te = h_all[test_idx].reshape(-1, H)
                y_tr = np.repeat(y_trial[train_idx], ts)
                y_te = np.repeat(y_trial[test_idx], ts)
            fold_accs.append(_fit_score(X_tr, y_tr, X_te, y_te))
        acc[s] = np.mean(fold_accs)
    return acc


def value_decode_pooled(act_dict, period, pooling, value_matrix, threshold=0.25,
                        n_folds=5, random_state=42):
    """Mean of binary_value_decode_within's per-context accuracy (0%-cue vs
    100%-cue, labelled by that context's OWN current reward value -- so a
    swap cue is correctly relabelled post-reversal) -- a single summary
    number, the model-side analogue of the real repo's single 'value_xor'
    bar. Requires cxval.analysis.binary_value_decode_within."""
    from cxval.analysis import binary_value_decode_within
    acc = binary_value_decode_within(act_dict, period, pooling, value_matrix,
                                     threshold=threshold, n_folds=n_folds,
                                     random_state=random_state)
    return float(np.nanmean(acc))


def stimidentity_decode_from_cross(cross_mean, stim_pair=(0, 2)):
    """CORRECTED design (see chat -- the original pooled-CV version below is
    kept only as a documented dead end): stim-identity decode MUST be a
    proper cross-context train/test split (train on context A's (si,sj)
    trials, test on context B's), exactly like crosscontext_decode already
    does for every pair -- so this just reads the (si,sj) entry straight out
    of the cross_mean array run_decoding.py already computes via
    A.crosscontext_decode, averaged over both train/test directions.

    WHY NOT pool pre+post into one k-fold CV (the original approach)? With
    only 2 identities x 2 phases = 4 tight, well-separated clusters in a
    128-d hidden space, a linear decoder can find SOME hyperplane that
    perfectly separates almost ANY consistent 2-vs-2 grouping of those 4
    clusters within a single CV fold (excess capacity relative to how few,
    how tight the clusters are) -- so pooled-CV 'identity' accuracy came
    back at ceiling (1.0) even for rl_only, flatly contradicting rl_only's
    already-established near-chance CROSS-CONTEXT transfer (~0.50-0.58,
    generalisation_matrix off-diagonal) for the exact same units. The
    pooled version was measuring "are these 4 clusters linearly separable
    somehow", not "is there a context-invariant identity axis" -- only a
    genuine held-out-context test answers that question, which is exactly
    what crosscontext_decode already provides."""
    si, sj = stim_pair
    return float(np.nanmean([cross_mean[0, 1, si, sj], cross_mean[1, 0, si, sj]]))


def stimidentity_decode_pooled(act_dict, period, pooling, stim_pair=(0, 2),
                               n_folds=5, random_state=42):
    """DEAD END -- do not use for anything claiming to measure context-
    invariant identity coding. Kept only so the failure mode documented in
    stimidentity_decode_from_cross's docstring is reproducible/inspectable;
    superseded by stimidentity_decode_from_cross, which reuses
    crosscontext_decode's already-correct held-out-context design instead
    of pooling contexts into one CV split."""
    stim_arr = act_dict["stimulus"]
    hidden = act_dict["stim_hidden" if period == "stim" else "reward_hidden"]
    ts, H = hidden.shape[1], hidden.shape[2]
    si, sj = stim_pair
    h_i = hidden[stim_arr == si]
    h_j = hidden[stim_arr == sj]
    n_min = min(len(h_i), len(h_j))
    if n_min < 2 or 2 * n_min < n_folds * 2:
        return float("nan")
    rng = np.random.default_rng(random_state)
    h_i = h_i[rng.choice(len(h_i), n_min, replace=False)]
    h_j = h_j[rng.choice(len(h_j), n_min, replace=False)]
    h_all = np.concatenate([h_i, h_j], axis=0)
    y_trial = np.array([0] * n_min + [1] * n_min)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_accs = []
    for train_idx, test_idx in skf.split(np.arange(2 * n_min), y_trial):
        if pooling == "average":
            X_tr, y_tr = h_all[train_idx].mean(1), y_trial[train_idx]
            X_te, y_te = h_all[test_idx].mean(1), y_trial[test_idx]
        else:
            X_tr = h_all[train_idx].reshape(-1, H)
            X_te = h_all[test_idx].reshape(-1, H)
            y_tr = np.repeat(y_trial[train_idx], ts)
            y_te = np.repeat(y_trial[test_idx], ts)
        fold_accs.append(_fit_score(X_tr, y_tr, X_te, y_te))
    return float(np.mean(fold_accs))


def context_decode_baseline_control(act_dict, n_folds=5, random_state=42):
    """Control for context_decode_per_stim: decode context from the ITI
    BASELINE window alone (no stimulus input at all, pooled across all
    stimuli). If this comes back near-ceiling too, context_decode_per_stim's
    per-stimulus numbers mostly reflect a global state/gain drift between
    the two separately-optimized training runs (pre model.pt warm-started
    into a FRESH optimizer for the reversal continuation) rather than
    anything specific to how a given cue's response is coded -- see chat.
    Returns a single float; chance = 0.5."""
    ctx_arr = act_dict["context"]
    baseline = act_dict.get("baseline_hidden")
    if baseline is None:
        return float("nan")
    ts, H = baseline.shape[1], baseline.shape[2]
    h_pre = baseline[ctx_arr == 0]
    h_post = baseline[ctx_arr == 1]
    n_min = min(len(h_pre), len(h_post))
    if n_min < 2 or 2 * n_min < n_folds * 2:
        return float("nan")
    rng = np.random.default_rng(random_state)
    h_pre = h_pre[rng.choice(len(h_pre), n_min, replace=False)]
    h_post = h_post[rng.choice(len(h_post), n_min, replace=False)]
    h_all = np.concatenate([h_pre, h_post], axis=0)
    y_trial = np.array([0] * n_min + [1] * n_min)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_accs = []
    for train_idx, test_idx in skf.split(np.arange(2 * n_min), y_trial):
        X_tr, y_tr = h_all[train_idx].mean(1), y_trial[train_idx]
        X_te, y_te = h_all[test_idx].mean(1), y_trial[test_idx]
        fold_accs.append(_fit_score(X_tr, y_tr, X_te, y_te))
    return float(np.mean(fold_accs))
