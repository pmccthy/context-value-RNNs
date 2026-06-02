"""
Decoding and analysis utilities for context-value RNN experiments.
Author: patrick.mccarthy@dpag.ox.ac.uk
"""
import numpy as np
from itertools import combinations

from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.stats import pearsonr, spearmanr


# ── helpers ────────────────────────────────────────────────────────────────

def filter_act_dict(act_dict, mask):
    """Return a copy of act_dict with trial-indexed arrays filtered by boolean mask."""
    n_trials = len(act_dict["context"])
    return {
        k: v[mask] if (isinstance(v, np.ndarray) and
                       v.ndim > 0 and v.shape[0] == n_trials and
                       k != "hidden_states")
           else v
        for k, v in act_dict.items()
    }


def mean_pairs(mat):
    """Mean over unique off-diagonal (i, j) pairs with i < j, ignoring NaN."""
    vals = [mat[i, j] for i, j in combinations(range(mat.shape[0]), 2)
            if not np.isnan(mat[i, j])]
    return np.nanmean(vals) if vals else np.nan


def mean_offdiag(mat):
    """Mean of all off-diagonal entries, ignoring NaN."""
    mask = ~np.eye(mat.shape[0], dtype=bool)
    return np.nanmean(mat[mask])


# ── stimulus identity decoders ─────────────────────────────────────────────

def pairwise_decode(act_dict, period, pooling, n_folds=5, random_state=42):
    """5-fold CV pairwise linear SVM decoder, balanced by downsampling.

    Returns acc of shape (n_contexts, n_stimuli, n_stimuli); diagonal is NaN.

    Args:
        act_dict: Activations dict with keys stim_hidden/reward_hidden, context, stimulus.
        period: "stim" or "reward" — which epoch's hidden states to use.
        pooling: "average" (mean over time) or "pool" (each timestep as a sample).
        n_folds: Number of cross-validation folds.
        random_state: RNG seed.
    """
    ctx_arr  = act_dict["context"]
    stim_arr = act_dict["stimulus"]
    hidden   = act_dict["stim_hidden" if period == "stim" else "reward_hidden"]

    n_stim = int(stim_arr.max()) + 1
    n_ctx  = int(ctx_arr.max()) + 1
    acc    = np.full((n_ctx, n_stim, n_stim), np.nan)

    rng = np.random.default_rng(random_state)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for c in range(n_ctx):
        for si, sj in combinations(range(n_stim), 2):
            h_i = hidden[(ctx_arr == c) & (stim_arr == si)]
            h_j = hidden[(ctx_arr == c) & (stim_arr == sj)]

            n_min = min(len(h_i), len(h_j))
            if n_min < 2 or 2 * n_min < n_folds * 2:
                continue

            h_i = h_i[rng.choice(len(h_i), n_min, replace=False)]
            h_j = h_j[rng.choice(len(h_j), n_min, replace=False)]

            h_all   = np.concatenate([h_i, h_j], axis=0)
            y_trial = np.array([0] * n_min + [1] * n_min)
            ts, H   = h_all.shape[1], h_all.shape[2]

            fold_accs = []
            for train_idx, test_idx in skf.split(np.arange(2 * n_min), y_trial):
                if pooling == "average":
                    X_tr, y_tr = h_all[train_idx].mean(1), y_trial[train_idx]
                    X_te, y_te = h_all[test_idx].mean(1),  y_trial[test_idx]
                else:
                    X_tr = h_all[train_idx].reshape(-1, H)
                    X_te = h_all[test_idx].reshape(-1, H)
                    y_tr = np.repeat(y_trial[train_idx], ts)
                    y_te = np.repeat(y_trial[test_idx],  ts)

                sc  = StandardScaler().fit(X_tr)
                clf = LinearSVC(max_iter=2000, dual="auto").fit(sc.transform(X_tr), y_tr)
                fold_accs.append(clf.score(sc.transform(X_te), y_te))

            a = np.mean(fold_accs)
            acc[c, si, sj] = a
            acc[c, sj, si] = a

    return acc


def crosscontext_decode(act_dict, period, pooling, random_state=42):
    """Train on one context, test on another (no CV needed).

    Returns cross_acc of shape (n_ctx_train, n_ctx_test, n_stim, n_stim);
    same-context entries are NaN.
    """
    ctx_arr  = act_dict["context"]
    stim_arr = act_dict["stimulus"]
    hidden   = act_dict["stim_hidden" if period == "stim" else "reward_hidden"]

    n_stim    = int(stim_arr.max()) + 1
    n_ctx     = int(ctx_arr.max()) + 1
    cross_acc = np.full((n_ctx, n_ctx, n_stim, n_stim), np.nan)

    rng    = np.random.default_rng(random_state)
    ts, H  = hidden.shape[1], hidden.shape[2]

    for c_train in range(n_ctx):
        for c_test in range(n_ctx):
            if c_train == c_test:
                continue
            for si, sj in combinations(range(n_stim), 2):
                tr_i = hidden[(ctx_arr == c_train) & (stim_arr == si)]
                tr_j = hidden[(ctx_arr == c_train) & (stim_arr == sj)]
                te_i = hidden[(ctx_arr == c_test)  & (stim_arr == si)]
                te_j = hidden[(ctx_arr == c_test)  & (stim_arr == sj)]

                if any(len(x) == 0 for x in [tr_i, tr_j, te_i, te_j]):
                    continue

                n_tr = min(len(tr_i), len(tr_j))
                n_te = min(len(te_i), len(te_j))
                tr_i = tr_i[rng.choice(len(tr_i), n_tr, replace=False)]
                tr_j = tr_j[rng.choice(len(tr_j), n_tr, replace=False)]
                te_i = te_i[rng.choice(len(te_i), n_te, replace=False)]
                te_j = te_j[rng.choice(len(te_j), n_te, replace=False)]

                if pooling == "average":
                    X_tr = np.vstack([tr_i.mean(1), tr_j.mean(1)])
                    X_te = np.vstack([te_i.mean(1), te_j.mean(1)])
                    y_tr = np.array([0] * n_tr + [1] * n_tr)
                    y_te = np.array([0] * n_te + [1] * n_te)
                else:
                    X_tr = np.vstack([tr_i.reshape(-1, H), tr_j.reshape(-1, H)])
                    X_te = np.vstack([te_i.reshape(-1, H), te_j.reshape(-1, H)])
                    y_tr = np.array([0] * (n_tr * ts) + [1] * (n_tr * ts))
                    y_te = np.array([0] * (n_te * ts) + [1] * (n_te * ts))

                sc  = StandardScaler().fit(X_tr)
                clf = LinearSVC(max_iter=2000, dual="auto").fit(sc.transform(X_tr), y_tr)
                a   = clf.score(sc.transform(X_te), y_te)
                cross_acc[c_train, c_test, si, sj] = a
                cross_acc[c_train, c_test, sj, si] = a

    return cross_acc


def generalisation_matrix(within_acc, cross_acc):
    """Build n_ctx × n_ctx summary matrix.

    Diagonal entries: mean pairwise within-context accuracy (from within_acc).
    Off-diagonal entries: mean pairwise cross-context accuracy (from cross_acc).
    """
    n_ctx = within_acc.shape[0]
    gm = np.full((n_ctx, n_ctx), np.nan)
    for c in range(n_ctx):
        gm[c, c] = mean_pairs(within_acc[c])
    for ct, ce in combinations(range(n_ctx), 2):
        gm[ct, ce] = mean_pairs(cross_acc[ct, ce])
        gm[ce, ct] = mean_pairs(cross_acc[ce, ct])
    return gm


# ── value decoders ─────────────────────────────────────────────────────────

def value_decode_within(act_dict, period, pooling, value_matrix, n_folds=5, random_state=42):
    """Within-context Ridge regression with k-fold CV.

    Returns Pearson r of shape (n_contexts,).

    Args:
        act_dict: Activations dict.
        period: "stim" or "reward".
        pooling: "average" or "pool".
        value_matrix: (n_stimuli, n_contexts) array of reward probabilities.
        n_folds: Cross-validation folds.
        random_state: RNG seed.
    """
    ctx_arr  = act_dict["context"]
    stim_arr = act_dict["stimulus"]
    hidden   = act_dict["stim_hidden" if period == "stim" else "reward_hidden"]
    ts, H    = hidden.shape[1], hidden.shape[2]

    n_ctx    = int(ctx_arr.max()) + 1
    r_within = np.full(n_ctx, np.nan)
    kf       = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for c in range(n_ctx):
        mask = ctx_arr == c
        y    = value_matrix[stim_arr[mask], c]
        X    = hidden[mask].mean(1) if pooling == "average" else hidden[mask].reshape(-1, H)
        if pooling != "average":
            y = np.repeat(y, ts)
        if len(np.unique(y)) < 2:
            continue

        y_pred = np.full(len(y), np.nan)
        for train_idx, test_idx in kf.split(X):
            sc = StandardScaler().fit(X[train_idx])
            y_pred[test_idx] = Ridge().fit(
                sc.transform(X[train_idx]), y[train_idx]
            ).predict(sc.transform(X[test_idx]))

        r_within[c] = pearsonr(y, y_pred)[0]

    return r_within


def value_decode_cross(act_dict, period, pooling, value_matrix):
    """Cross-context value decoding via Ridge regression.

    Train on one context, test on another.
    Returns Pearson r of shape (n_ctx_train, n_ctx_test); diagonal is NaN.
    """
    ctx_arr  = act_dict["context"]
    stim_arr = act_dict["stimulus"]
    hidden   = act_dict["stim_hidden" if period == "stim" else "reward_hidden"]
    ts, H    = hidden.shape[1], hidden.shape[2]

    n_ctx   = int(ctx_arr.max()) + 1
    r_cross = np.full((n_ctx, n_ctx), np.nan)

    for c_train in range(n_ctx):
        mask_tr = ctx_arr == c_train
        y_tr    = value_matrix[stim_arr[mask_tr], c_train]
        X_tr    = hidden[mask_tr].mean(1) if pooling == "average" else hidden[mask_tr].reshape(-1, H)
        if pooling != "average":
            y_tr = np.repeat(y_tr, ts)
        if len(np.unique(y_tr)) < 2:
            continue

        sc  = StandardScaler().fit(X_tr)
        clf = Ridge().fit(sc.transform(X_tr), y_tr)

        for c_test in range(n_ctx):
            if c_test == c_train:
                continue
            mask_te = ctx_arr == c_test
            y_te    = value_matrix[stim_arr[mask_te], c_test]
            X_te    = hidden[mask_te].mean(1) if pooling == "average" else hidden[mask_te].reshape(-1, H)
            if pooling != "average":
                y_te = np.repeat(y_te, ts)

            y_pred = clf.predict(sc.transform(X_te))
            r_cross[c_train, c_test] = pearsonr(y_te, y_pred)[0]

    return r_cross


def value_gen_matrix(r_within, r_cross):
    """Build n_ctx × n_ctx generalisation matrix for value decoding.

    Diagonal = within-context Pearson r; off-diagonal = cross-context Pearson r.
    """
    gm = r_cross.copy()
    np.fill_diagonal(gm, r_within)
    return gm


# ── binary value decoder (SVM: low vs high) ────────────────────────────────

def binary_value_decode_within(act_dict, period, pooling, value_matrix,
                               threshold=0.25, n_folds=5, random_state=42):
    """Within-context binary (low vs high) value decoding with k-fold CV.

    Stimuli with reward prob ≤ threshold are labelled 0 (low); those with
    reward prob ≥ 1 − threshold are labelled 1 (high). All other stimuli
    (e.g. mid) are excluded.  Uses a LinearSVC.

    Returns accuracy of shape (n_contexts,).
    """
    ctx_arr  = act_dict["context"]
    stim_arr = act_dict["stimulus"]
    hidden   = act_dict["stim_hidden" if period == "stim" else "reward_hidden"]
    ts, H    = hidden.shape[1], hidden.shape[2]
    n_ctx    = int(ctx_arr.max()) + 1

    acc_within = np.full(n_ctx, np.nan)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for c in range(n_ctx):
        low_stim  = np.where(value_matrix[:, c] <= threshold)[0]
        high_stim = np.where(value_matrix[:, c] >= 1 - threshold)[0]

        ctx_mask = ctx_arr == c
        h_ctx    = hidden[ctx_mask]
        s_ctx    = stim_arr[ctx_mask]

        X_low  = h_ctx[np.isin(s_ctx, low_stim)]
        X_high = h_ctx[np.isin(s_ctx, high_stim)]

        if len(X_low) == 0 or len(X_high) == 0:
            continue

        if pooling == "average":
            X_all = np.vstack([X_low.mean(1), X_high.mean(1)])
            y_all = np.array([0] * len(X_low) + [1] * len(X_high))
        else:
            X_all = np.vstack([X_low.reshape(-1, H), X_high.reshape(-1, H)])
            y_all = np.array([0] * (len(X_low) * ts) + [1] * (len(X_high) * ts))

        if len(np.unique(y_all)) < 2 or len(y_all) < n_folds * 2:
            continue

        fold_accs = []
        for train_idx, test_idx in skf.split(X_all, y_all):
            sc  = StandardScaler().fit(X_all[train_idx])
            clf = LinearSVC(max_iter=2000, dual="auto").fit(
                sc.transform(X_all[train_idx]), y_all[train_idx]
            )
            fold_accs.append(clf.score(sc.transform(X_all[test_idx]), y_all[test_idx]))

        acc_within[c] = np.mean(fold_accs)

    return acc_within


def binary_value_decode_cross(act_dict, period, pooling, value_matrix, threshold=0.25):
    """Cross-context binary (low vs high) value decoding via LinearSVC.

    Trains on one context, tests on another.  The low/high labels are
    determined by the reward probability in the *respective* context, so a
    swap stimulus that is 'low' in the training context is correctly labelled
    'high' in the test context.

    Returns accuracy of shape (n_ctx_train, n_ctx_test); diagonal is NaN.
    Chance level = 0.5.
    """
    ctx_arr  = act_dict["context"]
    stim_arr = act_dict["stimulus"]
    hidden   = act_dict["stim_hidden" if period == "stim" else "reward_hidden"]
    ts, H    = hidden.shape[1], hidden.shape[2]
    n_ctx    = int(ctx_arr.max()) + 1

    acc = np.full((n_ctx, n_ctx), np.nan)

    for c_train in range(n_ctx):
        low_tr  = np.where(value_matrix[:, c_train] <= threshold)[0]
        high_tr = np.where(value_matrix[:, c_train] >= 1 - threshold)[0]

        ctx_mask = ctx_arr == c_train
        h_ctx    = hidden[ctx_mask]
        s_ctx    = stim_arr[ctx_mask]

        X_low  = h_ctx[np.isin(s_ctx, low_tr)]
        X_high = h_ctx[np.isin(s_ctx, high_tr)]

        if len(X_low) == 0 or len(X_high) == 0:
            continue

        if pooling == "average":
            X_tr = np.vstack([X_low.mean(1), X_high.mean(1)])
            y_tr = np.array([0] * len(X_low) + [1] * len(X_high))
        else:
            X_tr = np.vstack([X_low.reshape(-1, H), X_high.reshape(-1, H)])
            y_tr = np.array([0] * (len(X_low) * ts) + [1] * (len(X_high) * ts))

        sc  = StandardScaler().fit(X_tr)
        clf = LinearSVC(max_iter=2000, dual="auto").fit(sc.transform(X_tr), y_tr)

        for c_test in range(n_ctx):
            if c_test == c_train:
                continue

            low_te  = np.where(value_matrix[:, c_test] <= threshold)[0]
            high_te = np.where(value_matrix[:, c_test] >= 1 - threshold)[0]

            ctx_mask_te = ctx_arr == c_test
            h_te        = hidden[ctx_mask_te]
            s_te        = stim_arr[ctx_mask_te]

            X_low_te  = h_te[np.isin(s_te, low_te)]
            X_high_te = h_te[np.isin(s_te, high_te)]

            if len(X_low_te) == 0 or len(X_high_te) == 0:
                continue

            if pooling == "average":
                X_te = np.vstack([X_low_te.mean(1), X_high_te.mean(1)])
                y_te = np.array([0] * len(X_low_te) + [1] * len(X_high_te))
            else:
                X_te = np.vstack([X_low_te.reshape(-1, H), X_high_te.reshape(-1, H)])
                y_te = np.array([0] * (len(X_low_te) * ts) + [1] * (len(X_high_te) * ts))

            acc[c_train, c_test] = clf.score(sc.transform(X_te), y_te)

    return acc


# ── unit selectivity ───────────────────────────────────────────────────────

def compute_unit_tuning(activations, period="stim"):
    """Mean activation per unit per (stimulus, context) during the specified period.

    Args:
        activations: Activations dict with stim_hidden/reward_hidden, context, stimulus.
        period: "stim" or "reward".

    Returns:
        tuning: (n_units, n_stimuli, n_contexts) array of mean activations.
    """
    hidden   = activations["stim_hidden" if period == "stim" else "reward_hidden"]
    ctx_arr  = activations["context"]
    stim_arr = activations["stimulus"]

    n_units    = hidden.shape[2]
    n_stimuli  = int(stim_arr.max()) + 1
    n_contexts = int(ctx_arr.max()) + 1

    tuning = np.full((n_units, n_stimuli, n_contexts), np.nan)
    for si in range(n_stimuli):
        for ci in range(n_contexts):
            mask = (stim_arr == si) & (ctx_arr == ci)
            if mask.sum() == 0:
                continue
            tuning[:, si, ci] = hidden[mask].mean(axis=(0, 1))
    return tuning


def selectivity_index(tuning):
    """Selectivity index per unit: 1 − (mean / max) after shifting to non-negative.

    SI = 0 → responds equally to all stimuli (flat tuning curve).
    SI_max = 1 − 1/n_stimuli (e.g. 0.667 for 3 stimuli, 0.75 for 4) when responding
    to exactly one stimulus with all others at the minimum.
    Computed on tuning averaged across contexts.

    Args:
        tuning: (n_units, n_stimuli, n_contexts) array from compute_unit_tuning.

    Returns:
        si: (n_units,) array of SI values in [0, 1].
    """
    mean_ctx = np.nanmean(tuning, axis=2)                              # (n_units, n_stimuli)
    t_min    = np.nanmin(mean_ctx, axis=1, keepdims=True)
    shifted  = mean_ctx - t_min                                        # non-negative
    t_max    = np.nanmax(shifted, axis=1)
    t_mean   = np.nanmean(shifted, axis=1)

    si = np.zeros(tuning.shape[0])
    valid = t_max > 0
    si[valid] = 1.0 - (t_mean[valid] / t_max[valid])
    return si


def selectivity_index_range(tuning, eps=1e-8):
    """Range-over-absolute-sum selectivity index, matching the cogNN convention.

    SI_u = (max_s r_{s,u} − min_s r_{s,u}) / (Σ_s |r_{s,u}| + ε)

    Computed on raw (unshifted) tuning averaged across contexts.
    SI = 0 for a flat tuning curve; SI → 1 for a neuron that responds to one
    stimulus and is fully suppressed by another.  Handles negative activations
    correctly because the denominator uses absolute values.

    Args:
        tuning: (n_units, n_stimuli, n_contexts) array from compute_unit_tuning.

    Returns:
        si: (n_units,) array of SI values in [0, 1].
    """
    mean_ctx = np.nanmean(tuning, axis=2)                              # (n_units, n_stimuli)
    r_max    = np.nanmax(mean_ctx, axis=1)
    r_min    = np.nanmin(mean_ctx, axis=1)
    abs_sum  = np.nansum(np.abs(mean_ctx), axis=1)
    return (r_max - r_min) / (abs_sum + eps)


def selectivity_index_concentration(tuning):
    """Response-concentration selectivity index: max / sum after shifting to non-negative.

    SI_conc = max(shifted) / sum(shifted) for each unit, where shifted = response - min.

    SI_conc = 1/n_stimuli → flat tuning (equal response to all stimuli).
    SI_conc = 1.0         → responds only to one stimulus (all others at minimum).

    Unlike selectivity_index(), the maximum is always 1.0 regardless of n_stimuli,
    making thresholds directly comparable across tasks with different stimulus counts.
    Equivalent to the preferred-stimulus response fraction used in the neuroscience
    literature (e.g. related to Treves & Rolls 1991 sparseness).

    Args:
        tuning: (n_units, n_stimuli, n_contexts) array from compute_unit_tuning.

    Returns:
        si: (n_units,) array of SI values in [1/n_stimuli, 1].
    """
    mean_ctx = np.nanmean(tuning, axis=2)                              # (n_units, n_stimuli)
    t_min    = np.nanmin(mean_ctx, axis=1, keepdims=True)
    shifted  = mean_ctx - t_min                                        # non-negative
    t_max    = np.nanmax(shifted, axis=1)
    t_sum    = np.nansum(shifted, axis=1)

    si = np.full(tuning.shape[0], 1.0 / tuning.shape[1])              # default = flat (1/n)
    valid = t_sum > 0
    si[valid] = t_max[valid] / t_sum[valid]
    return si


def preferred_stim_proportions(tuning, si_threshold=0.1, si_values=None,
                               silent_threshold=None):
    """For each context, count what fraction of selective units prefer each stimulus.

    A unit is selective if its SI ≥ si_threshold AND its max absolute activation
    across stimuli exceeds silent_threshold (if provided).  Preferred stimulus is
    the argmax of the mean-centred tuning curve, so preference reflects which
    stimulus drives the neuron *above its own baseline*.

    Args:
        tuning: (n_units, n_stimuli, n_contexts) from compute_unit_tuning.
        si_threshold: Minimum SI to count a unit as selective.
        si_values: optional (n_units,) pre-computed SI array. If None uses
            selectivity_index() internally.
        silent_threshold: if set, exclude units whose max |activation| across
            stimuli (mean over contexts) is below this value.  Matches the
            cogNN analysis convention (SILENT_THR=1e-4 there).

    Returns:
        props: dict keyed by context index, each with:
            frac_per_stim — (n_stimuli,) fraction of selective neurons preferring
                            each stimulus index (sums to 1 over selective neurons).
            n_selective   — number of selective units.
            n_total       — total number of units.
        si: (n_units,) the SI array used for thresholding.
    """
    si        = selectivity_index(tuning) if si_values is None else si_values
    selective = si >= si_threshold

    if silent_threshold is not None:
        mean_ctx   = np.nanmean(tuning, axis=2)                    # (n_units, n_stim)
        max_abs    = np.nanmax(np.abs(mean_ctx), axis=1)           # (n_units,)
        selective  = selective & (max_abs >= silent_threshold)

    n_units, n_stim, n_contexts = tuning.shape
    n_sel = int(selective.sum())

    props = {}
    for ci in range(n_contexts):
        tc       = tuning[:, :, ci]                                # (n_units, n_stim)
        tc_c     = tc - np.nanmean(tc, axis=1, keepdims=True)     # mean-centre per neuron
        pref_idx = np.nanargmax(tc_c, axis=1)                     # preferred stim index
        sel_pref = pref_idx[selective]
        frac_per_stim = np.array([
            float((sel_pref == si).sum() / n_sel) if n_sel > 0 else np.nan
            for si in range(n_stim)
        ])
        props[ci] = dict(
            frac_per_stim=frac_per_stim,
            n_selective=n_sel,
            n_total=n_units,
        )
    return props, si


def preferred_value_proportions(tuning, value_matrix, si_threshold=0.1,
                                 value_threshold=0.25, si_values=None):
    """Deprecated: use preferred_stim_proportions instead.

    Kept for backward compatibility with sweep scripts that write CSVs.
    Bins selective neurons by the value of their preferred stimulus using a
    threshold — unnecessary when stimulus values are known exactly.
    """
    si        = selectivity_index(tuning) if si_values is None else si_values
    selective = si >= si_threshold
    n_units, _, n_contexts = tuning.shape

    props = {}
    for ci in range(n_contexts):
        pref_stim = np.nanargmax(tuning[:, :, ci], axis=1)
        pref_val  = value_matrix[pref_stim, ci]
        sel_vals  = pref_val[selective]
        n_sel     = int(selective.sum())

        def _frac(mask):
            return float(mask.sum() / n_sel) if n_sel > 0 else np.nan

        props[ci] = dict(
            frac_low=_frac(sel_vals <= value_threshold),
            frac_mid=_frac((sel_vals > value_threshold) & (sel_vals < 1 - value_threshold)),
            frac_high=_frac(sel_vals >= 1 - value_threshold),
            n_selective=n_sel,
            n_total=n_units,
        )
    return props, si


# ── time-resolved decoding ─────────────────────────────────────────────────

def time_resolved_decode(activations, value_matrix, n_iti_pre=3,
                          value_threshold=0.25, n_folds=5, random_state=42):
    """Binary (low vs high) value decoding at each aligned timestep within a trial.

    Aligns each trial as: last n_iti_pre ITI timesteps + stim window + reward window.
    At each position runs within-context k-fold CV and cross-context (train/test on
    different contexts) binary LinearSVC.

    Args:
        activations: Dict with hidden_states (T, H), context, stimulus, trial_structure.
        value_matrix: (n_stimuli, n_contexts) reward probabilities.
        n_iti_pre: Number of ITI timesteps to prepend to each trial window.
        value_threshold: Boundary for low/high labels (low ≤ thr, high ≥ 1−thr).
        n_folds: CV folds for within-context decode.
        random_state: RNG seed.

    Returns:
        acc_within: (n_contexts, n_timesteps) within-context accuracy.
        acc_cross: (n_contexts, n_contexts, n_timesteps) cross-context accuracy;
                   diagonal is NaN.
        epoch_boundaries: dict with stim_onset, reward_onset, total.
    """
    hidden_seq   = activations["hidden_states"]
    ctx_arr      = activations["context"]
    stim_arr     = activations["stimulus"]
    trial_struct = activations["trial_structure"]

    n_units    = hidden_seq.shape[1]
    n_contexts = int(ctx_arr.max()) + 1
    n_trials   = len(trial_struct)

    stim_ts = (trial_struct[0]["stim_window"][1]
               - trial_struct[0]["stim_window"][0])
    rew_ts  = (trial_struct[0]["reward_window"][1]
               - trial_struct[0]["reward_window"][0])
    n_total_ts = n_iti_pre + stim_ts + rew_ts

    # Build aligned hidden states: (n_trials, n_total_ts, n_units)
    aligned_h = np.full((n_trials, n_total_ts, n_units), np.nan)
    for i, trial in enumerate(trial_struct):
        iti_e            = trial["stim_window"][0]
        stim_s, stim_e   = trial["stim_window"]
        rew_s,  rew_e    = trial["reward_window"]

        iti_slice = hidden_seq[max(0, iti_e - n_iti_pre):iti_e]
        iti_padded = np.full((n_iti_pre, n_units), np.nan)
        iti_padded[-len(iti_slice):] = iti_slice

        aligned_h[i, :n_iti_pre]                     = iti_padded
        aligned_h[i, n_iti_pre:n_iti_pre + stim_ts]  = hidden_seq[stim_s:stim_e]
        aligned_h[i, n_iti_pre + stim_ts:]            = hidden_seq[rew_s:rew_e]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    acc_within = np.full((n_contexts, n_total_ts), np.nan)
    acc_cross  = np.full((n_contexts, n_contexts, n_total_ts), np.nan)

    for t in range(n_total_ts):
        X_t = aligned_h[:, t, :]                          # (n_trials, n_units)
        valid = ~np.isnan(X_t).any(axis=1)

        # within-context
        for c in range(n_contexts):
            low_s  = np.where(value_matrix[:, c] <= value_threshold)[0]
            high_s = np.where(value_matrix[:, c] >= 1 - value_threshold)[0]
            mask   = (ctx_arr == c) & valid
            X_c, s_c = X_t[mask], stim_arr[mask]

            X_lo = X_c[np.isin(s_c, low_s)]
            X_hi = X_c[np.isin(s_c, high_s)]
            if len(X_lo) == 0 or len(X_hi) == 0:
                continue

            X = np.vstack([X_lo, X_hi])
            y = np.array([0] * len(X_lo) + [1] * len(X_hi))
            if len(np.unique(y)) < 2 or len(y) < n_folds * 2:
                continue

            fold_accs = []
            for tr_idx, te_idx in skf.split(X, y):
                sc  = StandardScaler().fit(X[tr_idx])
                clf = LinearSVC(max_iter=2000, dual="auto").fit(
                    sc.transform(X[tr_idx]), y[tr_idx]
                )
                fold_accs.append(clf.score(sc.transform(X[te_idx]), y[te_idx]))
            if fold_accs:
                acc_within[c, t] = np.mean(fold_accs)

        # cross-context
        for c_train in range(n_contexts):
            for c_test in range(n_contexts):
                if c_train == c_test:
                    continue
                lo_tr  = np.where(value_matrix[:, c_train] <= value_threshold)[0]
                hi_tr  = np.where(value_matrix[:, c_train] >= 1 - value_threshold)[0]
                lo_te  = np.where(value_matrix[:, c_test]  <= value_threshold)[0]
                hi_te  = np.where(value_matrix[:, c_test]  >= 1 - value_threshold)[0]

                m_tr = (ctx_arr == c_train) & valid
                m_te = (ctx_arr == c_test)  & valid
                X_tr, s_tr = X_t[m_tr], stim_arr[m_tr]
                X_te, s_te = X_t[m_te], stim_arr[m_te]

                Xlo_tr = X_tr[np.isin(s_tr, lo_tr)]
                Xhi_tr = X_tr[np.isin(s_tr, hi_tr)]
                Xlo_te = X_te[np.isin(s_te, lo_te)]
                Xhi_te = X_te[np.isin(s_te, hi_te)]

                if any(len(x) == 0 for x in [Xlo_tr, Xhi_tr, Xlo_te, Xhi_te]):
                    continue

                X_tr_all = np.vstack([Xlo_tr, Xhi_tr])
                y_tr_all = np.array([0] * len(Xlo_tr) + [1] * len(Xhi_tr))
                X_te_all = np.vstack([Xlo_te, Xhi_te])
                y_te_all = np.array([0] * len(Xlo_te) + [1] * len(Xhi_te))

                sc  = StandardScaler().fit(X_tr_all)
                clf = LinearSVC(max_iter=2000, dual="auto").fit(
                    sc.transform(X_tr_all), y_tr_all
                )
                acc_cross[c_train, c_test, t] = clf.score(
                    sc.transform(X_te_all), y_te_all
                )

    epoch_boundaries = dict(
        stim_onset=n_iti_pre,
        reward_onset=n_iti_pre + stim_ts,
        total=n_total_ts,
    )
    return acc_within, acc_cross, epoch_boundaries


# ── identity-partialled decoding ───────────────────────────────────────────

def _partial_out_identity(X_tr, X_te, stim_tr, stim_te, n_stimuli):
    """OLS identity regression fit on train; subtracted from both train and test."""
    id_tr = np.eye(n_stimuli)[stim_tr]
    id_te = np.eye(n_stimuli)[stim_te]
    coef, _, _, _ = np.linalg.lstsq(id_tr, X_tr, rcond=None)  # (n_stimuli, n_units)
    return X_tr - id_tr @ coef, X_te - id_te @ coef


def value_decode_within_partialled(act_dict, period, pooling, value_matrix,
                                    n_folds=5, random_state=42):
    """Within-context Ridge on identity-partialled hidden states.

    Stimulus identity (one-hot) is regressed from hidden states within each
    CV fold before fitting Ridge.  Returns Pearson r of shape (n_contexts,).
    """
    ctx_arr   = act_dict["context"]
    stim_arr  = act_dict["stimulus"]
    hidden    = act_dict["stim_hidden" if period == "stim" else "reward_hidden"]
    ts, H     = hidden.shape[1], hidden.shape[2]
    n_stimuli = int(stim_arr.max()) + 1
    n_ctx     = int(ctx_arr.max()) + 1
    r_within  = np.full(n_ctx, np.nan)
    kf        = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for c in range(n_ctx):
        mask = ctx_arr == c
        y    = value_matrix[stim_arr[mask], c]
        X    = hidden[mask].mean(1) if pooling == "average" else hidden[mask].reshape(-1, H)
        s    = stim_arr[mask] if pooling == "average" else np.repeat(stim_arr[mask], ts)
        if pooling != "average":
            y = np.repeat(y, ts)
        if len(np.unique(y)) < 2:
            continue

        y_pred = np.full(len(y), np.nan)
        for train_idx, test_idx in kf.split(X):
            X_tr_r, X_te_r = _partial_out_identity(
                X[train_idx], X[test_idx], s[train_idx], s[test_idx], n_stimuli
            )
            sc = StandardScaler().fit(X_tr_r)
            y_pred[test_idx] = Ridge().fit(
                sc.transform(X_tr_r), y[train_idx]
            ).predict(sc.transform(X_te_r))

        r_within[c] = pearsonr(y, y_pred)[0]

    return r_within


def value_decode_cross_partialled(act_dict, period, pooling, value_matrix):
    """Cross-context Ridge on identity-partialled hidden states.

    Identity OLS is fit on the training context and applied to both contexts.
    Returns Pearson r of shape (n_ctx_train, n_ctx_test); diagonal is NaN.
    """
    ctx_arr   = act_dict["context"]
    stim_arr  = act_dict["stimulus"]
    hidden    = act_dict["stim_hidden" if period == "stim" else "reward_hidden"]
    ts, H     = hidden.shape[1], hidden.shape[2]
    n_stimuli = int(stim_arr.max()) + 1
    n_ctx     = int(ctx_arr.max()) + 1
    r_cross   = np.full((n_ctx, n_ctx), np.nan)

    for c_train in range(n_ctx):
        mask_tr = ctx_arr == c_train
        y_tr    = value_matrix[stim_arr[mask_tr], c_train]
        X_tr    = hidden[mask_tr].mean(1) if pooling == "average" else hidden[mask_tr].reshape(-1, H)
        s_tr    = stim_arr[mask_tr] if pooling == "average" else np.repeat(stim_arr[mask_tr], ts)
        if pooling != "average":
            y_tr = np.repeat(y_tr, ts)
        if len(np.unique(y_tr)) < 2:
            continue

        for c_test in range(n_ctx):
            if c_test == c_train:
                continue
            mask_te = ctx_arr == c_test
            y_te    = value_matrix[stim_arr[mask_te], c_test]
            X_te    = hidden[mask_te].mean(1) if pooling == "average" else hidden[mask_te].reshape(-1, H)
            s_te    = stim_arr[mask_te] if pooling == "average" else np.repeat(stim_arr[mask_te], ts)
            if pooling != "average":
                y_te = np.repeat(y_te, ts)

            X_tr_r, X_te_r = _partial_out_identity(X_tr, X_te, s_tr, s_te, n_stimuli)
            sc     = StandardScaler().fit(X_tr_r)
            y_pred = Ridge().fit(sc.transform(X_tr_r), y_tr).predict(sc.transform(X_te_r))
            r_cross[c_train, c_test] = pearsonr(y_te, y_pred)[0]

    return r_cross


def binary_value_decode_within_partialled(act_dict, period, pooling, value_matrix,
                                           threshold=0.25, n_folds=5, random_state=42):
    """Within-context binary SVM on identity-partialled hidden states.

    Returns accuracy of shape (n_contexts,).
    """
    ctx_arr   = act_dict["context"]
    stim_arr  = act_dict["stimulus"]
    hidden    = act_dict["stim_hidden" if period == "stim" else "reward_hidden"]
    ts, H     = hidden.shape[1], hidden.shape[2]
    n_stimuli = int(stim_arr.max()) + 1
    n_ctx     = int(ctx_arr.max()) + 1
    acc_within = np.full(n_ctx, np.nan)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for c in range(n_ctx):
        low_stim  = np.where(value_matrix[:, c] <= threshold)[0]
        high_stim = np.where(value_matrix[:, c] >= 1 - threshold)[0]

        ctx_mask  = ctx_arr == c
        h_ctx     = hidden[ctx_mask]
        s_ctx     = stim_arr[ctx_mask]
        mask_lo   = np.isin(s_ctx, low_stim)
        mask_hi   = np.isin(s_ctx, high_stim)

        X_low, X_high = h_ctx[mask_lo], h_ctx[mask_hi]
        s_low, s_high = s_ctx[mask_lo], s_ctx[mask_hi]
        if len(X_low) == 0 or len(X_high) == 0:
            continue

        if pooling == "average":
            X_all = np.vstack([X_low.mean(1), X_high.mean(1)])
            s_all = np.concatenate([s_low, s_high])
            y_all = np.array([0] * len(X_low) + [1] * len(X_high))
        else:
            X_all = np.vstack([X_low.reshape(-1, H), X_high.reshape(-1, H)])
            s_all = np.concatenate([np.repeat(s_low, ts), np.repeat(s_high, ts)])
            y_all = np.array([0] * (len(X_low) * ts) + [1] * (len(X_high) * ts))

        if len(np.unique(y_all)) < 2 or len(y_all) < n_folds * 2:
            continue

        fold_accs = []
        for train_idx, test_idx in skf.split(X_all, y_all):
            X_tr_r, X_te_r = _partial_out_identity(
                X_all[train_idx], X_all[test_idx],
                s_all[train_idx], s_all[test_idx], n_stimuli
            )
            sc  = StandardScaler().fit(X_tr_r)
            clf = LinearSVC(max_iter=2000, dual="auto").fit(
                sc.transform(X_tr_r), y_all[train_idx]
            )
            fold_accs.append(clf.score(sc.transform(X_te_r), y_all[test_idx]))

        acc_within[c] = np.mean(fold_accs)

    return acc_within


def binary_value_decode_cross_partialled(act_dict, period, pooling, value_matrix,
                                          threshold=0.25):
    """Cross-context binary SVM on identity-partialled hidden states.

    Returns accuracy of shape (n_ctx_train, n_ctx_test); diagonal is NaN.
    """
    ctx_arr   = act_dict["context"]
    stim_arr  = act_dict["stimulus"]
    hidden    = act_dict["stim_hidden" if period == "stim" else "reward_hidden"]
    ts, H     = hidden.shape[1], hidden.shape[2]
    n_stimuli = int(stim_arr.max()) + 1
    n_ctx     = int(ctx_arr.max()) + 1
    acc       = np.full((n_ctx, n_ctx), np.nan)

    for c_train in range(n_ctx):
        low_tr  = np.where(value_matrix[:, c_train] <= threshold)[0]
        high_tr = np.where(value_matrix[:, c_train] >= 1 - threshold)[0]

        ctx_mask  = ctx_arr == c_train
        h_ctx     = hidden[ctx_mask]
        s_ctx     = stim_arr[ctx_mask]
        mask_lo   = np.isin(s_ctx, low_tr)
        mask_hi   = np.isin(s_ctx, high_tr)

        X_low, X_high = h_ctx[mask_lo], h_ctx[mask_hi]
        s_low, s_high = s_ctx[mask_lo], s_ctx[mask_hi]
        if len(X_low) == 0 or len(X_high) == 0:
            continue

        if pooling == "average":
            X_tr = np.vstack([X_low.mean(1), X_high.mean(1)])
            s_tr = np.concatenate([s_low, s_high])
            y_tr = np.array([0] * len(X_low) + [1] * len(X_high))
        else:
            X_tr = np.vstack([X_low.reshape(-1, H), X_high.reshape(-1, H)])
            s_tr = np.concatenate([np.repeat(s_low, ts), np.repeat(s_high, ts)])
            y_tr = np.array([0] * (len(X_low) * ts) + [1] * (len(X_high) * ts))

        for c_test in range(n_ctx):
            if c_test == c_train:
                continue
            low_te  = np.where(value_matrix[:, c_test] <= threshold)[0]
            high_te = np.where(value_matrix[:, c_test] >= 1 - threshold)[0]

            ctx_mask_te   = ctx_arr == c_test
            h_te          = hidden[ctx_mask_te]
            s_te_arr      = stim_arr[ctx_mask_te]
            mask_lo_te    = np.isin(s_te_arr, low_te)
            mask_hi_te    = np.isin(s_te_arr, high_te)

            X_low_te, X_high_te = h_te[mask_lo_te], h_te[mask_hi_te]
            s_low_te, s_high_te = s_te_arr[mask_lo_te], s_te_arr[mask_hi_te]
            if len(X_low_te) == 0 or len(X_high_te) == 0:
                continue

            if pooling == "average":
                X_te = np.vstack([X_low_te.mean(1), X_high_te.mean(1)])
                s_te = np.concatenate([s_low_te, s_high_te])
                y_te = np.array([0] * len(X_low_te) + [1] * len(X_high_te))
            else:
                X_te = np.vstack([X_low_te.reshape(-1, H), X_high_te.reshape(-1, H)])
                s_te = np.concatenate([np.repeat(s_low_te, ts), np.repeat(s_high_te, ts)])
                y_te = np.array([0] * (len(X_low_te) * ts) + [1] * (len(X_high_te) * ts))

            X_tr_r, X_te_r = _partial_out_identity(X_tr, X_te, s_tr, s_te, n_stimuli)
            sc  = StandardScaler().fit(X_tr_r)
            clf = LinearSVC(max_iter=2000, dual="auto").fit(sc.transform(X_tr_r), y_tr)
            acc[c_train, c_test] = clf.score(sc.transform(X_te_r), y_te)

    return acc


# ── RSA ────────────────────────────────────────────────────────────────────

def compute_neural_rdm(act_dict, period, n_iti_pre=3):
    """Compute neural RDM over (stimulus, context) conditions.

    Mean hidden state per condition is the average across all trials and
    timesteps. Dissimilarity = 1 − Pearson correlation.

    Args:
        period: "stim", "reward", or "iti". For "iti", the last n_iti_pre
                timesteps of each trial's ITI window are used; requires
                act_dict to contain "hidden_states" and "trial_structure".
        n_iti_pre: ITI timesteps to use (only relevant when period="iti").

    Returns:
        rdm: (n_cond, n_cond) dissimilarity matrix
        conditions: list of (stim_idx, ctx_idx) tuples, ordered ctx-major
    """
    ctx_arr  = act_dict["context"]
    stim_arr = act_dict["stimulus"]

    if period == "iti":
        hidden_seq   = act_dict["hidden_states"]   # (T, n_units)
        trial_struct = act_dict["trial_structure"]
        n_units = hidden_seq.shape[1]
        iti_slices = []
        for trial in trial_struct:
            iti_s, iti_e = trial["iti_window"]
            sl  = hidden_seq[max(0, iti_e - n_iti_pre):iti_e]
            pad = np.full((n_iti_pre, n_units), np.nan)
            pad[-len(sl):] = sl
            iti_slices.append(pad)
        hidden = np.stack(iti_slices)              # (n_trials, n_iti_pre, n_units)
    else:
        hidden = act_dict["stim_hidden" if period == "stim" else "reward_hidden"]

    n_stim = int(stim_arr.max()) + 1
    n_ctx  = int(ctx_arr.max()) + 1

    conditions = [(si, ci) for ci in range(n_ctx) for si in range(n_stim)]
    n_cond     = len(conditions)
    mean_h     = np.full((n_cond, hidden.shape[2]), np.nan)

    for k, (si, ci) in enumerate(conditions):
        mask = (stim_arr == si) & (ctx_arr == ci)
        if mask.sum() == 0:
            continue
        mean_h[k] = np.nanmean(hidden[mask], axis=(0, 1))

    rdm = np.full((n_cond, n_cond), np.nan)
    for i in range(n_cond):
        for j in range(n_cond):
            if np.isnan(mean_h[i]).any() or np.isnan(mean_h[j]).any():
                continue
            rdm[i, j] = 1.0 - pearsonr(mean_h[i], mean_h[j])[0]

    return rdm, conditions


def compute_model_rdms(conditions, value_matrix):
    """Compute identity and value model RDMs for a set of (stim, ctx) conditions.

    Returns:
        identity_rdm: 0 if same stimulus regardless of context, 1 if different
        value_rdm: |value_i − value_j| for each condition pair
    """
    n_cond  = len(conditions)
    id_rdm  = np.zeros((n_cond, n_cond))
    val_rdm = np.zeros((n_cond, n_cond))

    for i, (si, ci) in enumerate(conditions):
        for j, (sj, cj) in enumerate(conditions):
            id_rdm[i, j]  = 0.0 if si == sj else 1.0
            val_rdm[i, j] = abs(float(value_matrix[si, ci]) - float(value_matrix[sj, cj]))

    return id_rdm, val_rdm


def rsa_compare(neural_rdm, model_rdms):
    """Spearman correlation between neural RDM and each model RDM (upper triangle).

    Args:
        neural_rdm: (n_cond, n_cond) neural dissimilarity matrix
        model_rdms: dict mapping name → (n_cond, n_cond) model RDM

    Returns:
        dict mapping name → Spearman r (NaN if too few valid pairs)
    """
    n        = neural_rdm.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    neural_v = neural_rdm[triu_idx]

    results = {}
    for name, rdm in model_rdms.items():
        model_v = rdm[triu_idx]
        valid   = ~(np.isnan(neural_v) | np.isnan(model_v))
        results[name] = spearmanr(neural_v[valid], model_v[valid])[0] if valid.sum() >= 3 else np.nan

    return results


# ── held-out stimulus value decoding ──────────────────────────────────────

def held_out_stim_decode(act_dict, period, pooling, value_matrix,
                          train_stim_idx, test_stim_idx):
    """Train Ridge on train_stim_idx stimuli, test on test_stim_idx stimuli.

    For each training context, fits Ridge on the train stimuli, then evaluates
    on the test stimuli in every context (including the same context).

    Returns:
        r_matrix: (n_ctx_train, n_ctx_test) Pearson r; NaN where data is missing
    """
    ctx_arr  = act_dict["context"]
    stim_arr = act_dict["stimulus"]
    hidden   = act_dict["stim_hidden" if period == "stim" else "reward_hidden"]
    ts, H    = hidden.shape[1], hidden.shape[2]
    n_ctx    = int(ctx_arr.max()) + 1
    r_matrix = np.full((n_ctx, n_ctx), np.nan)

    train_mask_stim = np.isin(stim_arr, train_stim_idx)
    test_mask_stim  = np.isin(stim_arr, test_stim_idx)

    for c_train in range(n_ctx):
        mask_tr = (ctx_arr == c_train) & train_mask_stim
        if mask_tr.sum() == 0:
            continue
        y_tr = value_matrix[stim_arr[mask_tr], c_train]
        X_tr = hidden[mask_tr].mean(1) if pooling == "average" else hidden[mask_tr].reshape(-1, H)
        if pooling != "average":
            y_tr = np.repeat(y_tr, ts)
        if len(np.unique(y_tr)) < 2:
            continue

        sc  = StandardScaler().fit(X_tr)
        clf = Ridge().fit(sc.transform(X_tr), y_tr)

        for c_test in range(n_ctx):
            mask_te = (ctx_arr == c_test) & test_mask_stim
            if mask_te.sum() == 0:
                continue
            y_te = value_matrix[stim_arr[mask_te], c_test]
            X_te = hidden[mask_te].mean(1) if pooling == "average" else hidden[mask_te].reshape(-1, H)
            if pooling != "average":
                y_te = np.repeat(y_te, ts)
            if len(np.unique(y_te)) < 2:
                continue

            y_pred = clf.predict(sc.transform(X_te))
            r_matrix[c_train, c_test] = pearsonr(y_te, y_pred)[0]

    return r_matrix


# ── cross-group decoding and decoder axis comparison ───────────────────────

def cross_group_decode(act_dict, period, pooling, value_matrix,
                        train_groups, test_groups, n_folds=5, random_state=42):
    """Ridge decoding across all combinations of train/test stimulus groups and contexts.

    For cross-context pairs (c_train ≠ c_test), trains on all train-group trials
    in c_train and tests on test-group trials in c_test.  For within-context
    (c_train = c_test), uses n_folds-fold CV: each fold holds out a subset of
    test-group trials from the training set so the diagonal is unbiased.

    Args:
        train_groups: dict mapping name → array-like of stim indices, or None for all stimuli
        test_groups:  dict mapping name → array-like of stim indices, or None for all stimuli
        n_folds: Number of CV folds for within-context diagonal.
        random_state: RNG seed.

    Returns:
        results: dict keyed by (train_name, test_name) → (n_ctx, n_ctx) Pearson r matrix
    """
    ctx_arr  = act_dict["context"]
    stim_arr = act_dict["stimulus"]
    hidden   = act_dict["stim_hidden" if period == "stim" else "reward_hidden"]
    ts, H    = hidden.shape[1], hidden.shape[2]
    n_ctx    = int(ctx_arr.max()) + 1

    def _stim_mask(idx):
        return np.ones(len(stim_arr), dtype=bool) if idx is None else np.isin(stim_arr, idx)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    results = {}
    for tr_name, tr_idx in train_groups.items():
        tr_mask_stim = _stim_mask(tr_idx)
        for te_name, te_idx in test_groups.items():
            te_mask_stim = _stim_mask(te_idx)
            r_mat = np.full((n_ctx, n_ctx), np.nan)

            for c_train in range(n_ctx):
                # Fit decoder on all tr_mask_stim trials in c_train (used for cross-context)
                mask_tr_all = (ctx_arr == c_train) & tr_mask_stim
                sc_cross = clf_cross = None
                if mask_tr_all.sum() > 0:
                    y_tr_c = value_matrix[stim_arr[mask_tr_all], c_train]
                    X_tr_c = (hidden[mask_tr_all].mean(1) if pooling == "average"
                              else hidden[mask_tr_all].reshape(-1, H))
                    if pooling != "average":
                        y_tr_c = np.repeat(y_tr_c, ts)
                    if len(np.unique(y_tr_c)) >= 2:
                        sc_cross  = StandardScaler().fit(X_tr_c)
                        clf_cross = Ridge().fit(sc_cross.transform(X_tr_c), y_tr_c)

                for c_test in range(n_ctx):
                    if c_train != c_test:
                        # Cross-context: no CV needed
                        if clf_cross is None:
                            continue
                        mask_te = (ctx_arr == c_test) & te_mask_stim
                        if mask_te.sum() == 0:
                            continue
                        y_te = value_matrix[stim_arr[mask_te], c_test]
                        X_te = (hidden[mask_te].mean(1) if pooling == "average"
                                else hidden[mask_te].reshape(-1, H))
                        if pooling != "average":
                            y_te = np.repeat(y_te, ts)
                        if len(np.unique(y_te)) < 2:
                            continue
                        y_pred = clf_cross.predict(sc_cross.transform(X_te))
                        r_mat[c_train, c_test] = pearsonr(y_te, y_pred)[0]

                    else:
                        # Within-context: k-fold CV holding out test-group trials
                        te_global = np.where((ctx_arr == c_train) & te_mask_stim)[0]
                        tr_global = np.where((ctx_arr == c_train) & tr_mask_stim)[0]
                        if len(te_global) < n_folds or len(tr_global) == 0:
                            continue

                        y_te_flat = value_matrix[stim_arr[te_global], c_train]
                        if pooling != "average":
                            y_te_flat = np.repeat(y_te_flat, ts)
                        if len(np.unique(y_te_flat)) < 2:
                            continue

                        y_pred_flat = np.full(len(y_te_flat), np.nan)
                        for _, fold_te_local in kf.split(te_global):
                            fold_te_global = te_global[fold_te_local]
                            fold_tr_global = np.setdiff1d(tr_global, fold_te_global)
                            if len(fold_tr_global) == 0:
                                continue

                            y_tr_f = value_matrix[stim_arr[fold_tr_global], c_train]
                            if pooling == "average":
                                X_tr_f = hidden[fold_tr_global].mean(1)
                                X_te_f = hidden[fold_te_global].mean(1)
                                fold_pos = fold_te_local
                            else:
                                X_tr_f = hidden[fold_tr_global].reshape(-1, H)
                                y_tr_f = np.repeat(y_tr_f, ts)
                                X_te_f = hidden[fold_te_global].reshape(-1, H)
                                fold_pos = np.concatenate([
                                    np.arange(i * ts, (i + 1) * ts) for i in fold_te_local
                                ])
                            if len(np.unique(y_tr_f)) < 2:
                                continue

                            sc_f  = StandardScaler().fit(X_tr_f)
                            clf_f = Ridge().fit(sc_f.transform(X_tr_f), y_tr_f)
                            y_pred_flat[fold_pos] = clf_f.predict(sc_f.transform(X_te_f))

                        valid = ~np.isnan(y_pred_flat)
                        if valid.sum() >= 2 and len(np.unique(y_te_flat[valid])) >= 2:
                            r_mat[c_train, c_test] = pearsonr(
                                y_te_flat[valid], y_pred_flat[valid]
                            )[0]

            results[(tr_name, te_name)] = r_mat

    return results


def cross_group_binary_decode(act_dict, period, pooling, value_matrix,
                              train_groups, test_groups, threshold=0.25,
                              n_folds=5, random_state=42):
    """Binary (low vs high) SVM decoding across train/test stimulus groups and contexts.

    Cross-context pairs use a single fit; within-context diagonal uses n_folds-fold CV
    with test-group trials held out of training in each fold.

    Args:
        train_groups: dict name → stim index array or None (all stimuli)
        test_groups:  dict name → stim index array or None (all stimuli)
        threshold: reward prob boundary; low ≤ threshold, high ≥ 1−threshold
        n_folds: CV folds for within-context diagonal
        random_state: RNG seed

    Returns:
        results: dict keyed by (train_name, test_name) → (n_ctx, n_ctx) accuracy matrix
    """
    ctx_arr  = act_dict["context"]
    stim_arr = act_dict["stimulus"]
    hidden   = act_dict["stim_hidden" if period == "stim" else "reward_hidden"]
    ts, H    = hidden.shape[1], hidden.shape[2]
    n_ctx    = int(ctx_arr.max()) + 1

    def _lo_hi_global(c, stim_idx):
        lo_stim = np.where(value_matrix[:, c] <= threshold)[0]
        hi_stim = np.where(value_matrix[:, c] >= 1 - threshold)[0]
        if stim_idx is not None:
            stim_idx = np.asarray(stim_idx)
            lo_stim  = np.intersect1d(lo_stim, stim_idx)
            hi_stim  = np.intersect1d(hi_stim, stim_idx)
        ctx_mask = ctx_arr == c
        return (np.where(ctx_mask & np.isin(stim_arr, lo_stim))[0],
                np.where(ctx_mask & np.isin(stim_arr, hi_stim))[0])

    def _Xy(lo_g, hi_g):
        h_lo, h_hi = hidden[lo_g], hidden[hi_g]
        if pooling == "average":
            X = np.vstack([h_lo.mean(1), h_hi.mean(1)])
            y = np.array([0] * len(lo_g) + [1] * len(hi_g))
        else:
            X = np.vstack([h_lo.reshape(-1, H), h_hi.reshape(-1, H)])
            y = np.array([0] * (len(lo_g) * ts) + [1] * (len(hi_g) * ts))
        return X, y

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    results = {}
    for tr_name, tr_idx in train_groups.items():
        for te_name, te_idx in test_groups.items():
            acc_mat = np.full((n_ctx, n_ctx), np.nan)

            for c_train in range(n_ctx):
                tr_lo_g, tr_hi_g = _lo_hi_global(c_train, tr_idx)

                sc_cross = clf_cross = None
                if len(tr_lo_g) > 0 and len(tr_hi_g) > 0:
                    X_tr_c, y_tr_c = _Xy(tr_lo_g, tr_hi_g)
                    if len(np.unique(y_tr_c)) == 2:
                        sc_cross  = StandardScaler().fit(X_tr_c)
                        clf_cross = LinearSVC(max_iter=2000, dual="auto").fit(
                            sc_cross.transform(X_tr_c), y_tr_c
                        )

                for c_test in range(n_ctx):
                    te_lo_g, te_hi_g = _lo_hi_global(c_test, te_idx)
                    if len(te_lo_g) == 0 or len(te_hi_g) == 0:
                        continue

                    if c_train != c_test:
                        if clf_cross is None:
                            continue
                        X_te, y_te = _Xy(te_lo_g, te_hi_g)
                        acc_mat[c_train, c_test] = clf_cross.score(sc_cross.transform(X_te), y_te)

                    else:
                        # Within-context: k-fold CV holding out test-group trials
                        te_global    = np.concatenate([te_lo_g, te_hi_g])
                        te_labels    = np.array([0] * len(te_lo_g) + [1] * len(te_hi_g))
                        tr_global    = np.concatenate([tr_lo_g, tr_hi_g])
                        tr_lo_set    = set(tr_lo_g.tolist())

                        if len(te_global) < n_folds or len(np.unique(te_labels)) < 2:
                            continue

                        n_flat    = len(te_global) if pooling == "average" else len(te_global) * ts
                        y_pred    = np.full(n_flat, np.nan)

                        for _, fold_te_local in kf.split(te_global):
                            fold_te_g      = te_global[fold_te_local]
                            fold_tr_g      = np.setdiff1d(tr_global, fold_te_g)
                            if len(fold_tr_g) == 0:
                                continue
                            fold_tr_labels = np.array(
                                [0 if t in tr_lo_set else 1 for t in fold_tr_g]
                            )
                            if len(np.unique(fold_tr_labels)) < 2:
                                continue

                            if pooling == "average":
                                X_tr_f = hidden[fold_tr_g].mean(1)
                                X_te_f = hidden[fold_te_g].mean(1)
                                fold_pos = fold_te_local
                            else:
                                X_tr_f   = hidden[fold_tr_g].reshape(-1, H)
                                X_te_f   = hidden[fold_te_g].reshape(-1, H)
                                fold_pos = np.concatenate([
                                    np.arange(i * ts, (i + 1) * ts) for i in fold_te_local
                                ])
                                fold_tr_labels = np.repeat(fold_tr_labels, ts)

                            sc_f  = StandardScaler().fit(X_tr_f)
                            clf_f = LinearSVC(max_iter=2000, dual="auto").fit(
                                sc_f.transform(X_tr_f), fold_tr_labels
                            )
                            y_pred[fold_pos] = clf_f.predict(sc_f.transform(X_te_f))

                        y_true = te_labels if pooling == "average" else np.repeat(te_labels, ts)
                        valid  = ~np.isnan(y_pred)
                        if valid.sum() >= 2:
                            acc_mat[c_train, c_test] = (y_pred[valid] == y_true[valid]).mean()

            results[(tr_name, te_name)] = acc_mat

    return results


def decoder_axis_similarity(act_dict, period, pooling, value_matrix,
                              stim_idx_a, stim_idx_b):
    """Cosine similarity between Ridge weight vectors for two stimulus subsets.

    Fits a common StandardScaler on all trials in each context, then fits
    separate Ridge decoders for stim_idx_a and stim_idx_b using that shared
    scaler.  This puts both weight vectors in the same coordinate system so
    cosine similarity is meaningful.

    Args:
        stim_idx_a: array-like of stim indices for first decoder, or None for all
        stim_idx_b: array-like of stim indices for second decoder, or None for all

    Returns:
        cos_sim:   (n_contexts,) cosine similarity in [−1, 1]
        weights_a: (n_contexts, n_units) Ridge weights for stim_idx_a
        weights_b: (n_contexts, n_units) Ridge weights for stim_idx_b
    """
    ctx_arr  = act_dict["context"]
    stim_arr = act_dict["stimulus"]
    hidden   = act_dict["stim_hidden" if period == "stim" else "reward_hidden"]
    ts, H    = hidden.shape[1], hidden.shape[2]
    n_ctx    = int(ctx_arr.max()) + 1

    cos_sim   = np.full(n_ctx, np.nan)
    weights_a = np.full((n_ctx, H), np.nan)
    weights_b = np.full((n_ctx, H), np.nan)

    def _idx_mask(s_arr, idx):
        return np.ones(len(s_arr), dtype=bool) if idx is None else np.isin(s_arr, idx)

    for c in range(n_ctx):
        mask_c = ctx_arr == c
        X_all  = hidden[mask_c].mean(1) if pooling == "average" else hidden[mask_c].reshape(-1, H)
        s_all  = stim_arr[mask_c] if pooling == "average" else np.repeat(stim_arr[mask_c], ts)

        sc   = StandardScaler().fit(X_all)
        X_sc = sc.transform(X_all)

        for w_out, idx in [(weights_a, stim_idx_a), (weights_b, stim_idx_b)]:
            mask_s = _idx_mask(s_all, idx)
            y_sub  = value_matrix[s_all[mask_s], c]
            if len(np.unique(y_sub)) < 2:
                continue
            w_out[c] = Ridge().fit(X_sc[mask_s], y_sub).coef_

        wa, wb = weights_a[c], weights_b[c]
        if not (np.isnan(wa).any() or np.isnan(wb).any()):
            denom    = np.linalg.norm(wa) * np.linalg.norm(wb)
            cos_sim[c] = float(wa @ wb / denom) if denom > 0 else np.nan

    return cos_sim, weights_a, weights_b


# ── stimulus activation geometry ──────────────────────────────────────────

def stimulus_mean_activations(act_dict, period="stim"):
    """Mean hidden state per stimulus, averaged across all trials and timesteps.

    Args:
        act_dict: activations dict with stim_hidden/reward_hidden and stimulus.
        period: "stim" or "reward".

    Returns:
        mean_acts: dict mapping stim_idx → (hidden_size,) array.
    """
    hidden   = act_dict["stim_hidden" if period == "stim" else "reward_hidden"]
    stim_arr = act_dict["stimulus"]
    n_stim   = int(stim_arr.max()) + 1

    mean_acts = {}
    for si in range(n_stim):
        mask = stim_arr == si
        if mask.sum() == 0:
            continue
        mean_acts[si] = np.nanmean(hidden[mask], axis=(0, 1))
    return mean_acts


def stimulus_distance_matrix(mean_acts):
    """Pairwise Euclidean and cosine distances between stimulus mean activations.

    Args:
        mean_acts: dict mapping stim_idx → (hidden_size,) array, e.g. from
                   stimulus_mean_activations.

    Returns:
        euclidean: (n_stim, n_stim) matrix of L2 distances.
        cosine:    (n_stim, n_stim) matrix of cosine distances (1 − cosine sim).
        stim_keys: ordered list of stimulus indices corresponding to rows/cols.
    """
    stim_keys = sorted(mean_acts.keys())
    n = len(stim_keys)
    euclidean = np.zeros((n, n))
    cosine    = np.zeros((n, n))

    vecs = np.stack([mean_acts[k] for k in stim_keys])  # (n, hidden_size)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1e-12
    vecs_n = vecs / norms

    for i in range(n):
        for j in range(n):
            euclidean[i, j] = float(np.linalg.norm(vecs[i] - vecs[j]))
            cosine[i, j]    = float(1.0 - (vecs_n[i] @ vecs_n[j]))

    return euclidean, cosine, stim_keys


def neuron_response_groups(tuning, response_threshold=0.5):
    """Classify neurons by which stimuli they strongly respond to, and compute overlaps.

    A neuron is assigned to stimulus s if its mean activation for s (averaged
    across contexts) is ≥ response_threshold × its maximum activation across
    all stimuli.

    Args:
        tuning: (n_units, n_stimuli, n_contexts) from compute_unit_tuning.
        response_threshold: fraction of max activation required (default 0.5).

    Returns:
        groups: list of length n_stimuli, each a boolean (n_units,) mask.
        overlap_counts: (n_stimuli, n_stimuli) matrix of |group_i ∩ group_j|.
        overlap_fracs: (n_stimuli, n_stimuli) Jaccard index of group overlaps.
        frac_responding: (n_stimuli,) fraction of neurons in each group.
    """
    mean_ctx = np.nanmean(tuning, axis=2)                              # (n_units, n_stimuli)
    n_units, n_stim = mean_ctx.shape
    max_per_unit = np.nanmax(mean_ctx, axis=1)                         # (n_units,)

    groups = []
    for si in range(n_stim):
        thresh = response_threshold * np.maximum(max_per_unit, 1e-8)
        groups.append(mean_ctx[:, si] >= thresh)                       # (n_units,) bool

    overlap_counts = np.zeros((n_stim, n_stim), dtype=int)
    overlap_fracs  = np.zeros((n_stim, n_stim))
    for i in range(n_stim):
        for j in range(n_stim):
            inter = int((groups[i] & groups[j]).sum())
            union = int((groups[i] | groups[j]).sum())
            overlap_counts[i, j] = inter
            overlap_fracs[i, j]  = inter / union if union > 0 else np.nan

    frac_responding = np.array([g.mean() for g in groups])
    return groups, overlap_counts, overlap_fracs, frac_responding


# ── policy similarity analysis ────────────────────────────────────────────

def policy_similarity_analysis(lick_sc, value_matrix, threshold=0.25,
                                clip_lo=0.0, clip_hi=1.0):
    """Policy Similarity Analysis (PSA): per-context policy distance from optimal.

    Classifies each stimulus per context as high (reward prob ≥ 1−threshold),
    low (≤ threshold), or mid.  High stimuli should ideally have lick prob = clip_hi,
    low stimuli = clip_lo.  clip_lo/clip_hi can be set to the empirically observed
    min/max lick prob to account for policy clipping (e.g. a softmax with finite
    temperature that never saturates to 0 or 1).  Mid stimuli are excluded from the
    PSA score but their lick probs are reported separately as an informative secondary
    metric.

    Args:
        lick_sc: (n_stimuli, n_contexts) mean lick probability per stimulus × context.
        value_matrix: (n_stimuli, n_contexts) reward probabilities.
        threshold: boundary between low/mid and mid/high (default 0.25).
        clip_lo: reference lick prob for low-value stimuli (default 0.0).
        clip_hi: reference lick prob for high-value stimuli (default 1.0).

    Returns:
        dict keyed by context index, each with:
            high_lick  — mean lick prob for high-value stimuli
            low_lick   — mean lick prob for low-value stimuli
            mid_lick   — mean lick prob for mid-value stimuli (NaN if none)
            psa_score  — 1 − mean absolute error from desired policy (excludes mid)
            psa_delta  — high_lick − low_lick
            high_stim  — stimulus indices classified as high
            low_stim   — stimulus indices classified as low
            mid_stim   — stimulus indices classified as mid
    """
    n_stimuli, n_contexts = value_matrix.shape
    results = {}
    for ci in range(n_contexts):
        v         = value_matrix[:, ci]
        high_stim = np.where(v >= 1 - threshold)[0]
        low_stim  = np.where(v <= threshold)[0]
        mid_stim  = np.where((v > threshold) & (v < 1 - threshold))[0]

        def _mean_lick(stim_idx):
            vals = lick_sc[stim_idx, ci]
            return float(np.nanmean(vals)) if len(vals) > 0 else np.nan

        high_lick = _mean_lick(high_stim)
        low_lick  = _mean_lick(low_stim)
        mid_lick  = _mean_lick(mid_stim)

        errors = []
        for si in high_stim:
            if not np.isnan(lick_sc[si, ci]):
                errors.append(abs(float(lick_sc[si, ci]) - clip_hi))
        for si in low_stim:
            if not np.isnan(lick_sc[si, ci]):
                errors.append(abs(float(lick_sc[si, ci]) - clip_lo))

        psa_score = 1.0 - float(np.mean(errors)) if errors else np.nan
        psa_delta = (
            float(high_lick - low_lick)
            if not (np.isnan(high_lick) or np.isnan(low_lick))
            else np.nan
        )

        results[ci] = dict(
            high_lick=high_lick,
            low_lick=low_lick,
            mid_lick=mid_lick,
            psa_score=psa_score,
            psa_delta=psa_delta,
            high_stim=high_stim,
            low_stim=low_stim,
            mid_stim=mid_stim,
        )
    return results


# ── plotting helpers ───────────────────────────────────────────────────────

def plot_generalisation_heatmap(ax, gm, contexts, vmin, vmax, cmap, colorbar_label,
                                title=None, xlabel="Test context", ylabel="Train context"):
    """Render a square generalisation matrix on ax with annotated cells."""
    import matplotlib.pyplot as plt
    from cxval.vis import add_colorbar
    n_ctx      = gm.shape[0]
    ctx_colors = [plt.cm.Set2(v) for v in np.linspace(0, 0.75, max(n_ctx, 1))]

    im = ax.imshow(gm, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    for i in range(n_ctx):
        for j in range(n_ctx):
            if not np.isnan(gm[i, j]):
                ax.text(j, i, f"{gm[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold" if i == j else "normal",
                        color="white")

    ax.set_xticks(range(n_ctx)); ax.set_xticklabels(contexts, fontsize=8)
    ax.set_yticks(range(n_ctx)); ax.set_yticklabels(contexts, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    if title:
        ax.set_title(title, fontsize=9)
    for tick, c in zip(ax.get_xticklabels(), range(n_ctx)):
        tick.set_color(ctx_colors[c])
    for tick, c in zip(ax.get_yticklabels(), range(n_ctx)):
        tick.set_color(ctx_colors[c])
    add_colorbar(ax, im, label=colorbar_label)
    return im
