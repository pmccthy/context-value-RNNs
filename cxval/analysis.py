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
from scipy.stats import pearsonr


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


# ── plotting helpers ───────────────────────────────────────────────────────

def plot_generalisation_heatmap(ax, gm, contexts, vmin, vmax, cmap, colorbar_label,
                                title=None, xlabel="Test context", ylabel="Train context"):
    """Render a square generalisation matrix on ax with annotated cells."""
    import matplotlib.pyplot as plt
    n_ctx      = gm.shape[0]
    ctx_colors = [plt.cm.Set2(v) for v in np.linspace(0, 0.75, max(n_ctx, 1))]

    im = ax.imshow(gm, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    for i in range(n_ctx):
        for j in range(n_ctx):
            if not np.isnan(gm[i, j]):
                ax.text(j, i, f"{gm[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold" if i == j else "normal")

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
    plt.colorbar(im, ax=ax, shrink=0.8, label=colorbar_label)
    return im
