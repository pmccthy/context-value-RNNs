"""
Population-geometry diagnostics for the context-switch task, adapted from
nb/08_07_26_neural_geometry_models.ipynb (which developed and validated these
metrics on synthetic pseudo-populations) to run on REAL per-trial hidden-state
activations extracted from a trained model (see cxval.context_vigour.
extract_activations / scripts/csha_ccnss_2026/16_07_26_extract_activations.py).

Conditions are (stimulus, context) pairs -- n_stim x n_ctx of them (8 for the
default 4-stimulus/2-context swap task). Three dichotomies are used
throughout, matching the notebook's final (cell 40+) convention exactly (NOT
the full C(n,n/2)/2 exhaustive dichotomy enumeration sketched in an earlier,
superseded notebook cell):
    "stim"    -- swap-type: stimuli grouped by their reward probability in
                 context 0 (value_matrix[:, 0]) -- this is the notebook's
                 "stim" feature (which of the two value-swap patterns a
                 stimulus follows), NOT raw stimulus identity.
    "context" -- context 0 vs context 1.
    "value"   -- true reward probability (0 vs 1), pooled across contexts.

A fourth, per-stimulus factor ("pair" -- which exemplar within a swap-type
group, e.g. distinguishing the two 0%->100% stimuli from each other) is used
purely as a CCGP stratification / holdout factor, exactly as in the notebook.

All three dichotomies split the 8 conditions into two groups of 4.

Author: patrick.mccarthy@dpag.ox.ac.uk
"""
from __future__ import annotations

from itertools import permutations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

# marker EDGE colour convention used throughout the PCA plots: blue = high
# value (reward probability 1), red = low value (reward probability 0) --
# independent of the fill colour (stimulus identity) / marker shape (context).
VALUE_EDGE_COLOR = {1: "tab:blue", 0: "tab:red"}


# =============================================================================
# CONDITIONS / DICHOTOMIES
# =============================================================================

def build_conditions(value_matrix):
    """dict {cond_id: {stim, ctx, value, swap_type, pair}} for every (stim, ctx)
    pair, where cond_id = stim * n_ctx + ctx.

    swap_type: 0/1 grouping of STIMULI by their context-0 reward probability
        (value_matrix[:, 0]) -- the notebook's "stim" feature.
    pair: 0-based rank of a stimulus within its own swap_type group (e.g. the
        two swap_type=0 stimuli get pair=0 and pair=1) -- purely a
        stratification/holdout factor for CCGP, distinguishing exemplars that
        would otherwise be pooled together. Assumes swap_type groups are the
        same size (true for build_swap_value_matrix's default n_swap_low ==
        n_swap_high == 2); with unequal group sizes, pair still assigns a
        unique within-group rank but CCGP strata become unbalanced.

    NOTE on non-binary (probabilistic) value_matrix entries: swap_type/value
    are ROUNDED to the nearest integer (>=0.5 -> 1, <0.5 -> 0), not
    truncated -- e.g. p_high=0.8 must classify as "high" (1), and plain
    int(0.8) would silently truncate to 0. This matters once
    build_swap_value_matrix is used with p_low/p_high strictly between 0
    and 1 (probabilistic reward variant) rather than the original 0.0/1.0
    deterministic swap.
    """
    value_matrix = np.asarray(value_matrix)
    n_stim, n_ctx = value_matrix.shape
    swap_type = np.round(value_matrix[:, 0]).astype(int)
    pair = np.zeros(n_stim, dtype=int)
    for st in np.unique(swap_type):
        idx = np.where(swap_type == st)[0]
        pair[idx] = np.arange(len(idx))

    conditions = {}
    for s in range(n_stim):
        for c in range(n_ctx):
            cid = s * n_ctx + c
            conditions[cid] = dict(stim=int(s), ctx=int(c),
                                   value=int(round(float(value_matrix[s, c]))),
                                   swap_type=int(swap_type[s]), pair=int(pair[s]))
    return conditions, n_stim, n_ctx


def cond_ids_for_trials(stimulus, context, n_ctx):
    """Per-trial (stim, ctx) -> integer condition id, matching build_conditions."""
    return np.asarray(stimulus).astype(int) * n_ctx + np.asarray(context).astype(int)


# attribute (in the conditions dict) to hold out when computing CCGP for each
# dichotomy -- matches nb/08_07_26_neural_geometry_models.ipynb's holdout_map,
# just keyed by the conditions-dict attribute name rather than the loose
# design_template feature name.
HOLDOUT_ATTR = {"stim": "ctx", "context": "swap_type", "value": "pair"}


def make_dichotomies(conditions):
    """dict {"stim": (group_A, group_B), "context": (...), "value": (...)},
    each group a list of condition ids (4 of the 8 conditions)."""
    def group(key, val):
        return [cid for cid, c in conditions.items() if c[key] == val]
    return {
        "stim":    (group("swap_type", 0), group("swap_type", 1)),
        "context": (group("ctx", 0),       group("ctx", 1)),
        "value":   (group("value", 0),     group("value", 1)),
    }


# =============================================================================
# METRICS  (all operate on X: (n_trials, n_units), cond_id: (n_trials,) int)
# =============================================================================

# libsvm's SMO solver (sklearn's SVC) scales badly with sample count and has
# NO iteration cap by default (max_iter=-1) -- on a large (tens of thousands
# of trials), less-than-perfectly-separable dataset this can run for a very
# long time (in practice indistinguishable from hanging). Neither issue
# requires that many trials for a statistically stable CV accuracy estimate,
# so every SVC fit below is (a) subsampled to at most MAX_TRIALS_PER_CLASS
# per class and (b) capped at SVC_MAX_ITER iterations as a hard safety net
# -- if that cap is ever hit, sklearn raises a ConvergenceWarning, which is
# surfaced (not silenced) so a truncated fit is visible rather than silently
# treated as fully converged.
MAX_TRIALS_PER_CLASS = 2000
SVC_MAX_ITER = 20000


def _subsample_per_class(Xc, y, max_per_class, seed):
    """Cap each class to at most max_per_class trials (random, without
    replacement, seeded for reproducibility). No-op if already smaller."""
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return Xc[keep], y[keep]


def dichotomy_accuracy(X, cond_id, group_A, group_B, n_splits=5, seed=42,
                       max_trials_per_class=MAX_TRIALS_PER_CLASS):
    """Cross-validated linear-SVM decoding accuracy for one dichotomy (group_A
    vs group_B, each a list of condition ids), pooling all trials from either
    group's conditions (subsampled per class for speed -- see
    _subsample_per_class)."""
    mask = np.isin(cond_id, group_A + group_B)
    Xc, cc = X[mask], cond_id[mask]
    y = np.isin(cc, group_B).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    Xc, y = _subsample_per_class(Xc, y, max_trials_per_class, seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in cv.split(Xc, y):
        clf = SVC(kernel="linear", random_state=seed, max_iter=SVC_MAX_ITER)
        clf.fit(Xc[tr], y[tr])
        accs.append(np.mean(clf.predict(Xc[te]) == y[te]))
    return float(np.mean(accs))


def ccgp(X, cond_id, conditions, group_A, group_B, holdout_key, stratify_key="pair", seed=42,
        max_trials_per_class=MAX_TRIALS_PER_CLASS):
    """Cross-condition generalization performance (Bernardi et al. 2020):
    train a linear decoder on one level of `holdout_key`, test on the other,
    and vice versa -- so the decoder never sees the test condition's value of
    the held-out factor during training. Averaged over both train/test
    directions and (unless stratify_key == holdout_key) over every level of
    `stratify_key`, so a decoder is never trained on two different exemplars
    pooled into one class (matches the notebook's per-pair-level averaging
    rather than silently mixing exemplars)."""
    holdout_levels = sorted({conditions[cid][holdout_key] for cid in group_A + group_B})
    if len(holdout_levels) != 2:
        return np.nan

    stratify = stratify_key is not None and stratify_key != holdout_key
    if stratify:
        strat_levels = sorted({conditions[cid][stratify_key] for cid in group_A + group_B})
    else:
        strat_levels = [None]

    accs = []
    for train_level, test_level in [(holdout_levels[0], holdout_levels[1]),
                                     (holdout_levels[1], holdout_levels[0])]:
        for strat_level in strat_levels:
            def _match(cid, level):
                ok = conditions[cid][holdout_key] == level
                if stratify:
                    ok = ok and conditions[cid][stratify_key] == strat_level
                return ok
            train_conds = [cid for cid in group_A + group_B if _match(cid, train_level)]
            test_conds = [cid for cid in group_A + group_B if _match(cid, test_level)]
            if not train_conds or not test_conds:
                continue
            train_mask = np.isin(cond_id, train_conds)
            test_mask = np.isin(cond_id, test_conds)
            y_train = np.isin(cond_id[train_mask], group_B).astype(int)
            y_test = np.isin(cond_id[test_mask], group_B).astype(int)
            if len(np.unique(y_train)) < 2:
                continue
            X_train, y_train = _subsample_per_class(X[train_mask], y_train, max_trials_per_class, seed)
            X_test, y_test = _subsample_per_class(X[test_mask], y_test, max_trials_per_class, seed)
            clf = SVC(kernel="linear", random_state=seed, max_iter=SVC_MAX_ITER)
            clf.fit(X_train, y_train)
            accs.append(np.mean(clf.predict(X_test) == y_test))
    return float(np.mean(accs)) if accs else np.nan


def parallelism_score(X, cond_id, group_A, group_B):
    """Parallelism score (Bernardi et al. 2020): condition-mean coding
    vectors (group_B centroid - group_A centroid, for some A<->B pairing),
    max over every pairing of the average pairwise cosine similarity between
    all of that pairing's coding vectors. +1 = perfectly parallel (abstract/
    generalizable code), 0 = orthogonal, -1 = anti-parallel."""
    centroids = {cid: X[cond_id == cid].mean(axis=0) for cid in group_A + group_B}
    best = -np.inf
    for perm_B in permutations(group_B):
        vectors = []
        for a, b in zip(group_A, perm_B):
            v = centroids[b] - centroids[a]
            n = np.linalg.norm(v)
            vectors.append(v / n if n > 0 else v)
        vectors = np.array(vectors)
        sims = [np.dot(vectors[i], vectors[j])
                for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
        if sims:
            best = max(best, float(np.mean(sims)))
    return best if np.isfinite(best) else np.nan


def participation_ratio(X):
    """Participation ratio (effective dimensionality): PR = (sum(eigvals))^2
    / sum(eigvals^2), where eigvals are the eigenvalues of the covariance
    matrix of X's rows (computed via SVD for numerical stability, since the
    condition-averaged use case here is typically 8 points in a
    hidden_size-dimensional space -- far more features than samples).

    PR = 1 if all variance sits on a single axis; PR = min(n_samples - 1,
    n_features) if variance is spread perfectly evenly across every
    available axis. Unlike raw PCA component count, this is a continuous,
    scale-invariant measure of "how many dimensions does this data actually
    use" that doesn't require picking a variance-explained cutoff.

    Called on the 8 CONDITION-MEAN activity vectors (not raw trials) by
    compute_geometry_metrics, matching the population-geometry convention of
    asking how many dimensions the task STRUCTURE occupies, with trial-to-
    trial noise averaged out first (a trial-level PR would instead be
    dominated by whatever noise floor the recording/simulation has).
    """
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    eigvals = np.clip(s, 0, None) ** 2
    s1, s2 = eigvals.sum(), (eigvals ** 2).sum()
    return float(s1 ** 2 / s2) if s2 > 0 else 0.0


def _level_of(c, key):
    """key may be a single conditions-dict attribute name, or a tuple of
    them (grouped jointly, e.g. ("pair", "ctx") -> one level per
    (pair, ctx) combination)."""
    if isinstance(key, tuple):
        return tuple(c[k] for k in key)
    return c[key]


def value_axis_alignment(X, cond_id, conditions, split_key="pair", stratify_key=None):
    """Cosine similarity between the "value coding vector" (centroid(value=1)
    - centroid(value=0)) computed SEPARATELY within each level of split_key,
    averaged pairwise over levels -- optionally averaged again over levels of
    a stratify_key first, mirroring ccgp()'s train/test-per-stratum-then-
    average structure, so the comparison is never confounded by whatever
    stratify_key varies over.

    This is the direct, classifier-free answer to "is the value axis
    factorized (shared across [split_key]) or shattered (each level uses its
    own, unaligned axis)?": +1 = same axis everywhere (factorized/abstract),
    ~0 = orthogonal per-level axes (shattered: value IS linearly decodable
    within each level -- see dichotomy_accuracy["value"] -- but the decision
    boundaries don't line up), -1 = anti-parallel.

    Three configurations are used by compute_geometry_metrics:
      split_key="pair" (default, stratify_key=None) -- does the value axis
        generalise across stimulus-pair exemplars? The vector-geometry
        analog of ccgp["value"] (holdout=pair). Pools BOTH contexts into
        each pair-level's vector (see below for why "pair" and not
        "swap_type"), so this does NOT test context-generalization.
      split_key="ctx", stratify_key="pair" -- does the value axis generalise
        across context? The vector-geometry analog of ccgp_value_context
        (holdout=ctx, stratify=pair): computed separately within each pair
        level (so a stimulus's own ctx0-vs-ctx1 value flip isn't trivially
        conflated with a DIFFERENT stimulus's), then averaged over pair.
      split_key=("pair", "ctx"), stratify_key=None -- the strictest,
        single-vector-per-condition-pair version: are all FOUR
        (pair, context) value vectors mutually aligned at once? Each vector
        is a single condition-pair centroid difference (no within-level
        averaging to denoise), so it's noisier than the other two, and it
        overlaps conceptually with parallelism_score["value"] (which
        searches over all condition-pairings and reports the BEST alignment
        found; this instead uses the one specific, physically meaningful
        pairing -- same pair AND same context -- so it can be lower than
        parallelism_score if that specific pairing isn't the best one).

    split_key="pair" (not "swap_type") specifically because splitting by
    swap_type is degenerate for this purpose: within one swap_type group
    every stimulus has the SAME value at a given context (that's what makes
    them the same swap type), so there is no value contrast to build a
    vector from without crossing context -- which would confound the vector
    with a context axis instead. "pair" groups one exemplar from EACH
    swap_type together (see build_conditions), so every level already
    contains both value levels at both contexts, giving a value vector that
    is not confounded with either stimulus identity or context.

    If dichotomy_accuracy["value"] is itself low (chance), don't over-read
    this metric -- a low-magnitude/noisy vector's cosine similarity to
    another noisy vector is not very meaningful; check that value is at
    least linearly decodable somewhere before asking whether it's the SAME
    decodable axis everywhere.
    """
    split_levels = sorted({_level_of(c, split_key) for c in conditions.values()})
    strat_levels = (sorted({_level_of(c, stratify_key) for c in conditions.values()})
                    if stratify_key is not None else [None])

    def _vector(level, strat_level):
        def _match(c):
            ok = _level_of(c, split_key) == level
            if strat_level is not None:
                ok = ok and _level_of(c, stratify_key) == strat_level
            return ok
        ids0 = [cid for cid, c in conditions.items() if _match(c) and c["value"] == 0]
        ids1 = [cid for cid, c in conditions.items() if _match(c) and c["value"] == 1]
        if not ids0 or not ids1:
            return None
        c0 = np.mean([X[cond_id == cid].mean(axis=0) for cid in ids0], axis=0)
        c1 = np.mean([X[cond_id == cid].mean(axis=0) for cid in ids1], axis=0)
        v = c1 - c0
        n = np.linalg.norm(v)
        return v / n if n > 0 else None

    strat_sims = []
    for strat_level in strat_levels:
        vectors = [v for v in (_vector(level, strat_level) for level in split_levels)
                  if v is not None]
        if len(vectors) < 2:
            continue
        sims = [np.dot(vectors[i], vectors[j])
                for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
        strat_sims.append(float(np.mean(sims)))
    return float(np.mean(strat_sims)) if strat_sims else np.nan


def condition_average_pca(X, cond_id, conditions, n_components=2, seed=42):
    """Fit PCA on the (trial-level) activity and return both the per-trial
    projection and the condition centroids in PC space, for the classic
    Bernardi-style condition-averaged geometry plot."""
    pca = PCA(n_components=n_components, random_state=seed)
    X_pca = pca.fit_transform(X)
    uniq = sorted(conditions.keys())
    centroids = np.array([X_pca[cond_id == cid].mean(axis=0) for cid in uniq])
    return dict(pca=pca, X_pca=X_pca, centroid_ids=uniq, centroids=centroids)


def align_pca_signs(X_pca, centroids, centroid_ids, conditions):
    """PCA sign/orientation is arbitrary per fit, which makes comparing (or
    just visually scanning) multiple independently-fit seeds' condition
    geometries incoherent even when the underlying geometry is the same
    shape. Fix each axis's sign using a semantic convention -- PC1 points
    towards value=1, PC2 towards context=1, PC3 towards swap_type=1 -- purely
    a plotting convenience (does not affect any of the quantitative metrics
    above, which are sign/rotation invariant). Each check is skipped if that
    many components aren't present. Mutates and returns X_pca, centroids."""
    val = np.array([conditions[cid]["value"] for cid in centroid_ids])
    ctx = np.array([conditions[cid]["ctx"] for cid in centroid_ids])
    swap = np.array([conditions[cid]["swap_type"] for cid in centroid_ids])
    for axis, labels in ((0, val), (1, ctx), (2, swap)):
        if centroids.shape[1] <= axis:
            continue
        if not ((labels == 1).any() and (labels == 0).any()):
            continue
        if centroids[labels == 1, axis].mean() < centroids[labels == 0, axis].mean():
            X_pca[:, axis] *= -1
            centroids[:, axis] *= -1
    return X_pca, centroids


def condition_legend_entries(value_matrix):
    """One entry per (stim, ctx) condition, for a self-documenting legend:
    (cond_id, stim, ctx, color, marker, label) where label spells out the
    actual reward-probability outcome, e.g. "stim0 ctx1 (high)" -- so a color
    + marker combination in a plot never has to be decoded by eye against a
    separate value_matrix printout.

    Colour encodes raw stimulus INDEX (viridis, n_stim colours) -- NOT value
    directly, since value flips with context for every stimulus (that's the
    whole point of the swap task). Marker shape encodes context (circle =
    context 0, square = context 1). Because build_swap_value_matrix lists
    the "low-in-ctx0" stimuli before the "high-in-ctx0" ones, adjacent
    viridis colours happen to fall on the same swap-type pair (e.g. stim0/
    stim1 both blue-ish, stim2/stim3 both green/yellow-ish, for the default
    n_swap_low=n_swap_high=2) -- but that's a byproduct of stimulus ordering,
    not an intentional value encoding, and won't hold for a differently
    ordered or asymmetric value_matrix.
    """
    import matplotlib.pyplot as plt
    conditions, n_stim, n_ctx = build_conditions(value_matrix)
    cmap = plt.get_cmap("viridis")
    entries = []
    for cid in sorted(conditions):
        c = conditions[cid]
        color = cmap(c["stim"] / max(n_stim - 1, 1))
        marker = "o" if c["ctx"] == 0 else "s"
        level = "high" if c["value"] == 1 else "low"
        label = f"stim{c['stim']} ctx{c['ctx']} ({level})"
        entries.append(dict(cond_id=cid, stim=c["stim"], ctx=c["ctx"], value=c["value"],
                            color=color, marker=marker, edge_color=VALUE_EDGE_COLOR[c["value"]],
                            label=label))
    return entries


def condition_loop_order(conditions):
    """A generic vertex ordering that traces a coherent loop through all
    conditions for plotting: grouped by `pair` (exemplar), and within each
    pair group cycling (ctx=0,swap_type=0) -> (ctx=1,swap_type=0) ->
    (ctx=1,swap_type=1) -> (ctx=0,swap_type=1) -- the same "context x
    swap-type square, repeated per exemplar" pattern used in
    nb/08_07_26_neural_geometry_models.ipynb's vertex_order."""
    pair_levels = sorted({c["pair"] for c in conditions.values()})
    order = [(0, 0), (1, 0), (1, 1), (0, 1)]  # (ctx, swap_type)
    loop = []
    for pair in pair_levels:
        for ctx_v, swap_v in order:
            match = [cid for cid, c in conditions.items()
                    if c["pair"] == pair and c["ctx"] == ctx_v and c["swap_type"] == swap_v]
            loop.extend(match)
    return loop


# =============================================================================
# TOP-LEVEL
# =============================================================================

def compute_geometry_metrics(X, stimulus, context, value_matrix, seed=42, n_splits=5,
                             pca_components=2):
    """Compute every geometry diagnostic for one (model, epoch) activity
    matrix X (n_trials, n_units).

    Returns a dict:
        dichotomy_accuracy: {"stim": acc, "context": acc, "value": acc}
        ccgp: {"stim": ccgp, "context": ccgp, "value": ccgp}  (holdout per
            HOLDOUT_ATTR: context, swap_type, pair respectively)
        ccgp_value_context: CCGP for the value dichotomy holding out CONTEXT
            instead of pair -- tests whether the value code generalises
            across context (abstract) vs. is context-conjunctive.
        parallelism_score: {"stim": ps, "context": ps, "value": ps}
        shattering_dim: mean of the three dichotomy_accuracy values.
        participation_ratio: effective dimensionality (see participation_ratio)
            of the 8 condition-mean activity vectors, in the FULL n_units
            space (not the truncated PCA projection) -- bounded above by
            min(n_conditions - 1, n_units) = 7 for this task.
        value_axis_alignment: see value_axis_alignment() -- cosine similarity
            between the per-exemplar (pair-generalization) value coding
            vectors; distinguishes a factorized value axis (~1) from a
            shattered one (~0, each exemplar pair uses its own unaligned but
            still-decodable axis) from a genuinely non-linear/conjunctive
            code (low dichotomy_accuracy["value"] to begin with, in which
            case this number shouldn't be over-interpreted).
        value_axis_alignment_context: same idea, but split by context
            (stratified by pair) instead of by pair -- the vector-geometry
            analog of ccgp_value_context. Tests context-generalization of
            the value axis instead of pair-generalization.
        value_axis_alignment_4way: strictest version -- all four individual
            (pair, context) value vectors compared pairwise at once (see
            value_axis_alignment docstring for why this overlaps with, but
            isn't identical to, parallelism_score["value"]).
        pca: dict from condition_average_pca (pca object, per-trial and
            per-condition projections).
        conditions, cond_id: the condition metadata / per-trial condition ids
            used, for any further custom analysis.
    """
    conditions, n_stim, n_ctx = build_conditions(value_matrix)
    cond_id = cond_ids_for_trials(stimulus, context, n_ctx)
    dichots = make_dichotomies(conditions)

    X = np.asarray(X)
    if not np.isfinite(X).all():
        # Most commonly a numerically DIVERGED model (NaN weights -> NaN
        # activations at inference; caller should really have filtered this
        # out via config["diverged"] / classify_outcome's "no_data" outcome
        # -- see 16_07_26_plot_neural_geometry.py's load_all). Degrade to an
        # all-NaN result instead of an opaque sklearn crash, so one bad seed
        # doesn't take down an entire sweep's worth of metrics.
        nan_accs = {name: np.nan for name in dichots}
        return dict(
            dichotomy_accuracy=nan_accs, ccgp=dict(nan_accs), ccgp_value_context=np.nan,
            parallelism_score=dict(nan_accs), shattering_dim=np.nan, participation_ratio=np.nan,
            value_axis_alignment=np.nan, value_axis_alignment_context=np.nan,
            value_axis_alignment_4way=np.nan,
            pca=dict(pca=None, X_pca=np.full((X.shape[0], pca_components), np.nan),
                    centroid_ids=sorted(conditions.keys()),
                    centroids=np.full((len(conditions), pca_components), np.nan)),
            conditions=conditions, cond_id=cond_id, n_stim=n_stim, n_ctx=n_ctx,
            error="non-finite activity (NaN/inf) -- likely a diverged model; excluded",
        )

    accs, ccgps, pss = {}, {}, {}
    for name, (gA, gB) in dichots.items():
        accs[name] = dichotomy_accuracy(X, cond_id, gA, gB, n_splits=n_splits, seed=seed)
        pss[name] = parallelism_score(X, cond_id, gA, gB)
        ccgps[name] = ccgp(X, cond_id, conditions, gA, gB, holdout_key=HOLDOUT_ATTR[name],
                           stratify_key="pair", seed=seed)

    ccgp_value_context = ccgp(X, cond_id, conditions, *dichots["value"], holdout_key="ctx",
                              stratify_key="pair", seed=seed)

    shattering_dim = float(np.mean(list(accs.values())))
    pca_info = condition_average_pca(X, cond_id, conditions, n_components=pca_components, seed=seed)

    cond_means = np.array([X[cond_id == cid].mean(axis=0) for cid in sorted(conditions)])
    pr = participation_ratio(cond_means)
    va = value_axis_alignment(X, cond_id, conditions, split_key="pair")
    va_ctx = value_axis_alignment(X, cond_id, conditions, split_key="ctx", stratify_key="pair")
    va_4way = value_axis_alignment(X, cond_id, conditions, split_key=("pair", "ctx"))

    return dict(
        dichotomy_accuracy=accs, ccgp=ccgps, ccgp_value_context=ccgp_value_context,
        parallelism_score=pss, shattering_dim=shattering_dim, participation_ratio=pr,
        value_axis_alignment=va, value_axis_alignment_context=va_ctx,
        value_axis_alignment_4way=va_4way, pca=pca_info,
        conditions=conditions, cond_id=cond_id, n_stim=n_stim, n_ctx=n_ctx,
    )
