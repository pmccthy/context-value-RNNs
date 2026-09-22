"""circuit_graph.py

Aggregate the trained recurrent weight matrix (backbone.h2h) into an effective
circuit graph over functionally-defined unit subgroups (responsiveness/tuning
categories from model_group_categories.py). Every ordered group pair (including
self-connections) gets TWO edges: one summing the positive h2h weights between
that pair (excitatory) and one summing the magnitude of the negative weights
(inhibitory) -- both can be non-zero for the same pair, since there is no
Dale's law constraint in this model (cxval/models.py has no sign constraint on
h2h; nonneg_coef in the training config is an unused loss-penalty knob, not a
hard constraint). Raw weight sign is therefore a crude per-connection
excitatory/inhibitory proxy, aggregated at the group level.

Every function here takes a `phase` argument ("pre" or "post"): "pre" uses the
PRE-reversal group definitions/weights (figure_data.pkl + the final pre-reversal
checkpoint, as before); "post" uses the matching POST-reversal ones
(figure_data_reversal{REV_TAG}.pkl + model_runs_reversal{REV_TAG}/.../model.pt).
Group membership (which units are "pref 0%" etc.) is allowed to differ between
phases -- a unit's tuning can itself change across the reversal -- so "pre" and
"post" diagrams are two independently-computed snapshots, not the same groups
re-weighted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "transfer" / "code"))
sys.path.insert(0, str(_HERE.parent / "panels"))  # for vigour_value.recovered_seeds

import figures as F  # noqa: E402
import model_group_categories as MG  # noqa: E402
import checkpoint_weights as CW  # noqa: E402
import reversal_gradients as RG  # noqa: E402
import vigour_value as VV  # noqa: E402

REV_TAG = RG.REV_TAG

FIGURE_DATA_DIR = _HERE.parent / "transfer" / "figure_data"
FIGURE_DATA_POST_DIR = _HERE.parent / "transfer" / f"figure_data_reversal{REV_TAG}"

GRANULARITIES = {
    "broad": dict(
        n_groups=4,
        categorize=lambda tuning, responsive: MG.unit_categories_broad(tuning, responsive),
        labels=MG.BROAD_LABELS,
    ),
    "fine": dict(
        n_groups=8,
        categorize=lambda tuning, responsive: MG.unit_categories_fine(responsive),
        labels=MG.FINE_LABELS,
    ),
}

_D_CACHE: dict = {}


def _D(phase: str = "pre"):
    key = f"D_{phase}"
    if key not in _D_CACHE:
        path = FIGURE_DATA_DIR if phase == "pre" else FIGURE_DATA_POST_DIR
        _D_CACHE[key] = F.load(str(path))
    return _D_CACHE[key]


def _final_state(model_type: str, seed: int, phase: str = "pre") -> dict:
    if phase == "pre":
        return CW.final_pre_reversal_state(model_type, seed)
    return CW.final_post_reversal_state(model_type, seed, rev_tag=REV_TAG)


def _default_seeds(model_type: str, phase: str):
    """Recovered-only seeds (vigour_value.recovered_seeds), intersected with
    seeds actually present for this phase -- falling back to every present
    seed if recovery filtering finds none (e.g. missing
    model_runs_reversal{REV_TAG} history.json data, so recovered_seeds()
    can't be computed). Used as the default `seeds=None` fallback across
    every _averaged() function below, for BOTH phases: a seed that failed to
    recover post-reversal is excluded even from its own PRE-reversal circuit
    summary, matching vigour_value.py's own convention of applying one
    consistent recovered-seed set to both phases of a given model_type (see
    that module's docstring) -- so this row lines up with the vigour/value/
    weight-change/coding-angle rows in the same composite figure, all of
    which use the same seed set."""
    D = _D(phase)
    present = set(D["seeds_present"].get(model_type, D["seeds"]))
    rec = VV.recovered_seeds(model_type)
    if rec:
        kept = sorted(rec & present)
        if kept:
            return kept
    return sorted(present)


def _cats_for(model_type: str, seed: int, granularity: str, phase: str):
    D = _D(phase)
    ti = D["model_types"].index(model_type)
    si = D["seeds"].index(seed)
    tuning = D["tuning"]["data"][ti, si]
    responsive = D["responsive"]["data"][ti, si]
    spec = GRANULARITIES[granularity]
    cats = np.asarray(spec["categorize"](tuning, responsive))
    return cats, spec["n_groups"]


def group_edges(model_type: str, seed: int, granularity: str = "broad", phase: str = "pre"):
    """Per-seed group-to-group excitatory/inhibitory weight-mass matrices.

    Returns (n_per_group, pos_mat, neg_mat):
      n_per_group : (n_groups,) unit count per group
      pos_mat[t, s] : sum of positive h2h[i, j] for i in group t (target), j in group s (source)
      neg_mat[t, s] : sum of |negative h2h[i, j]| for the same (t, s) pair
    """
    cats, n_groups = _cats_for(model_type, seed, granularity, phase)
    sd = _final_state(model_type, seed, phase)
    W = sd[CW.RECURRENT_KEY]  # (128,128): row i = target unit, col j = source unit

    n_per_group = np.array([(cats == g).sum() for g in range(n_groups)])
    pos_mat = np.zeros((n_groups, n_groups))
    neg_mat = np.zeros((n_groups, n_groups))
    for t in range(n_groups):
        rows = cats == t
        if not rows.any():
            continue
        for s in range(n_groups):
            cols = cats == s
            if not cols.any():
                continue
            sub = W[np.ix_(rows, cols)]
            pos_mat[t, s] = sub[sub > 0].sum()
            neg_mat[t, s] = -sub[sub < 0].sum()
    return n_per_group, pos_mat, neg_mat


def group_edges_averaged(model_type: str, granularity: str = "broad", seeds=None, phase: str = "pre"):
    """Mean n_per_group/pos_mat/neg_mat across seeds (group definitions are per-seed
    but the granularity's n_groups/labels are canonical, so seeds are comparable).

    NOTE on averaging: n_per_group is the mean unit count per group ACROSS SEEDS
    (so it is generally non-integer -- e.g. "n=37.8" -- by design, since group
    membership is data-dependent per seed and 30 independent networks rarely
    agree exactly on a group's size). pos_mat/neg_mat are likewise the elementwise
    mean edge-mass across every seed that had a usable checkpoint, NOT a single
    example network's weights."""
    D = _D(phase)
    seeds = seeds if seeds is not None else _default_seeds(model_type, phase)
    n_list, pos_list, neg_list = [], [], []
    for seed in seeds:
        try:
            n, pos, neg = group_edges(model_type, seed, granularity=granularity, phase=phase)
        except Exception:
            continue
        n_list.append(n)
        pos_list.append(pos)
        neg_list.append(neg)
    if not n_list:
        raise FileNotFoundError(f"no circuit-graph data for {model_type}/{granularity}/{phase}")
    return (
        np.mean(n_list, axis=0),
        np.mean(pos_list, axis=0),
        np.mean(neg_list, axis=0),
        len(n_list),
    )


# Input dim order confirmed directly against the ACTUAL training code (the pinned
# results/transfer/reproducibility/cxval snapshot the real training scripts import,
# NOT the top-level dev cxval/ -- the two disagree and only the pinned one is what
# actually trained these checkpoints): state = [context_onehot (n_ctx dims) |
# stimulus_onehot (n_stim dims) | reward_window_cue (1 dim)], and for these runs
# n_ctx=1, n_stim=3 (input2h.weight is (128,5) = 1+3+1).
#
# IMPORTANT correction from an earlier version of this label set: the 3 middle
# dims are a one-hot over PHYSICAL stimulus identity (which of 3 sensory stimuli
# was shown), NOT literally "stim 0%/50%/100%" as fixed reward labels -- exactly
# like the functional-group labels below (BROAD_LABELS), a physical stimulus's
# CURRENT reward association flips at the reversal (VM_PRE=[0,.5,1] ->
# VM_POST=[1,.5,0], 50% stim unchanged -- see rpe_proxy.py), while its identity/
# index does not. So "physical stim index 0" is labelled "0%→100%" here (0%
# pre-reversal, 100% post-reversal) rather than a single misleading "0%" that
# would silently mean the opposite thing in a post-reversal diagram. And the
# "context" dim, with n_ctx=1 for these (non-reversal-cued) runs, is a constant
# always-on bit carrying no trial-to-trial information -- it's a real weight
# column (so it's shown for architectural completeness) but not a meaningful
# varying "context signal" the way the term might suggest.
INPUT_LABELS = ["context (constant)", "stim 0%→100%", "stim 50%", "stim 100%→0%", "response cue"]

# stim_head has 3 output rows (one logit per physical stimulus, classifying
# "which stimulus was just shown" -- an SSL objective, unrelated to reward),
# confirmed directly against a real checkpoint's stim_head.weight shape (3, 128).
# Row order matches the same physical-stimulus-index convention as INPUT_LABELS
# above (the classification target labels used in training are the same 0/1/2
# stimulus indices), so each row gets its own transition-aware label rather than
# being collapsed into one aggregate "stim" edge.
STIM_HEAD_ROW_LABELS = ["stim 0%→100%", "stim 50%", "stim 100%→0%"]


def group_input_readout(model_type: str, seed: int, granularity: str = "broad", phase: str = "pre"):
    """Per-group AGGREGATED (summed over all input dims / all heads) input->hidden
    and hidden->readout-heads excitatory/inhibitory weight mass. Kept for backward
    compatibility -- circuit_diagram.py now uses group_input_readout_split()
    instead, which keeps each input dim and head separate.

    Returns (in_pos, in_neg, out_pos, out_neg), each (n_groups,).
    """
    cats, n_groups = _cats_for(model_type, seed, granularity, phase)
    sd = _final_state(model_type, seed, phase)
    Win = sd[CW.INPUT_KEY]  # (N_HIDDEN, input_size)

    in_pos = np.zeros(n_groups)
    in_neg = np.zeros(n_groups)
    out_pos = np.zeros(n_groups)
    out_neg = np.zeros(n_groups)
    for g in range(n_groups):
        rows = cats == g
        if not rows.any():
            continue
        sub = Win[rows, :]
        in_pos[g] = sub[sub > 0].sum()
        in_neg[g] = -sub[sub < 0].sum()

        head_vals = []
        for hk in CW.HEAD_KEYS:
            if hk not in sd:
                continue
            Wh = sd[hk]  # (out_dim, n_cols)
            n_cols = Wh.shape[1]
            cols = rows[:n_cols]
            if not cols.any():
                continue
            head_vals.append(Wh[:, cols].ravel())
        if head_vals:
            allv = np.concatenate(head_vals)
            out_pos[g] = allv[allv > 0].sum()
            out_neg[g] = -allv[allv < 0].sum()
    return in_pos, in_neg, out_pos, out_neg


def group_input_readout_averaged(model_type: str, granularity: str = "broad", seeds=None, phase: str = "pre"):
    """Mean in_pos/in_neg/out_pos/out_neg across seeds -- see group_input_readout()."""
    D = _D(phase)
    seeds = seeds if seeds is not None else _default_seeds(model_type, phase)
    collected = ([], [], [], [])
    for seed in seeds:
        try:
            vals = group_input_readout(model_type, seed, granularity=granularity, phase=phase)
        except Exception:
            continue
        for lst, v in zip(collected, vals):
            lst.append(v)
    if not collected[0]:
        raise FileNotFoundError(f"no input/readout circuit data for {model_type}/{granularity}/{phase}")
    return tuple(np.mean(lst, axis=0) for lst in collected)


def group_input_readout_split(model_type: str, seed: int, granularity: str = "broad", phase: str = "pre"):
    """Like group_input_readout(), but keeps each specific input dimension and
    each specific readout head SEPARATE instead of summing them into one
    aggregated "input" / "readout heads" scalar per group.

    Returns (in_pos, in_neg, out_pos, out_neg, head_names):
      in_pos, in_neg   : (n_groups, len(INPUT_LABELS)) -- one column per specific
                          OHE input dimension (context / stim 0% / stim 50% /
                          stim 100% / response cue), in INPUT_LABELS order.
      out_pos, out_neg : (n_groups, n_heads_present) -- one column per readout
                          OUTPUT UNIT actually present in this model_type's
                          checkpoint: "actor" and "critic" (one column each --
                          vigour_head/value_head are scalar), plus, only for
                          model types with an SSL stim_head (absent for
                          rl_only), THREE separate columns (one per physical
                          stimulus, using the same transition-aware labels as
                          STIM_HEAD_ROW_LABELS), in the order given by
                          head_names.
      head_names       : list[str] display label per output-unit column, e.g.
                          ["actor", "critic", "stim 0%\u2192100%", "stim 50%",
                          "stim 100%\u21920%"].
    """
    cats, n_groups = _cats_for(model_type, seed, granularity, phase)
    sd = _final_state(model_type, seed, phase)
    Win = sd[CW.INPUT_KEY]  # (N_HIDDEN, n_input_dims)
    n_input_dims = Win.shape[1]

    # Build one (head_key, row_index, label) entry per OUTPUT UNIT, not per head:
    # vigour_head/value_head have a single output row each (out_dim=1 -- scalar
    # actor mean / critic value), so they contribute one column each ("actor",
    # "critic"). stim_head has 3 output rows (one logit per physical stimulus --
    # confirmed against a real checkpoint's stim_head.weight shape (3,128)), so it
    # contributes THREE separate columns, one per physical stimulus, using the
    # same transition-aware labels as INPUT_LABELS/STIM_HEAD_ROW_LABELS above --
    # NOT one aggregated "stim" column, since collapsing 3 distinct predicted
    # classes into one edge would hide which specific stimulus each connection
    # actually drives.
    output_specs = []  # (head_key, row_idx, label)
    for hk in CW.HEAD_KEYS:
        if hk not in sd:
            continue
        out_dim = sd[hk].shape[0]
        if out_dim == 1:
            output_specs.append((hk, 0, CW.HEAD_SHORT_NAMES.get(hk, hk)))
        else:
            for ri in range(out_dim):
                label = STIM_HEAD_ROW_LABELS[ri] if ri < len(STIM_HEAD_ROW_LABELS) else f"{hk}[{ri}]"
                output_specs.append((hk, ri, label))
    head_names = [label for _, _, label in output_specs]
    n_heads = len(output_specs)

    in_pos = np.zeros((n_groups, n_input_dims))
    in_neg = np.zeros((n_groups, n_input_dims))
    out_pos = np.zeros((n_groups, n_heads))
    out_neg = np.zeros((n_groups, n_heads))
    for g in range(n_groups):
        rows = cats == g
        if not rows.any():
            continue
        sub = Win[rows, :]  # (n_rows_in_group, n_input_dims)
        in_pos[g] = np.where(sub > 0, sub, 0.0).sum(axis=0)
        in_neg[g] = -np.where(sub < 0, sub, 0.0).sum(axis=0)

        for hi, (hk, ri, _label) in enumerate(output_specs):
            Wh = sd[hk]  # (out_dim, n_cols)
            n_cols = Wh.shape[1]
            cols = rows[:n_cols]
            if not cols.any():
                continue
            sub_h = Wh[ri, cols]  # this output row's weights, restricted to this group's units
            out_pos[g, hi] = sub_h[sub_h > 0].sum()
            out_neg[g, hi] = -sub_h[sub_h < 0].sum()
    return in_pos, in_neg, out_pos, out_neg, head_names


def group_input_readout_split_averaged(model_type: str, granularity: str = "broad", seeds=None, phase: str = "pre"):
    """Mean in_pos/in_neg/out_pos/out_neg across seeds -- see
    group_input_readout_split(). head_names is taken from the last seed that
    succeeded (identical across seeds for a given model_type, since head
    presence is a model-type property, not a per-seed one)."""
    D = _D(phase)
    seeds = seeds if seeds is not None else _default_seeds(model_type, phase)
    in_pos_l, in_neg_l, out_pos_l, out_neg_l = [], [], [], []
    head_names = None
    for seed in seeds:
        try:
            ip, ineg, op, oneg, hn = group_input_readout_split(model_type, seed, granularity=granularity, phase=phase)
        except Exception:
            continue
        in_pos_l.append(ip)
        in_neg_l.append(ineg)
        out_pos_l.append(op)
        out_neg_l.append(oneg)
        head_names = hn
    if not in_pos_l:
        raise FileNotFoundError(f"no input/readout circuit data for {model_type}/{granularity}/{phase}")
    return (
        np.mean(in_pos_l, axis=0),
        np.mean(in_neg_l, axis=0),
        np.mean(out_pos_l, axis=0),
        np.mean(out_neg_l, axis=0),
        head_names,
        len(in_pos_l),
    )
