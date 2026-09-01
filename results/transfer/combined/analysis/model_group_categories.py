"""Pure-numpy ports of the model repo's own BROAD (winner-take-all preferred
stimulus) and FINE (mixed-selectivity powerset) unit categorisation and
pre->post transition-matrix helpers, verbatim from
reversal_study/code/seed_groups.py's _unit_categories / _unit_categories_mixed
/ _transition_matrix_4cat / _transition_matrix_mixed -- reimplemented here
(rather than imported) because seed_groups.py pulls in a long chain of
plotting/analysis modules (reversal_analysis, rsa, population_similarity) with
their own scipy/torch dependencies we don't need for this. Logic is identical;
verified against the same GROUP_ORDER/_GROUP_CODES bit convention figures.py
and model_responders.py already use.

Uses the EXISTING (window-mean paired t-test) responder definition already
stored in figure_data.pkl's "responsive"/"tuning" fields -- same test
reversal_fine.py's Sankey already uses, for the reason noted there (the new
temporal-cluster test's calibration is a separate, ongoing question; see
model_responders.py and the chat).
"""
from __future__ import annotations

import numpy as np

GROUP_ORDER = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
GROUP_CODES = [sum(1 << i for i in g) for g in GROUP_ORDER]
_BITS = np.array([1, 2, 4])

BROAD_LABELS = ["silent", "pref 0%", "pref 50%", "pref 100%"]
FINE_LABELS = ["silent"] + [
    "0%-only", "50%-only", "100%-only", "0% & 50%", "0% & 100%", "50% & 100%", "all three",
]


def unit_categories_broad(tuning, responsive):
    """tuning: (3, H) stim-window mean activation. responsive: (H, 3) bool.
    Returns (H,) int in {0,1,2,3}: 0=silent, 1/2/3=preferred stim (argmax
    tuning among ANY responsive unit -- verbatim port of seed_groups._unit_categories."""
    resp_any = responsive.any(1)
    pref = tuning.argmax(0)
    return np.where(resp_any, pref + 1, 0)


def unit_categories_fine(responsive):
    """responsive: (H, 3) bool. Returns (H,) int in {0..7}: 0=silent, 1..7 =
    index into GROUP_ORDER -- verbatim port of seed_groups._unit_categories_mixed."""
    codes = responsive.astype(int) @ _BITS
    code_to_cat = {0: 0}
    for i, c in enumerate(GROUP_CODES):
        code_to_cat[c] = i + 1
    return np.array([code_to_cat[int(c)] for c in codes])


def broad_counts_pooled(D, model_type, seeds=None, stim_perm=None):
    """3-category (pref 0/50/100, EXCLUDING silent) unit counts pooled over
    seeds -- verbatim port of seed_groups._broad_counts_pooled.

    stim_perm: optional length-3 permutation applied to the LAST axis of
    tuning/responsive before classifying, e.g. to re-express a post-reversal
    array in CURRENT-value order when the array itself is identity-order (see
    model_reversal_anchoring.py)."""
    if model_type not in D["model_types"]:
        return np.zeros(3), 0
    ti = D["model_types"].index(model_type)
    seeds = D["seeds_present"][model_type] if seeds is None else seeds
    counts = np.zeros(3)
    n_silent = 0
    for s in seeds:
        if s not in D["seeds"]:
            continue
        si = D["seeds"].index(s)
        tuning = D["tuning"]["data"][ti, si]
        responsive = D["responsive"]["data"][ti, si]
        if stim_perm is not None:
            tuning = tuning[stim_perm, :]
            responsive = responsive[:, stim_perm]
        if not np.isfinite(tuning).all():
            continue
        cats = unit_categories_broad(tuning, responsive)
        for k in range(3):
            counts[k] += int((cats == k + 1).sum())
        n_silent += int((cats == 0).sum())
    return counts, n_silent


def fine_counts_pooled(D, model_type, seeds=None, stim_perm=None):
    """7-category (fine mixed-selectivity, EXCLUDING silent) unit counts
    pooled over seeds, GROUP_ORDER order -- matches figures.responder_group_counts
    (which model_responders.py / population_similarity.py already use for the
    fine chi2)."""
    if model_type not in D["model_types"]:
        return np.zeros(7), 0
    ti = D["model_types"].index(model_type)
    seeds = D["seeds_present"][model_type] if seeds is None else seeds
    counts = np.zeros(7)
    n_silent = 0
    for s in seeds:
        if s not in D["seeds"]:
            continue
        si = D["seeds"].index(s)
        responsive = D["responsive"]["data"][ti, si]
        if stim_perm is not None:
            responsive = responsive[:, stim_perm]
        cats = unit_categories_fine(responsive)
        for k in range(7):
            counts[k] += int((cats == k + 1).sum())
        n_silent += int((cats == 0).sum())
    return counts, n_silent


def transition_matrix_broad(Dpre, Dpost, model_type, seeds=None, post_stim_perm=None):
    """4x4 (pre-category x post-category) unit-count matrix pooled over seeds
    -- verbatim port of seed_groups._transition_matrix_4cat. post_stim_perm
    re-orders Dpost's stim axis before classifying (identity->current-value)."""
    M = np.zeros((4, 4))
    if model_type not in Dpre["model_types"] or model_type not in Dpost["model_types"]:
        return M
    tp, to = Dpre["model_types"].index(model_type), Dpost["model_types"].index(model_type)
    seeds = seeds if seeds is not None else [
        s for s in Dpre["seeds_present"][model_type] if s in Dpost["seeds_present"][model_type]
    ]
    for s in seeds:
        if s not in Dpre["seeds"] or s not in Dpost["seeds"]:
            continue
        pi, qi = Dpre["seeds"].index(s), Dpost["seeds"].index(s)
        tpre, tpost = Dpre["tuning"]["data"][tp, pi], Dpost["tuning"]["data"][to, qi]
        rpre, rpost = Dpre["responsive"]["data"][tp, pi], Dpost["responsive"]["data"][to, qi]
        if post_stim_perm is not None:
            tpost = tpost[post_stim_perm, :]
            rpost = rpost[:, post_stim_perm]
        if not (np.isfinite(tpre).all() and np.isfinite(tpost).all()):
            continue
        cpre = unit_categories_broad(tpre, rpre)
        cpost = unit_categories_broad(tpost, rpost)
        for a in range(4):
            for b in range(4):
                M[a, b] += int(((cpre == a) & (cpost == b)).sum())
    return M


def transition_matrix_fine(Dpre, Dpost, model_type, seeds=None, post_stim_perm=None):
    """8x8 (pre-category x post-category) unit-count matrix pooled over seeds
    -- verbatim port of seed_groups._transition_matrix_mixed."""
    M = np.zeros((8, 8))
    if model_type not in Dpre["model_types"] or model_type not in Dpost["model_types"]:
        return M
    tp, to = Dpre["model_types"].index(model_type), Dpost["model_types"].index(model_type)
    seeds = seeds if seeds is not None else [
        s for s in Dpre["seeds_present"][model_type] if s in Dpost["seeds_present"][model_type]
    ]
    for s in seeds:
        if s not in Dpre["seeds"] or s not in Dpost["seeds"]:
            continue
        pi, qi = Dpre["seeds"].index(s), Dpost["seeds"].index(s)
        rpre, rpost = Dpre["responsive"]["data"][tp, pi], Dpost["responsive"]["data"][to, qi]
        if post_stim_perm is not None:
            rpost = rpost[:, post_stim_perm]
        cpre = unit_categories_fine(rpre)
        cpost = unit_categories_fine(rpost)
        for a in range(8):
            for b in range(8):
                M[a, b] += int(((cpre == a) & (cpost == b)).sum())
    return M
