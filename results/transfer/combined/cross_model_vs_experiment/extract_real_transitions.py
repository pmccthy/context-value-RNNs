"""Real pre->post reversal responder-group TRANSITION matrices (for Sankey
comparison), ported from neuronal-representations' subgroups.py
(_reversal_groups/_transition_matrix for BROAD [winner-take-all preferred
stimulus], _fine_cats/_cell_category/_fine_transition for FINE [mixed-
selectivity powerset]) -- generalized to take an explicit stimulus `order`
instead of the hardcoded REV_ORDER global, so the category axis can be
aligned to the model's fixed identity-anchored index order (see chat: the
model's "0%/50%/100%" stim labels are tied to each cue's PRE-reversal
identity, unchanged post-reversal -- same convention the real CSV's
transition-labelled `stimulus` column already uses natively; no relabeling
by current value, unlike the (now-corrected) group-size extraction bug in
extract_experiment_group_counts.py's old reversal_label_to_current).

order = ["0_to_100", "50", "100_to_0"] aligns index 0/1/2 with the model's
own [0%, 50%, 100%] cue-identity axis (0_to_100 = the cue that WAS 0%, etc).
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REVERSAL_CSV = (Path(
    os.environ.get("NEURONAL_REPO", str(Path.home() / "Documents" / "neuronal-representations"))
) / "results" / "transfer" / "data" / "subgroups" / "responsiveness_ttest_reversal_long.csv")
NONRESP = "non-responsive"


def reversal_groups(g_long, order):
    df = g_long.copy()
    df["stimulus"] = df["stimulus"].astype(str); df["phase"] = df["phase"].astype(str)
    rows = []
    for (sess, nid), cell in df.groupby(["session", "neuron_id"]):
        rec = {"session": sess, "neuron_id": nid}
        for phase in ["pre", "post"]:
            p = cell[cell.phase == phase]
            sig_rows = p[p.sig]
            src = sig_rows if len(sig_rows) else p
            pref = src.loc[src.resp.idxmax(), "stimulus"] if len(src) else order[0]
            rec[f"sig_{phase}"] = bool(len(sig_rows) > 0)
            rec[f"pref_{phase}"] = pref
        rows.append(rec)
    return pd.DataFrame(rows)


def transition_matrix_broad(g, order):
    cats = list(order) + [NONRESP]
    idx = {c: i for i, c in enumerate(cats)}
    left = np.where(g.sig_pre, g.pref_pre.astype(str), NONRESP)
    right = np.where(g.sig_post, g.pref_post.astype(str), NONRESP)
    M = np.zeros((len(cats), len(cats)))
    for l, r in zip(left, right):
        if l in idx and r in idx:
            M[idx[l], idx[r]] += 1
    return M, cats


def fine_cats(order):
    a, b, c = order
    return [a, b, c, f"{a}&{b}", f"{a}&{c}", f"{b}&{c}", f"{a}&{b}&{c}", NONRESP]


def cell_category(sig_stims, order):
    S = [s for s in order if s in sig_stims]
    if not S:
        return NONRESP
    a, b, c = order
    key = {(a,): a, (b,): b, (c,): c, (a, b): f"{a}&{b}", (a, c): f"{a}&{c}",
           (b, c): f"{b}&{c}", (a, b, c): f"{a}&{b}&{c}"}
    return key[tuple(S)]


def transition_matrix_fine(df, order):
    df = df.copy()
    df["stimulus"] = df["stimulus"].astype(str); df["phase"] = df["phase"].astype(str)
    cats = fine_cats(order)
    idx = {c: i for i, c in enumerate(cats)}
    per = {}
    for phase in ["pre", "post"]:
        sub = df[df.phase == phase]
        sig_sets = (sub[sub.sig].groupby(["session", "neuron_id"]).stimulus
                    .apply(lambda s: set(s.astype(str))).to_dict())
        per[phase] = sig_sets
    cells = df[["session", "neuron_id"]].drop_duplicates()
    M = np.zeros((len(cats), len(cats)))
    for _, (sess, nid) in cells.iterrows():
        l = cell_category(per["pre"].get((sess, nid), set()), order)
        r = cell_category(per["post"].get((sess, nid), set()), order)
        M[idx[l], idx[r]] += 1
    return M, cats


def nonresp_first(M):
    """Reorder an (k,k) matrix with NONRESP last -> NONRESP/silent first, to
    match the model side's [silent, ...] convention."""
    k = M.shape[0]
    perm = [k - 1] + list(range(k - 1))
    return M[np.ix_(perm, perm)]


if __name__ == "__main__":
    order = ["0_to_100", "50", "100_to_0"]   # aligned to model's [0%, 50%, 100%]
    df = pd.read_csv(REVERSAL_CSV)
    g = reversal_groups(df, order)
    M_broad, cats_b = transition_matrix_broad(g, order)
    M_fine, cats_f = transition_matrix_fine(df, order)
    M_broad_r = nonresp_first(M_broad)
    M_fine_r = nonresp_first(M_fine)
    out = {"broad": M_broad_r.tolist(), "fine": M_fine_r.tolist(),
           "broad_cats_reordered": [cats_b[-1]] + cats_b[:-1],
           "fine_cats_reordered": [cats_f[-1]] + cats_f[:-1]}
    Path("group_counts").mkdir(exist_ok=True)
    json.dump(out, open("group_counts/real_transitions.json", "w"), indent=2)
    print("broad cats:", out["broad_cats_reordered"])
    print("broad M:\n", M_broad_r)
    print("\nfine cats:", out["fine_cats_reordered"])
    print("fine M totals per row:", M_fine_r.sum(1))
    print("written group_counts/real_transitions.json")
