#!/usr/bin/env python3
"""Extract real experimental responder-group counts, in the exact schema that
reversal_study/code/population_similarity.py's chi2_fit_to_data() /
fig_group_chi2() expect via --data-counts, from the neuronal-representations
tidy responsiveness tables.

Background: chi2_fit_to_data (reversal_study/code/population_similarity.py:208,
identical copy at scripts/16_06_26_population_similarity.py) has existed for a
while but has ONLY ever been run against a synthetic placeholder
(fake_data_counts()) -- real neural counts were never plugged in. This script
produces that missing input, for both the non-reversal ("expert") dataset and
each phase of the reversal dataset.

Source data (from ~/Documents/neuronal-representations/results/transfer):
  data/subgroups/responsiveness_ttest_expert_long.csv
    columns: session, neuron_id, stimulus (int: 0/50/100), sig (bool), effect, resp
  data/subgroups/responsiveness_ttest_reversal_long.csv
    columns: session, neuron_id, phase (pre/post), stimulus (str: "50",
    "100_to_0", "0_to_100" -- a TRANSITION label, constant per physical
    stimulus across both phases), sig, effect, resp

Reversal stimulus labels encode the pre->post reward-probability transition
of that physical stimulus, not its current value, so they're re-mapped to a
"current reward probability at this phase" identity before grouping:
  "50"        -> 50   regardless of phase
  "100_to_0"  -> 100 if phase == pre else 0
  "0_to_100"  -> 0   if phase == pre else 100
This matches the model side, where group labels (0%-only / 50%-only / ...)
are also defined by the CURRENTLY-in-force reward probability of the input a
unit responds to (see reversal_study/code/figures.py's GROUPS / fig_pref_transition
in reversal_analysis.py), not by a fixed input identity.

A cell's responder-group membership follows the exact 7-way taxonomy used in
reversal_study/code/figures.py (GROUPS / GROUP_ORDER): for each (session,
neuron_id) [x phase, for the reversal set], take the set of CURRENT stimulus
identities it is `sig` for; significant for exactly one -> that stimulus's
"-only" group; exactly two -> the pairwise group; all three -> "all three".
Cells significant for none are excluded from the 7-group counts (matching
chi2_fit_to_data's dof=6), but reported separately for reference.

Usage:
  python3 extract_experiment_group_counts.py expert
  python3 extract_experiment_group_counts.py reversal --phase pre
  python3 extract_experiment_group_counts.py reversal --phase post

Output: group_counts/<name>.json, ready for e.g.
  python3 code/population_similarity.py --pre ... --out ... \
      --data-counts group_counts/expert.json
"""
import argparse
import json
import sys
import os
from pathlib import Path

import pandas as pd

EXPERIMENT_ROOT = Path(
    os.environ.get("NEURONAL_REPO", str(Path.home() / "Documents" / "neuronal-representations"))
) / "results" / "transfer"
EXPERT_CSV = EXPERIMENT_ROOT / "data" / "subgroups" / "responsiveness_ttest_expert_long.csv"
REVERSAL_CSV = EXPERIMENT_ROOT / "data" / "subgroups" / "responsiveness_ttest_reversal_long.csv"

# Must match GROUP_LABELS / GROUP_ORDER in reversal_study/code/figures.py exactly.
GROUP_LABELS = [
    "0%-only", "50%-only", "100%-only",
    "0% & 50%", "0% & 100%", "50% & 100%",
    "all three",
]
_ORDER = [0, 50, 100]
_SINGLE = {(0,): "0%-only", (50,): "50%-only", (100,): "100%-only"}
_PAIR = {(0, 50): "0% & 50%", (0, 100): "0% & 100%", (50, 100): "50% & 100%"}
_ALL = {(0, 50, 100): "all three"}
_CLASSIFY = {**_SINGLE, **_PAIR, **_ALL}


def classify(current_stim_set):
    key = tuple(s for s in _ORDER if s in current_stim_set)
    return _CLASSIFY.get(key)


# CORRECTED (see chat): this used to be reversal_label_to_current(), which
# relabelled each physical cue by its CURRENT reward probability at each
# phase (so "100_to_0" -> 100 pre, 0 post). That was based on a wrong
# assumption that the model side does the same. It doesn't: reversal_study/
# code/reversal_analysis.py's fig_pref_transition() plots the model's
# pre->post preferred-stimulus transition on a FIXED, identity-anchored 3x3
# axis (same stim index = same physical cue at both phases; its title reads
# "diagonal = identity kept, anti-diagonal = code followed the value swap"
# -- i.e. identity-vs-value tracking is the thing being TESTED, not baked
# into the label). So the model's own "0%/50%/100%" stim labels are tied to
# the cue's ORIGINAL (pre-reversal) identity/value, unchanged post-reversal --
# matching the real data's own REV_ORDER labels (also identity-anchored: a
# neuron_id's "stimulus" field is the fixed transition label "100_to_0" etc,
# constant across phase). Fixed here to match: identity is now the cue's
# PRE-reversal value, used unchanged at both phases.
def reversal_label_to_current(stim_label, phase):
    """Map a reversal-CSV transition label to its FIXED physical-cue identity
    (the cue's pre-reversal reward probability), independent of phase -- see
    the correction note above. `phase` is accepted (unused) to keep call
    sites unchanged."""
    if stim_label == "50":
        return 50
    if stim_label == "100_to_0":
        return 100
    if stim_label == "0_to_100":
        return 0
    raise ValueError(f"unrecognized reversal stimulus label: {stim_label!r}")


def group_counts_from_long(df, current_stim_col):
    """df must have one row per (session, neuron_id, <current_stim_col>) with
    boolean column `sig`; `current_stim_col` holds the int 0/50/100 identity."""
    sig_stims = (
        df[df["sig"]]
        .groupby(["session", "neuron_id"])[current_stim_col]
        .apply(lambda s: frozenset(int(x) for x in s))
    )
    counts = {label: 0 for label in GROUP_LABELS}
    n_total_cells = df.groupby(["session", "neuron_id"]).ngroups
    for stim_set in sig_stims:
        label = classify(stim_set)
        if label is None:
            continue
        counts[label] += 1
    n_any_sig = len(sig_stims)
    n_none = n_total_cells - n_any_sig
    return counts, n_total_cells, n_none


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["expert", "reversal"])
    ap.add_argument("--phase", choices=["pre", "post"], default=None,
                     help="required when dataset=reversal")
    ap.add_argument("--out-dir", default="group_counts")
    args = ap.parse_args()

    if args.dataset == "expert":
        if not EXPERT_CSV.exists():
            sys.exit(f"missing: {EXPERT_CSV}")
        df = pd.read_csv(EXPERT_CSV)
        df["current_stim"] = df["stimulus"].astype(int)
        src = EXPERT_CSV
        out_name = "expert"
    else:
        if args.phase is None:
            sys.exit("--phase pre|post is required for dataset=reversal")
        if not REVERSAL_CSV.exists():
            sys.exit(f"missing: {REVERSAL_CSV}")
        df = pd.read_csv(REVERSAL_CSV)
        if "phase" not in df.columns:
            sys.exit(f"expected a 'phase' column in {REVERSAL_CSV}, found: {list(df.columns)}")
        df = df[df["phase"] == args.phase].copy()
        df["current_stim"] = df["stimulus"].apply(lambda s: reversal_label_to_current(s, args.phase))
        src = REVERSAL_CSV
        out_name = f"reversal_{args.phase}"

    counts, n_total_cells, n_none = group_counts_from_long(df, "current_stim")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_name}.json"
    with open(out_path, "w") as f:
        json.dump(counts, f, indent=2)

    print(f"source: {src}" + (f"  (phase={args.phase})" if args.dataset == "reversal" else ""))
    print(f"n_total_cells (with any stimulus row): {n_total_cells}")
    print(f"n_non_responsive (sig for none of the 3, current-contingency basis): {n_none}")
    print(f"n_in_7_groups: {n_total_cells - n_none}")
    print("group counts:")
    for label in GROUP_LABELS:
        print(f"  {label:12s} {counts[label]}")
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
