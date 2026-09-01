#!/usr/bin/env python3
"""Responder-significance proportion as a function of trials-per-stimulus
used, for BOTH candidate methods side by side:

  "old"     the shipped method (cxval.analysis.responsive_proportions_ttest):
            paired one-sample t-test on stim-window-mean minus baseline-
            window-mean per unit per stimulus (direction="excitatory").
            This is what all current figures use.
  "cluster" the ported temporal-cluster test (model_responders.py's
            responders_native_one_seed): independent two-sample t-test at
            each native stim timepoint against the pooled baseline, unit
            counted responsive if a contiguous (gap<=max_int) run of
            significant timepoints covers > frac of the stim window.

This is the SAME live comparison that originally justified adopting "old"
over "cluster" (see chat) -- turned into an actual figure instead of a
one-off print, and swept across model types too.

Needs sklearn-free but scipy-dependent code (ttest_ind, ttest_1samp via
cxval.analysis / model_responders.py) -- run in your cxval env:

    cd context-value-RNNs
    python3 unified_figures/analysis/responder_stability_sweep.py \
        --data unified_figures/transfer/figure_data \
        --out unified_figures/output/decoding/responder_stability.json

(~10 trial-counts x ~30 seeds x 3 model types x 2 methods; a few minutes.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import act_dict_adapter as AD
import model_responders as MR

N_TRIALS_GRID = [20, 30, 50, 75, 100, 150, 200, 300, 400, 500]
MODEL_TYPES = ["rl_only", "classif_rl", "classif_rl_readout_only"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(_HERE.parent / "transfer" / "figure_data"),
                    help="figure_data dir with a time_resolved/ subfolder (pre-reversal by default)")
    ap.add_argument("--out", default=str(_HERE.parent / "output" / "decoding" / "responder_stability.json"))
    ap.add_argument("--model-types", nargs="*", default=MODEL_TYPES)
    ap.add_argument("--n-trials-grid", nargs="*", type=int, default=N_TRIALS_GRID)
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    import cxval.analysis as A   # noqa: E402  (needs scipy -- see docstring)

    td = Path(args.data) / "time_resolved"

    results = {}
    for mt in args.model_types:
        files = AD.seed_files(td, mt)
        seeds = sorted(files)
        print(f"{mt}: {len(seeds)} seeds")

        old_by_n = {n: [] for n in args.n_trials_grid}    # n -> list over seeds of (3,) frac_per_stim
        clu_by_n = {n: [] for n in args.n_trials_grid}

        for seed in seeds:
            act_dict = AD.act_dict_from_npz(files[seed])
            for n in args.n_trials_grid:
                out = A.responsive_proportions_ttest(act_dict, period="stim", n_sub=n, n_rep=1)
                old_by_n[n].append(np.asarray(out["frac_per_stim"], float))

                resp, _diag = MR.responders_native_one_seed(
                    files[seed], n_trials_target=n, rng_seed=args.rng_seed)
                clu_by_n[n].append(resp.mean(axis=0))         # (3,) frac significant per stim

        def _summarise(by_n):
            means, sems = [], []
            for n in args.n_trials_grid:
                arr = np.stack(by_n[n])            # (n_seeds, 3)
                overall = arr.mean(axis=1)          # (n_seeds,) mean-over-stimuli per seed
                means.append(float(overall.mean()))
                sems.append(float(overall.std(ddof=1) / np.sqrt(len(overall))) if len(overall) > 1 else 0.0)
            return means, sems

        old_mean, old_sem = _summarise(old_by_n)
        clu_mean, clu_sem = _summarise(clu_by_n)
        results[mt] = dict(
            n_seeds=len(seeds), n_trials_grid=args.n_trials_grid,
            old=dict(mean=old_mean, sem=old_sem),
            cluster=dict(mean=clu_mean, sem=clu_sem),
        )
        print(f"  old:     {[round(m,3) for m in old_mean]}")
        print(f"  cluster: {[round(m,3) for m in clu_mean]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
