#!/usr/bin/env python3
"""Cross-context (pre- vs post-reversal) stimulus decoding, using the
EXISTING cxval.analysis decoders (pairwise_decode / crosscontext_decode /
generalisation_matrix) -- no new rollouts or retraining needed, everything
required is already on disk in transfer/figure_data{,_reversal}/time_resolved.

"Fully recovered" post-reversal result (per chat): uses figure_data_reversal
as-is (the completed/converged reversal run), not a partially-adapted
mid-training snapshot -- that would need periodic-checkpoint retraining,
which doesn't currently exist (see terminal_rpe.py's docstring) and is a
separate follow-up if you want the "not fully converged" comparison later.

Per seed (not pooled across seeds -- avoids the decoder picking up seed
identity as a confound): build one act_dict spanning that seed's pre AND
post trials (context=0/1), run pairwise_decode (within-context) and
crosscontext_decode (train on one phase, test on the other) on it, then
average the resulting accuracy matrices across seeds. This matches how a
real multi-session decoding analysis would average across sessions.

Needs sklearn + scipy (not available in the sandbox this was drafted in).
Run in your cxval env:

    cd context-value-RNNs
    python3 unified_figures/analysis/run_decoding.py \
        --pre unified_figures/transfer/figure_data \
        --post unified_figures/transfer/figure_data_reversal \
        --out unified_figures/output/decoding

Also computes three new decoders (model-side FIG2-equivalent panels, see
analysis/context_stimidentity_decode.py's docstring for exactly what each
one decodes and why): per-stimulus context decode (draw_reversal_context_bar
analogue), pooled value decode (draw_reversal_value_bar / 'value_xor'
analogue), and pooled stimulus-identity decode (draw_reversal_stimidentity_bar
analogue). All written into the SAME output JSON alongside the existing
within/cross-context stimulus decoding.
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
import context_stimidentity_decode as CSD

VM = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)   # matches train_model.py's VM
VM_REV = VM[::-1].copy()                                  # matches train_reversal.py's VM_rev
VALUE_MATRIX = np.concatenate([VM, VM_REV], axis=1)       # (n_stim, n_ctx): col0=pre, col1=post
STIM_LABELS = ["0%", "50%", "100%"]                        # identity-anchored, fixed index


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre", default=str(_HERE.parent / "transfer" / "figure_data"))
    ap.add_argument("--post", default=str(_HERE.parent / "transfer" / "figure_data_reversal"))
    ap.add_argument("--out", default=str(_HERE.parent / "output" / "decoding"))
    ap.add_argument("--period", default="stim", choices=["stim", "reward"])
    ap.add_argument("--pooling", default="average", choices=["average", "pool"])
    ap.add_argument("--model-types", nargs="*",
                     default=["rl_only", "classif_rl", "classif_rl_readout_only"])
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    import cxval.analysis as A   # noqa: E402  (needs sklearn+scipy -- see docstring)

    pre_td = Path(args.pre) / "time_resolved"
    post_td = Path(args.post) / "time_resolved"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for mt in args.model_types:
        pre_files = AD.seed_files(pre_td, mt)
        post_files = AD.seed_files(post_td, mt)
        seeds = sorted(set(pre_files) & set(post_files))
        print(f"{mt}: {len(seeds)} seeds present in both pre and post")

        within_accs, cross_accs = [], []
        for seed in seeds:
            act_dict = AD.act_dict_one_seed_pre_post(pre_files[seed], post_files[seed])
            within = A.pairwise_decode(act_dict, period=args.period, pooling=args.pooling)
            cross = A.crosscontext_decode(act_dict, period=args.period, pooling=args.pooling)
            within_accs.append(within)
            cross_accs.append(cross)

        within_stack = np.stack(within_accs)                       # (n_seed, n_ctx, n_stim, n_stim)
        within_mean = np.nanmean(within_stack, axis=0)             # (n_ctx, n_stim, n_stim)
        within_sem = np.nanstd(within_stack, axis=0) / np.sqrt(max(within_stack.shape[0], 1))
        cross_mean = np.nanmean(np.stack(cross_accs), axis=0)     # (n_ctx, n_ctx, n_stim, n_stim)
        gm = A.generalisation_matrix(within_mean, cross_mean)     # (n_ctx, n_ctx)

        # -- new: context / value / stim-identity decoders, per seed then averaged --
        # IMPORTANT (see chat): stim-identity is NOT computed by pooling pre+post
        # into one CV split (context_stimidentity_decode.stimidentity_decode_pooled
        # is a documented dead end -- with only 4 tight clusters in 128d, a linear
        # decoder trivially separates ANY 2-vs-2 grouping of them, which gave
        # ceiling "identity" accuracy even for rl_only, flatly contradicting its
        # already-near-chance cross-context transfer for the SAME units). Instead
        # it's read directly off the crosscontext_decode cross_mean array already
        # computed above -- a genuine held-out-context test, exactly like every
        # other pair in the generalisation matrix.
        context_accs, value_accs, ctx_baseline_accs = [], [], []
        for seed in seeds:
            act_dict = AD.act_dict_one_seed_pre_post(pre_files[seed], post_files[seed])
            context_accs.append(CSD.context_decode_per_stim(act_dict, period=args.period, pooling=args.pooling))
            value_accs.append(CSD.value_decode_pooled(act_dict, period=args.period, pooling=args.pooling,
                                                       value_matrix=VALUE_MATRIX))
            ctx_baseline_accs.append(CSD.context_decode_baseline_control(act_dict))

        context_mean = np.nanmean(np.stack(context_accs), axis=0)   # (n_stim,)
        context_sem = np.nanstd(np.stack(context_accs), axis=0) / np.sqrt(len(context_accs))
        value_arr = np.asarray(value_accs, float)
        ctx_baseline_arr = np.asarray(ctx_baseline_accs, float)
        # per-seed stim-identity values (from each seed's OWN cross_accs entry,
        # not the already-seed-averaged cross_mean) so we get a real cross-seed
        # SEM, not just a single pooled number.
        stimid_per_seed = np.array([CSD.stimidentity_decode_from_cross(c, stim_pair=(0, 2))
                                    for c in cross_accs])
        stimid_val = float(np.nanmean(stimid_per_seed))
        stimid_sem = float(np.nanstd(stimid_per_seed) / np.sqrt(len(stimid_per_seed)))

        results[mt] = dict(
            n_seeds=len(seeds),
            within_mean_by_context=within_mean.tolist(),
            within_sem_by_context=within_sem.tolist(),
            cross_mean_by_pair=cross_mean.tolist(),   # (n_ctx,n_ctx,n_stim,n_stim), raw -- for re-deriving pairs like stimidentity
            generalisation_matrix=gm.tolist(),
            stim_labels=STIM_LABELS,
            context_decode=dict(mean=context_mean.tolist(), sem=context_sem.tolist()),
            context_decode_baseline_control=dict(
                mean=float(np.nanmean(ctx_baseline_arr)),
                sem=float(np.nanstd(ctx_baseline_arr) / np.sqrt(len(ctx_baseline_arr)))),
            value_decode=dict(mean=float(np.nanmean(value_arr)),
                              sem=float(np.nanstd(value_arr) / np.sqrt(len(value_arr)))),
            stimidentity_decode=dict(mean=stimid_val, sem=stimid_sem),
        )
        print(f"  generalisation matrix (rows=train ctx, cols=test ctx, pre=0/post=1):\n{gm}")
        print(f"  context decode per stim {STIM_LABELS}: {np.round(context_mean, 3)}  "
              f"(baseline-window control: {np.nanmean(ctx_baseline_arr):.3f} -- if these are close, "
              f"context_decode mostly reflects global drift, not cue-specific coding)")
        print(f"  value decode (0% vs 100%, within-context, averaged pre+post): {np.nanmean(value_arr):.3f}")
        print(f"  stim-identity decode (0% vs 100% cue, held-out-context, both directions): {stimid_val:.3f}")

    out_path = out_dir / f"crosscontext_decode_{args.period}_{args.pooling}.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
