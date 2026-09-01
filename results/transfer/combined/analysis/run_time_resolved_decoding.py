#!/usr/bin/env python3
"""Time-resolved decoding for the model side -- the direct analogue of
neuronal-representations/analysis/run_time_resolved_decoding.py, but reading
the RNN's own per-trial, per-timestep hidden states from
transfer/figure_data{,_reversal}/time_resolved/<model>_seed<NN>.npz (already
on disk -- see figures.py's dims_aligned meta: (trial, time, unit)) instead
of per-session dF/F.

For each seed present in BOTH the pre and post time_resolved directories
(same seed-selection convention as the sibling time-pooled script,
run_decoding.py: NO recovered-seed filtering -- decoding uses every seed
that has data, unlike the vigour/value-vs-trials trajectory panels, which
filter to recovered seeds so a flat failure trace doesn't dilute a real
learning curve), runs a 5-fold linear-SVM decode AT EVERY TIMESTEP
(StandardScaler fit per fold + LinearSVC, 5-fold CV, random_state=42 --
matching the real pipeline's LinearDecoder.time_resolved_decoder) for:

  stim_pair_pre / stim_pair_post   3 stimulus-pair decoders (0 vs 50, 0 vs
      100, 50 vs 100), pre-reversal and post-reversal trials respectively --
      the analogue of decoding.draw_reversal_stimpair_tr_pre/_post.
  context                          per-cue (0%, 50%, 100%) pre-vs-post
      (context) decode -- the analogue of decoding.draw_reversal_context_tr.

Deliberately does NOT compute a time-resolved 'value_xor' / 'stim_identity'
decoder (pooling trials from two different contexts into one k-fold CV
split): see analysis/context_stimidentity_decode.py's "DEAD END" docstring --
with this model's tight, near-deterministic per-condition clusters that pool
is trivially separable and gives a spurious near-ceiling accuracy, exactly
the artifact already found and fixed for the time-POOLED value/stim-identity
decoders (context_stimidentity_decode.stimidentity_decode_from_cross). This
also isn't a loss of parity with the real repo: its own FIG2 composite
(figures/compose.py:figure2) doesn't put a time-resolved value/stim-identity
panel in the grid either -- the bottom-right slot reuses the TIME-POOLED
stim_identity bar. This script's bottom-row output is meant to fill that same
slot with decoding.py's existing corrected draw_stimidentity_bar_single.

Needs sklearn (same requirement as run_decoding.py). Run in your cxval env:

    cd context-value-RNNs
    python3 unified_figures/analysis/run_time_resolved_decoding.py \
        --pre unified_figures/transfer/figure_data \
        --post unified_figures/transfer/figure_data_reversal \
        --out unified_figures/output/decoding_tr

For the 5k horizon: point --post at figure_data_reversal_5k and --out at
output/decoding_tr_5k (i.e. REV_TAG=_5k's paths), matching run_decoding.py's
own REV_TAG convention.

Runtime: ~500 LinearSVC fits per seed per model (3 pairs + 3 stims, x 11
timesteps, x 5 folds) on ~128-dim features -- a few seconds per seed, so a
few minutes total across all seeds and model types.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

N_FOLDS = 5
RANDOM_STATE = 42
MODEL_TYPES = ["rl_only", "classif_rl", "classif_rl_readout_only"]
PAIRS = [(0, 1), (0, 2), (1, 2)]
PAIR_KEY = {(0, 1): "0-1", (0, 2): "0-2", (1, 2): "1-2"}
STIM_LABELS = ["0%", "50%", "100%"]


def _seed_files(time_resolved_dir, model_type):
    pattern = str(Path(time_resolved_dir) / f"{model_type}_seed*.npz")
    files = sorted(glob.glob(pattern))
    return {int(re.search(r"_seed(\d+)\.npz$", f).group(1)): f for f in files}


def _load_npz(path):
    z = np.load(path, allow_pickle=True)
    aligned = np.asarray(z["aligned"], dtype=np.float32)   # (trial, time, unit)
    stim = np.asarray(z["stimulus"]).astype(int)
    bounds = np.asarray(z["bounds"], dtype=int)             # [n_iti, stim_ts, rew_ts]
    return aligned, stim, bounds


def _balanced_indices(stim, sA, sB, rng):
    """Equal-count trial indices for two stimulus classes (min_n each --
    matches the real pipeline's build_within_data concatenating min_n trials
    per contributing trial type)."""
    iA = np.where(stim == sA)[0]
    iB = np.where(stim == sB)[0]
    n = min(len(iA), len(iB))
    iA = rng.choice(iA, size=n, replace=False)
    iB = rng.choice(iB, size=n, replace=False)
    return iA, iB


def _decode_time_resolved(aligned, labels, n_folds=N_FOLDS, random_state=RANDOM_STATE):
    """aligned: (n_trial, n_time, n_unit); labels: (n_trial,) in {0,1}.
    Returns acc: (n_time,) -- 5-fold CV linear-SVM accuracy at EACH timestep
    independently (a fresh StandardScaler + LinearSVC per fold per timestep),
    matching the real pipeline's per-timepoint decode."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    n_time = aligned.shape[1]
    acc = np.full(n_time, np.nan)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for t in range(n_time):
        X = aligned[:, t, :]
        correct = 0
        for tr_idx, te_idx in skf.split(X, labels):
            sc = StandardScaler().fit(X[tr_idx])
            clf = LinearSVC(max_iter=2000, dual="auto").fit(sc.transform(X[tr_idx]), labels[tr_idx])
            correct += int((clf.predict(sc.transform(X[te_idx])) == labels[te_idx]).sum())
        acc[t] = correct / len(labels)
    return acc


def _stimpair_seed(aligned, stim, rng):
    """{(sA,sB): acc(n_time,)} for the 3 pairs, one phase, one seed."""
    out = {}
    for sA, sB in PAIRS:
        iA, iB = _balanced_indices(stim, sA, sB, rng)
        idx = np.concatenate([iA, iB])
        labels = np.array([0] * len(iA) + [1] * len(iB))
        out[(sA, sB)] = _decode_time_resolved(aligned[idx], labels)
    return out


def _context_seed(aligned_pre, stim_pre, aligned_post, stim_post, s, rng):
    """One stimulus `s`: pre-vs-post (context) decode, time-resolved, one seed."""
    iA = np.where(stim_pre == s)[0]
    iB = np.where(stim_post == s)[0]
    n = min(len(iA), len(iB))
    iA = rng.choice(iA, size=n, replace=False)
    iB = rng.choice(iB, size=n, replace=False)
    X = np.concatenate([aligned_pre[iA], aligned_post[iB]], axis=0)
    labels = np.array([0] * n + [1] * n)
    return _decode_time_resolved(X, labels)


def _agg(acc_list_dict):
    """{key: [acc(n_time,), ...] per seed} -> {str(key): {mean, sem}} across seeds."""
    out = {}
    for k, v in acc_list_dict.items():
        stacked = np.stack(v)   # (n_seed, n_time)
        out[str(k) if not isinstance(k, tuple) else PAIR_KEY[k]] = {
            "mean": np.nanmean(stacked, axis=0).tolist(),
            "sem": (np.nanstd(stacked, axis=0) / np.sqrt(max(stacked.shape[0], 1))).tolist(),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre", default=str(_HERE.parent / "transfer" / "figure_data"))
    ap.add_argument("--post", default=str(_HERE.parent / "transfer" / "figure_data_reversal"))
    ap.add_argument("--out", default=str(_HERE.parent / "output" / "decoding_tr"))
    ap.add_argument("--model-types", nargs="*", default=MODEL_TYPES)
    args = ap.parse_args()

    pre_td = Path(args.pre) / "time_resolved"
    post_td = Path(args.post) / "time_resolved"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for mt in args.model_types:
        pre_files = _seed_files(pre_td, mt)
        post_files = _seed_files(post_td, mt)
        seeds = sorted(set(pre_files) & set(post_files))
        print(f"{mt}: {len(seeds)} seeds present in both pre and post")
        if not seeds:
            continue

        stimpair_pre_acc = {p: [] for p in PAIRS}
        stimpair_post_acc = {p: [] for p in PAIRS}
        context_acc = {s: [] for s in range(3)}
        n_time = None
        bounds = None

        for seed in seeds:
            rng = np.random.default_rng(seed)
            aligned_pre, stim_pre, b = _load_npz(pre_files[seed])
            aligned_post, stim_post, _ = _load_npz(post_files[seed])
            bounds = b
            n_time = aligned_pre.shape[1]

            pre_res = _stimpair_seed(aligned_pre, stim_pre, rng)
            post_res = _stimpair_seed(aligned_post, stim_post, rng)
            for p in PAIRS:
                stimpair_pre_acc[p].append(pre_res[p])
                stimpair_post_acc[p].append(post_res[p])
            for s in range(3):
                context_acc[s].append(_context_seed(aligned_pre, stim_pre, aligned_post, stim_post, s, rng))
            print(f"  seed {seed} done", flush=True)

        results[mt] = {
            "n_seeds": len(seeds),
            "n_time": int(n_time),
            "bounds": [int(b) for b in bounds],   # [n_iti, stim_ts, rew_ts]
            "stim_pair_pre": _agg(stimpair_pre_acc),
            "stim_pair_post": _agg(stimpair_post_acc),
            "context": _agg(context_acc),
        }

    out_path = out_dir / "time_resolved_decode.json"
    out_path.write_text(json.dumps(results, indent=2))
    print("Time-resolved decoding complete ->", out_path)


if __name__ == "__main__":
    main()
