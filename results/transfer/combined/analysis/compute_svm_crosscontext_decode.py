#!/usr/bin/env python
"""Recompute the pre-reversal -> post-reversal cross-context stimulus decode
(Fig. rnn_reversal panels j-l / draw_crosscontext_decode_vs_trials in
panels/vigour_value.py) using the same cross-validated linear SVM pipeline
used by every other decoding analysis in this repo (cxval.analysis /
results/transfer/combined/analysis/run_decoding.py), instead of the fast
nearest-centroid probe (_quick_crosscontext_decode in
scripts/16_06_26_train_reversal.py / results/transfer/reproducibility/
training/train_reversal.py) computed live during training.

WHY: the live probe is a 3-way NEAREST-CENTROID classifier -- it collapses
each stimulus's reference activity to a single mean vector (centroid), then
classifies new activity by which of the 3 centroids it's closest to in
Euclidean distance. That's fast enough to call at every training checkpoint
without a dependency on sklearn, but it implicitly assumes each stimulus's
activity forms a roughly spherical, similarly-scaled cluster around its own
mean -- an assumption a linear SVM doesn't need (it finds a proper separating
hyperplane and can weight dimensions unevenly). This script redoes the same
comparison with a LinearSVC instead, to check whether that assumption was
costing decodability anywhere in the curve.

CHANCE LEVEL IS UNCHANGED AT 1/3: this is still a 3-way classification of
stimulus identity (0%/50%/100%), balanced by construction (the reference/eval
rollout draws each stimulus equally often via generate_batch). Only the
classifier changes; the number of classes doesn't.

METHODOLOGY (mirrors the original nearest-centroid probe as closely as
possible, so the two are a fair apples-to-apples comparison):
  - Reference set: one deterministic rollout of the FROZEN pre-reversal model
    (model_init.pt) on the ORIGINAL (pre-reversal) value matrix -- the exact
    same infer_vigour(..., n_eval_episodes=4, n_trials_per_episode=150) call,
    with the same default base_seed=10_000, that train_reversal.py's main()
    uses to build centroids_pre. A StandardScaler + LinearSVC are fit on
    this set (trial-averaged stim_hidden -> stimulus label), instead of
    collapsing it to 3 centroids.
  - Evaluation: for every saved checkpoint under <seed_dir>/checkpoints/
    (only present for reversal runs launched with --checkpoint-every > 0,
    e.g. the reversal_5000_instrumented run set), reload that checkpoint's
    weights, run the SAME infer_vigour call (same base_seed=10_000, so it's
    the identical fixed trial battery every time -- only the model's weights
    differ), and score the frozen SVM's accuracy on that rollout.
  - This only covers the FORWARD (post-reversal) segment of the two-part
    curve panels j-l actually plot (the backward, pre-reversal-checkpoint
    segment comes from a separate reversal_5000_ckpt run set via
    compute_pretrain_crosscontext_decode.py / _load_seed_checkpoint_crosscontext
    and isn't touched here -- that segment is described in the code as "a
    representational-convergence curve, not an independent validation
    metric," so the forward segment is the one worth re-deriving first).

REQUIRES: the same Python environment used to run
results/transfer/reproducibility/training/train_reversal.py (torch + cxval
installed, e.g. `pip install -e .` from the repo root) plus scikit-learn.
Run this ON YOUR MACHINE, not from a sandbox that doesn't have torch.

USAGE:
  # one seed, to sanity-check timing/output before committing to a full run
  python compute_svm_crosscontext_decode.py --model-type classif_rl --seed 42

  # every seed of one model type
  python compute_svm_crosscontext_decode.py --model-type classif_rl

  # everything (all 3 model types x however many seeds have checkpoints/)
  python compute_svm_crosscontext_decode.py --all

OUTPUT: writes <reversal-dir>/<model_type>/seed<N>/svm_crosscontext_decode.json
per seed, e.g.:
  {
    "model_type": "classif_rl", "seed": 42, "chance": 0.3333333333333333,
    "classifier": "LinearSVC(max_iter=5000, dual='auto'), StandardScaler",
    "reference_train_acc": 0.97,
    "results": [{"update": 1, "svm_crosscontext_decode": 0.31}, ...]
  }
This uses the same "update" key as history.json's own probe records (see
_load_seed_histories / _stack_mean_sem in panels/vigour_value.py), so once
we've looked at the numbers we can wire this into
draw_crosscontext_decode_vs_trials as a second line on the same axes rather
than a from-scratch plot.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from cxval.models import RNN
from cxval.vigour import VigourActorCritic, infer_vigour

REPO_ROOT = Path(__file__).resolve().parents[4]  # .../context-value-RNNs, from results/transfer/combined/analysis/<this file>; adjust if you move this script
DEFAULT_REVERSAL_DIR = REPO_ROOT / "results/transfer/reversal/reversal_5000_instrumented"
MODEL_TYPES = ["rl_only", "classif_rl", "classif_rl_readout_only"]


def build_model(sd: dict) -> VigourActorCritic:
    """Reconstruct a VigourActorCritic from a state_dict (architecture read off
    the tensor shapes) -- identical in spirit to train_reversal.py's own
    build_model(), duplicated here so this script has no import-path
    dependency on the training scripts directory."""
    H = sd["backbone.h2h.weight"].shape[0]
    obs = sd["backbone.input2h.weight"].shape[1]
    rf = sd["vigour_head.weight"].shape[1] / H
    aux = sd["stim_head.weight"].shape[0] if "stim_head.weight" in sd else 0
    ac = VigourActorCritic(
        RNN(input_size=obs, hidden_size=H, output_size=1, recurrent_gain=0.9),
        action_std=0.05, readout_fraction=rf, aux_n_stim=aux,
    )
    ac.load_state_dict(sd)
    ac.eval()
    return ac


def rollout_features(model: VigourActorCritic, value_matrix: np.ndarray, cost: float):
    """Same rollout convention as the original probe/reference (infer_vigour,
    default base_seed=10_000, n_eval_episodes=4, n_trials_per_episode=150):
    trial-averaged stim_hidden (one row per trial) + stimulus label per trial."""
    acts, _ = infer_vigour(
        model, value_matrix, n_eval_episodes=4, n_trials_per_episode=150, vigour_cost=cost,
    )
    R = np.asarray(acts["stim_hidden"]).mean(1)  # (n_trials, H): mean over stim-epoch timesteps
    stim = np.asarray(acts["stimulus"])
    return R, stim


def process_seed(model_type: str, seed: int, reversal_dir: Path, overwrite: bool = False) -> None:
    seed_dir = reversal_dir / model_type / f"seed{seed}"
    ckpt_dir = seed_dir / "checkpoints"
    out_path = seed_dir / "svm_crosscontext_decode.json"

    if not seed_dir.exists():
        print(f"  [skip] {seed_dir} does not exist")
        return
    if out_path.exists() and not overwrite:
        print(f"  [skip] {out_path} exists (pass --overwrite to redo)")
        return
    if not ckpt_dir.exists():
        print(f"  [skip] no checkpoints/ under {seed_dir} "
              f"(this seed's reversal run wasn't launched with --checkpoint-every > 0)")
        return

    cfg = json.loads((seed_dir / "config.json").read_text())
    tr = cfg["train"]
    VM_rev = np.asarray(cfg["value_matrix"], np.float32)  # already the REVERSED matrix
    VM_orig = VM_rev[::-1].copy()  # undo the swap to get the ORIGINAL (pre-reversal) matrix
    cost = tr["vigour_cost"]

    # --- reference: frozen pre-reversal model (model_init.pt), ORIGINAL task ---
    sd_pre = torch.load(seed_dir / "model_init.pt", map_location="cpu")
    model_pre = build_model(sd_pre)
    R_ref, stim_ref = rollout_features(model_pre, VM_orig, cost)

    scaler = StandardScaler().fit(R_ref)
    clf = LinearSVC(max_iter=5000, dual="auto").fit(scaler.transform(R_ref), stim_ref)
    train_acc = float(clf.score(scaler.transform(R_ref), stim_ref))
    print(f"  {model_type}/seed{seed}: reference fit, train acc={train_acc:.3f} "
          f"(n={len(stim_ref)} trials, chance=1/3)")

    # --- evaluate at every saved post-reversal checkpoint ---
    ckpts = sorted(ckpt_dir.glob("checkpoint_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        print(f"  [skip] {ckpt_dir} exists but has no checkpoint_*.pt files")
        return

    results = []
    t0 = time.time()
    for i, ckpt_path in enumerate(ckpts):
        update = int(ckpt_path.stem.split("_")[1])
        sd = torch.load(ckpt_path, map_location="cpu")
        model = build_model(sd)
        R, stim = rollout_features(model, VM_rev, cost)
        acc = float(clf.score(scaler.transform(R), stim))
        results.append({"update": update, "svm_crosscontext_decode": acc})
        if (i + 1) % 10 == 0 or i == len(ckpts) - 1:
            elapsed = time.time() - t0
            print(f"    checkpoint {i + 1}/{len(ckpts)} (update {update}): "
                  f"acc={acc:.3f}  [{elapsed:.1f}s elapsed]")

    out_path.write_text(json.dumps({
        "model_type": model_type,
        "seed": seed,
        "chance": 1 / 3,
        "classifier": "LinearSVC(max_iter=5000, dual='auto'), StandardScaler",
        "reference_train_acc": train_acc,
        "results": results,
    }, indent=2) + "\n")
    print(f"  -> wrote {out_path} ({len(results)} checkpoints, {time.time() - t0:.1f}s total)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-type", choices=MODEL_TYPES)
    ap.add_argument("--seed", type=int, help="single seed; omit to run every seed found for --model-type")
    ap.add_argument("--all", action="store_true", help="run every model type x every seed found")
    ap.add_argument("--reversal-dir", type=Path, default=DEFAULT_REVERSAL_DIR)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.all:
        targets = [
            (mt, int(p.name.replace("seed", "")))
            for mt in MODEL_TYPES
            for p in sorted((args.reversal_dir / mt).glob("seed*"))
        ]
    elif args.model_type and args.seed is not None:
        targets = [(args.model_type, args.seed)]
    elif args.model_type:
        targets = [
            (args.model_type, int(p.name.replace("seed", "")))
            for p in sorted((args.reversal_dir / args.model_type).glob("seed*"))
        ]
    else:
        ap.error("pass --all, or --model-type, or --model-type --seed")
        return

    print(f"Processing {len(targets)} (model_type, seed) run(s) from {args.reversal_dir}...")
    for mt, seed in targets:
        process_seed(mt, seed, args.reversal_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
