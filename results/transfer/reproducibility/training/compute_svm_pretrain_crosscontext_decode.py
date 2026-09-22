#!/usr/bin/env python
"""Post-hoc: linear-SVM cross-context stimulus decode DURING pre-reversal
training, scored against the FINAL trained model's own representation --
the SVM-classifier counterpart to compute_pretrain_crosscontext_decode.py
(which uses a nearest-centroid classifier), and the pre-reversal-segment
counterpart to compute_svm_crosscontext_decode.py (which only ever covered
the post-reversal segment -- see that script's own docstring: "This only
covers the FORWARD (post-reversal) segment ... the backward ... segment
... isn't touched here"). This script is what fills that gap.

WHY A DIFFERENT REFERENCE POINT THAN THE POST-REVERSAL SVM SCRIPT: there is
no equivalent of "a frozen model that already exists before training
starts" for pre-reversal training itself -- by definition nothing exists
before the first checkpoint. So, following compute_pretrain_crosscontext_
decode.py's own convention, the reference here is instead the RUN'S OWN
FINAL pre-reversal model (model.pt), fit once (StandardScaler + LinearSVC
on trial-averaged stim_hidden -> stimulus label), and then every earlier
checkpoint from that same run is scored against that frozen fit *without
refitting*, exactly as the post-reversal script scores every post-reversal
checkpoint against ITS frozen pre-reversal reference. Because the reference
IS the final state of the very run being scored, this curve is necessarily
a representational-CONVERGENCE curve (rises toward the reference's own
~ceiling training accuracy as training progresses), not an independent
validation metric -- same caveat as the nearest-centroid version.

CHANCE LEVEL 1/3, classifier LinearSVC(max_iter=5000, dual='auto') +
StandardScaler -- same classifier family/hyperparameters as
compute_svm_crosscontext_decode.py, for a fair comparison across the
reversal point once both segments are plotted on the same axes.

REQUIRES: checkpoints/ under each run (only present in the checkpointed
pre-reversal rerun, model_runs_ckpt -- NOT model_runs_instrumented/
model_runs), plus torch + cxval + scikit-learn. Run ON YOUR MACHINE, not
from a sandbox without torch.

USAGE:
    # single run
    python compute_svm_pretrain_crosscontext_decode.py --run ../../model_runs_ckpt/rl_only/seed42

    # every run under a root (e.g. every model_type/seed* under model_runs_ckpt)
    python compute_svm_pretrain_crosscontext_decode.py --runs-root ../../model_runs_ckpt

Writes <run>/checkpoint_svm_crosscontext_decode.json:
    {"checkpoint_update": [...], "svm_crosscontext_decode": [...],
     "n_checkpoints": N, "chance": 0.3333333333333333,
     "classifier": "LinearSVC(max_iter=5000, dual='auto'), StandardScaler",
     "reference_train_acc": <float>,
     "reference": "final model.pt (this same run) -- representational-
                    convergence curve, not an independent validation metric"}

This is a companion to, and deliberately does not touch, the existing
nearest-centroid checkpoint_crosscontext_decode.json files in the same
directories -- both can coexist.
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cxval.vigour import infer_vigour  # noqa: E402
# reuse the exact state_dict -> model reconstruction train_reversal.py /
# compute_pretrain_crosscontext_decode.py already have, rather than
# duplicating it a third time.
from train_reversal import build_model  # noqa: E402


def _rollout_R_stim(model, value_matrix, cost, device="cpu"):
    acts, _ = infer_vigour(model, value_matrix, n_eval_episodes=4,
                           n_trials_per_episode=150, vigour_cost=cost, device=device)
    R = np.asarray(acts["stim_hidden"]).mean(1)
    stim = np.asarray(acts["stimulus"])
    return R, stim


def process_run(run_dir: Path, device="cpu", overwrite: bool = False) -> bool:
    run_dir = Path(run_dir)
    out_path = run_dir / "checkpoint_svm_crosscontext_decode.json"
    if out_path.exists() and not overwrite:
        print(f"  [skip] {out_path} exists (pass --overwrite to redo)")
        return False

    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        print(f"  (skip {run_dir}: no checkpoints/ -- this is only available "
              f"for the checkpointed rerun, model_runs_ckpt)")
        return False
    ckpt_files = sorted(glob.glob(str(ckpt_dir / "checkpoint_*.pt")))
    if not ckpt_files:
        print(f"  (skip {run_dir}: checkpoints/ exists but is empty)")
        return False

    cfg = json.loads((run_dir / "config.json").read_text())
    tr = cfg["train"]
    cost = tr["vigour_cost"]
    VM = np.asarray(cfg["value_matrix"], np.float32)

    # --- reference: this run's own FINAL pre-reversal model ---
    final_sd = torch.load(run_dir / "model.pt", map_location="cpu")
    final_model = build_model(final_sd)
    R_ref, stim_ref = _rollout_R_stim(final_model, VM, cost, device=device)

    scaler = StandardScaler().fit(R_ref)
    clf = LinearSVC(max_iter=5000, dual="auto").fit(scaler.transform(R_ref), stim_ref)
    train_acc = float(clf.score(scaler.transform(R_ref), stim_ref))

    updates, accs = [], []
    for f in ckpt_files:
        u = int(Path(f).stem.split("_")[-1])
        sd = torch.load(f, map_location="cpu")
        model = build_model(sd)
        R, stim = _rollout_R_stim(model, VM, cost, device=device)
        acc = float(clf.score(scaler.transform(R), stim))
        updates.append(u); accs.append(acc)

    order = np.argsort(updates)
    updates = [updates[i] for i in order]
    accs = [accs[i] for i in order]

    out = {
        "checkpoint_update": updates,
        "svm_crosscontext_decode": accs,
        "n_checkpoints": len(updates),
        "chance": 1 / 3,
        "classifier": "LinearSVC(max_iter=5000, dual='auto'), StandardScaler",
        "reference_train_acc": train_acc,
        "reference": "final model.pt (this same run) -- see module docstring: "
                     "this is a representational-convergence curve, not an "
                     "independent validation metric",
    }
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"  {run_dir}: {len(updates)} checkpoints, ref train_acc={train_acc:.3f}, "
          f"acc {accs[0]:.3f} (u={updates[0]}) -> {accs[-1]:.3f} (u={updates[-1]})")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="single run dir, e.g. ../../model_runs_ckpt/rl_only/seed42")
    g.add_argument("--runs-root", help="process every <root>/<model_type>/seed*/ run "
                                       "found under this directory (e.g. model_runs_ckpt)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.run:
        ok = process_run(Path(args.run), device=args.device, overwrite=args.overwrite)
        sys.exit(0 if ok else 1)

    root = Path(args.runs_root)
    run_dirs = sorted(p.parent for p in root.glob("*/seed*/model.pt"))
    if not run_dirs:
        print(f"No runs found under {root} (expected <root>/<model_type>/seed*/model.pt)")
        sys.exit(1)
    n_done = n_skip = 0
    for rd in run_dirs:
        if process_run(rd, device=args.device, overwrite=args.overwrite):
            n_done += 1
        else:
            n_skip += 1
    print(f"done. {n_done} run(s) processed, {n_skip} skipped.")


if __name__ == "__main__":
    main()
