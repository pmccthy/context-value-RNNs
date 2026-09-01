#!/usr/bin/env python
"""Post-hoc: cross-context stimulus decode accuracy DURING pre-reversal
training, computed from saved checkpoints against the FINAL trained model's
own representation.

Why this has to be post-hoc (can't be logged live like every other probe
metric): probe_crosscontext_decode (see train_reversal.py) works by testing
each probe point's stim_hidden representation against a FIXED reference
classifier -- nearest-centroid, built ONCE from a frozen rollout of the
model BEFORE any further training happens. For post-reversal training that
reference is the completed PRE-reversal model, which already exists when
reversal training starts, so it can be computed live in make_probe().

For PRE-reversal training itself there is no equivalent live option: the
natural reference for "how close is this checkpoint's representation to
where training eventually ends up" is the FINAL pre-reversal model -- which
by definition doesn't exist yet at any earlier point during that same
training run. So this script re-evaluates the pre-training run AFTER THE
FACT: it builds the reference centroids from the actual final model.pt,
then loads each saved checkpoint (--checkpoint-every / --checkpoint-fine-*
in train_model.py / run_all.sh) and classifies its stim_hidden
representation against those same centroids.

Interpretation: because the classifier's reference IS (very nearly) the
model's own end state, expect this curve to rise from near-chance (1/3) at
initialisation toward ~ceiling as training converges -- it's a self-
consistency / representational-convergence curve, not an independent
validation metric. That makes it the natural pre-reversal counterpart to
plot alongside probe_crosscontext_decode's post-reversal trajectory in the
"Pre->post accuracy" panel (panels/vigour_value.py's
draw_crosscontext_decode_vs_trials in unified_figures/), with the reversal
point marked where the two curves meet.

Requires the run to have been trained WITH checkpointing enabled
(--checkpoint-every > 0 in train_model.py / CHECKPOINT_EVERY in
run_all.sh) -- runs without a checkpoints/ subdirectory are skipped with a
clear message, not silently produce empty output.

Usage (single run):
    python compute_pretrain_crosscontext_decode.py --run model_runs/rl_only/seed42

Usage (batch, all seeds/model types under a root):
    python compute_pretrain_crosscontext_decode.py --runs-root model_runs

Writes <run>/checkpoint_crosscontext_decode.json:
    {"checkpoint_update": [...], "crosscontext_decode": [...],
     "n_checkpoints": N, "reference": "final model.pt (this same run)"}
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cxval.vigour import infer_vigour  # noqa: E402
# reuse the exact reference-classifier machinery train_reversal.py already
# has (frozen-centroid nearest-neighbour decode + state_dict -> model
# reconstruction) rather than duplicating it.
from train_reversal import build_model, _quick_crosscontext_decode  # noqa: E402


def _rollout_R_stim(model, value_matrix, cost, device="cpu"):
    acts, _ = infer_vigour(model, value_matrix, n_eval_episodes=4,
                           n_trials_per_episode=150, vigour_cost=cost, device=device)
    R = np.asarray(acts["stim_hidden"]).mean(1)
    stim = np.asarray(acts["stimulus"])
    return R, stim


def process_run(run_dir: Path, device="cpu"):
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        print(f"  (skip {run_dir}: no checkpoints/ -- retrain with "
              f"--checkpoint-every > 0 / CHECKPOINT_EVERY set)")
        return False
    ckpt_files = sorted(glob.glob(str(ckpt_dir / "checkpoint_*.pt")))
    if not ckpt_files:
        print(f"  (skip {run_dir}: checkpoints/ exists but is empty)")
        return False

    cfg = json.loads((run_dir / "config.json").read_text())
    tr = cfg["train"]
    cost = tr["vigour_cost"]
    VM = np.asarray(cfg["value_matrix"], np.float32)

    final_sd = torch.load(run_dir / "model.pt", map_location="cpu")
    final_model = build_model(final_sd)
    R_final, stim_final = _rollout_R_stim(final_model, VM, cost, device=device)
    centroids_final = np.stack([R_final[stim_final == s].mean(0) for s in range(3)])

    updates, accs = [], []
    for f in ckpt_files:
        u = int(Path(f).stem.split("_")[-1])
        sd = torch.load(f, map_location="cpu")
        model = build_model(sd)
        R, stim = _rollout_R_stim(model, VM, cost, device=device)
        acc = _quick_crosscontext_decode(R, stim, centroids_final)
        updates.append(u); accs.append(acc)

    order = np.argsort(updates)
    updates = [updates[i] for i in order]
    accs = [accs[i] for i in order]

    out = {
        "checkpoint_update": updates,
        "crosscontext_decode": accs,
        "n_checkpoints": len(updates),
        "reference": "final model.pt (this same run) -- see module docstring: "
                     "this is a representational-convergence curve, not an "
                     "independent validation metric",
    }
    (run_dir / "checkpoint_crosscontext_decode.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"  {run_dir}: {len(updates)} checkpoints, "
          f"acc {accs[0]:.3f} (u={updates[0]}) -> {accs[-1]:.3f} (u={updates[-1]})")
    return True


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="single run dir, e.g. model_runs/rl_only/seed42")
    g.add_argument("--runs-root", help="process every <root>/<model_type>/seed*/ run "
                                       "found under this directory (e.g. model_runs)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.run:
        ok = process_run(Path(args.run), device=args.device)
        sys.exit(0 if ok else 1)

    root = Path(args.runs_root)
    run_dirs = sorted(p.parent for p in root.glob("*/seed*/model.pt"))
    if not run_dirs:
        print(f"No runs found under {root} (expected <root>/<model_type>/seed*/model.pt)")
        sys.exit(1)
    n_done = n_skip = 0
    for rd in run_dirs:
        if process_run(rd, device=args.device):
            n_done += 1
        else:
            n_skip += 1
    print(f"done. {n_done} run(s) processed, {n_skip} skipped (no checkpoints).")


if __name__ == "__main__":
    main()
