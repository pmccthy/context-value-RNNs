#!/usr/bin/env python
"""Terminal (final-model) reward-prediction-error AND critic value estimate, for
already-COMPLETED reversal runs whose training didn't log these live -- no retraining.

Why "terminal" and not a real training-time curve: cxval.vigour.infer_rpe/infer_value
need the model's live weights at each point in training, but a run that only saved the
FINAL model per phase (model.pt, no periodic checkpoints) has no way to recover what
RPE/value looked like DURING learning after the fact. What IS recoverable without
retraining: repeatedly evaluate the frozen final pre-reversal model on the original
task, and the frozen final post-reversal model on the reversed task, and look at the
magnitude/spread at those two fixed points. There is no learning happening across these
repeated evaluations -- point-to-point variation across the "trials" axis is pure
sampling noise from the stochastic trial draws, NOT a training trend. Read the figure's
subtitle, which says this explicitly.

The value-estimate half of this exists to directly test the "critic collapse" idea:
whether a failed seed's low vigour persists because the critic has settled into an
accurate (self-consistent) LOW value prediction for the devalued/never-tried stimulus,
rather than an unresolved, still-surprised one -- i.e. whether V(s) tracks the observed
low reward rather than the counterfactual higher reward a better policy would earn.

For runs whose training DOES log real probe_rpe/probe_value already (anything trained
after infer_rpe/infer_value were wired into make_probe), you don't need this script for
that part -- seed_groups.py's key="rpe"/key="value" calls give you the genuine
training-time curve directly. This script is for filling in metrics that weren't logged
during a run that already finished, e.g. the critic value estimate for the
min_vigour=0.1 job (which logs real RPE, since that predates infer_value, but not real
value).

Writes synthetic per-seed history.json files (same schema _load_phase_seed expects, with
recovered_fraction carried over from the REAL post-reversal run so the recovered/failed
split matches the actual study) into --out-runs, then reuses seed_groups.fig_seed_curves
UNMODIFIED -- same house style, same rows=outcome-group / cols=model-class / STIM_COLORS
layout as seed_vigour_curves.

Usage:
  python code/terminal_rpe.py \
      --pre-runs results/model_runs --post-runs results/reversal_5000/model_runs_reversal \
      --out-runs results/terminal_rpe --out results/reversal_5000/figures_seed_groups
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, for cxval
from cxval.models import RNN
from cxval.vigour import VigourActorCritic, infer_rpe, infer_value

sys.path.insert(0, str(Path(__file__).resolve().parent))       # code/, for seed_groups etc.
try:
    import seed_groups as SG
except ModuleNotFoundError:
    import importlib
    SG = importlib.import_module("16_06_26_seed_groups")
F = SG.RA.F

VM = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)
VM_REV = VM[::-1].copy()


def build_model(sd, min_vigour=0.0):
    """Same reconstruction logic as train_reversal.py's build_model (architecture from
    weight shapes)."""
    H = sd["backbone.h2h.weight"].shape[0]
    obs = sd["backbone.input2h.weight"].shape[1]
    rf = sd["vigour_head.weight"].shape[1] / H
    aux = sd["stim_head.weight"].shape[0] if "stim_head.weight" in sd else 0
    ac = VigourActorCritic(RNN(input_size=obs, hidden_size=H, output_size=1,
                               recurrent_gain=0.9), action_std=0.05,
                           readout_fraction=rf, aux_n_stim=aux)
    ac.min_vigour = min_vigour
    ac.load_state_dict(sd); ac.eval()
    return ac


def terminal_metric_blocks(model, value_matrix, cost, infer_fn, n_blocks=10,
                           n_eval_episodes=6, n_trials_per_episode=150):
    """n_blocks independent deterministic-eval readings of infer_fn (different trial
    draws, SAME frozen weights throughout) -> (n_blocks, 3) array, one row per block,
    one column per stimulus. infer_fn is infer_rpe or infer_value."""
    out = np.zeros((n_blocks, 3), np.float32)
    for k in range(n_blocks):
        m = infer_fn(model, value_matrix, n_eval_episodes=n_eval_episodes,
                     n_trials_per_episode=n_trials_per_episode, vigour_cost=cost,
                     base_seed=30_000 + 977 * k)
        out[k] = [m[0], m[1], m[2]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-runs", default="results/model_runs")
    ap.add_argument("--post-runs", default="results/reversal_5000/model_runs_reversal")
    ap.add_argument("--out-runs", default="results/terminal_rpe",
                    help="where to write the synthetic per-seed history.json files")
    ap.add_argument("--out", default="results/reversal_5000/figures_seed_groups")
    ap.add_argument("--n-blocks", type=int, default=10)
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--style", default=str(F.DEFAULT_STYLE))
    ap.add_argument("--seed-min", type=int, default=0,
                    help="only compute seeds >= this (for splitting a long run "
                         "across multiple calls; writes are per-seed and idempotent)")
    ap.add_argument("--seed-max", type=int, default=10**9)
    ap.add_argument("--skip-plot", action="store_true",
                    help="just write synthetic history.json files, don't plot yet "
                         "(use for partial-seed-range batches)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if Path(args.style).exists():
        plt.style.use(args.style); print(f"style: {args.style}")
    else:
        print(f"** WARNING: style file not found at {args.style} — figures will use "
              f"default matplotlib styling, not the house style **")

    syn_pre = Path(args.out_runs) / "model_runs"
    syn_post = Path(args.out_runs) / "model_runs_reversal"

    n = 0
    for f in sorted(glob.glob(str(Path(args.post_runs) / "*" / "seed*" / "history.json"))):
        p = Path(f); mt = p.parent.parent.name; seed = int(p.parent.name[4:])
        if not (args.seed_min <= seed <= args.seed_max):
            continue
        post_run = p.parent
        pre_run = Path(args.pre_runs) / mt / f"seed{seed}"
        if not (pre_run / "model.pt").exists():
            continue
        if (syn_post / mt / f"seed{seed}" / "history.json").exists():
            continue          # already computed (safe to re-run this script incrementally)
        post_cfg = json.loads((post_run / "config.json").read_text())
        pre_cfg = json.loads((pre_run / "config.json").read_text())
        cost = pre_cfg["train"]["vigour_cost"]
        mv_pre = pre_cfg["train"].get("min_vigour", 0.0)
        mv_post = post_cfg["train"].get("min_vigour", 0.0)

        pre_sd = torch.load(pre_run / "model.pt", map_location="cpu")
        post_sd = torch.load(post_run / "model.pt", map_location="cpu")
        pre_model = build_model(pre_sd, min_vigour=mv_pre)
        post_model = build_model(post_sd, min_vigour=mv_post)

        rpe_pre = terminal_metric_blocks(pre_model, VM, cost, infer_rpe, n_blocks=args.n_blocks)
        rpe_post = terminal_metric_blocks(post_model, VM_REV, cost, infer_rpe, n_blocks=args.n_blocks)
        val_pre = terminal_metric_blocks(pre_model, VM, cost, infer_value, n_blocks=args.n_blocks)
        val_post = terminal_metric_blocks(post_model, VM_REV, cost, infer_value, n_blocks=args.n_blocks)

        hist = json.loads((post_run / "history.json").read_text())
        rf = hist.get("recovered_fraction")
        n_pre = pre_cfg["train"]["n_trials_per_episode"]
        n_post = hist.get("n_trials", n_pre)
        K = args.n_blocks

        (syn_pre / mt / f"seed{seed}").mkdir(parents=True, exist_ok=True)
        (syn_pre / mt / f"seed{seed}" / "history.json").write_text(json.dumps({
            "n_trials": n_pre, "total_updates": K,
            "probe_update": list(range(1, K + 1)),
            "probe_rpe": rpe_pre.tolist(),
            "probe_value": val_pre.tolist(),
        }))
        (syn_post / mt / f"seed{seed}").mkdir(parents=True, exist_ok=True)
        (syn_post / mt / f"seed{seed}" / "history.json").write_text(json.dumps({
            "n_trials": n_post, "total_updates": K,
            "probe_update": list(range(1, K + 1)),
            "probe_rpe": rpe_post.tolist(),
            "probe_value": val_post.tolist(),
            "recovered_fraction": rf,
        }))
        n += 1
        print(f"  {mt:26s} seed{seed}: rpe_post={rpe_post.mean(0)}  val_post={val_post.mean(0)}")
    print(f"terminal RPE/value: computed {n} seeds -> {args.out_runs}")
    if args.skip_plot:
        return

    table = SG.load_recovery_table(str(syn_post))
    keep, fail = SG.split_groups(table, args.thr)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    SG.fig_seed_curves(
        str(syn_pre), str(syn_post), keep, fail, out,
        key="rpe", ylabel="mean RPE (TD error)",
        title="Reward-prediction error (terminal model)",
        stem="terminal_rpe_curves",
        phase_note=("frozen FINAL model, repeated eval blocks (dashed = model swapped "
                    "pre→post) — NOT a training trajectory, point-to-point spread is "
                    "sampling noise only"))
    SG.fig_seed_curves(
        str(syn_pre), str(syn_post), keep, fail, out,
        key="value", ylabel="critic value estimate V(s)",
        title="Critic value prediction (terminal model)",
        stem="terminal_value_curves",
        phase_note=("frozen FINAL model, repeated eval blocks (dashed = model swapped "
                    "pre→post) — NOT a training trajectory, point-to-point spread is "
                    "sampling noise only"))


if __name__ == "__main__":
    main()
