#!/usr/bin/env python
"""Train one model of the final 3s1c continuous-vigour family and save everything
needed to reconstruct and document the run.

Three model types (selected with --model-type):

  classif_rl                RL + auxiliary stimulus-classification head (aux_coef=1).
  rl_only                   matched RL-only baseline (aux_coef=0).
  classif_rl_readout_only   classif_rl, but the RL actor/critic gradients are stopped
                            at the readout (detach_readout=True) so the hidden units
                            are shaped ONLY by the classification (+ activity) losses;
                            the vigour/value heads are linear readouts trained off
                            (not into) that representation.

Saved to --out:
  model.pt        trained VigourActorCritic state_dict
  model_init.pt   the model's initial (pre-training) state_dict
  config.json     EVERY effective parameter (training, task timing, model type) —
                  the full, machine-readable run config
  meta.txt        one-line human summary (back-compat)

Usage:
  python scripts/16_06_26_train_model.py --model-type classif_rl --seed 42 \
         --out results/.../classif_rl/seed42
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cxval.vigour import train_vigour, infer_vigour, infer_value, vigour_metrics
from cxval.analysis import responsive_proportions_ttest



def _quick_stim_decode(R, stim, rng_seed=0):
    """Fast, dependency-free (numpy only) 3-way nearest-centroid stimulus
    decode accuracy from the time-averaged stim_hidden R (n_trials, H) that
    make_probe's caller already computed -- split each stimulus's trials in
    half, build centroids from one half, classify the other half by nearest
    centroid. Costs NOTHING extra (reuses the same infer_vigour rollout
    already run for probe_vigour/pop_activity, no new forward passes), so
    it's cheap enough to log at every probe point -- giving a genuine
    update-resolved decodability curve through training (and, in
    train_reversal.py, through the reversal itself), not just pre/post
    endpoints. Less rigorous than cxval.analysis.pairwise_decode's
    cross-validated linear SVM (that needs sklearn, this needs nothing) --
    good enough to see whether/when decodability dips and recovers."""
    rng = np.random.default_rng(rng_seed)
    train_idx, test_idx = [], []
    for s in range(3):
        idx = np.where(stim == s)[0]
        rng.shuffle(idx)
        half = len(idx) // 2
        if half < 1:
            return float("nan")
        train_idx.append(idx[:half]); test_idx.append(idx[half:])
    train_idx = np.concatenate(train_idx); test_idx = np.concatenate(test_idx)
    centroids = np.stack([R[train_idx][stim[train_idx] == s].mean(0) for s in range(3)])
    d = np.linalg.norm(R[test_idx][:, None, :] - centroids[None, :, :], axis=2)
    pred = d.argmin(axis=1)
    return float((pred == stim[test_idx]).mean())

def make_probe(value_matrix, cost):
    """Cheap deterministic probe (per-stim vigour, population activity, responder fraction)
    logged periodically during training to trace the time course."""
    def probe(model):
        acts, vmean = infer_vigour(model, value_matrix, n_eval_episodes=4,
                                   n_trials_per_episode=150, vigour_cost=cost)
        R = np.asarray(acts["stim_hidden"]).mean(1); stim = np.asarray(acts["stimulus"])
        tuning = np.stack([R[stim == s].mean(0) for s in range(3)])
        rp = responsive_proportions_ttest(acts, period="stim")
        # genuine trial-resolved critic value estimate V(s), same rollout cost class as
        # infer_vigour (one extra forward-pass rollout) -- NOT terminal_rpe.py's frozen-
        # model reconstruction, a real point on the training-time curve.
        vval = infer_value(model, value_matrix, n_eval_episodes=4,
                           n_trials_per_episode=150, vigour_cost=cost)
        return {"vigour": [float(vmean[s]) for s in range(3)],
                "pop_activity": tuning.mean(1).astype(float).tolist(),
                "frac_responsive": np.asarray(rp["frac_per_stim"], float).tolist(),
                "value": [float(vval[s]) for s in range(3)],
                "stim_decode": _quick_stim_decode(R, stim)}
    return probe

VM = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)     # 0% / 50% / 100% stimuli

# model type -> the two switches that distinguish the arms (+ a human label)
MODEL_TYPES = {
    "classif_rl":              dict(aux_coef=1.0, detach_readout=False,
                                    label="classification + RL"),
    "rl_only":                 dict(aux_coef=0.0, detach_readout=False,
                                    label="RL only"),
    "classif_rl_readout_only": dict(aux_coef=1.0, detach_readout=True,
                                    label="classification + RL (readout-only RL)"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", required=True, choices=list(MODEL_TYPES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    # final-config knobs (defaults ARE the final config; expose for sweeps)
    ap.add_argument("--n-trials", type=int, default=2500)
    ap.add_argument("--cost", type=float, default=0.8)
    ap.add_argument("--init", type=float, default=0.02)
    ap.add_argument("--readout-fraction", type=float, default=0.5)
    ap.add_argument("--action-std", type=float, default=0.05)
    ap.add_argument("--aux-at", choices=["reward", "stim"], default="reward")
    ap.add_argument("--activity-coef", type=float, default=2.0)
    ap.add_argument("--activity-at", choices=["all", "iti"], default="iti")
    ap.add_argument("--probe-every", type=int, default=0,
                    help="log per-stimulus metrics every N updates (0=off); enables the "
                         "combined pre->post reversal time course")
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="save a full model checkpoint every N updates under "
                         "<out>/checkpoints/ (0 to disable, the default -- these "
                         "are cheap, ~72KB each, but off by default so existing "
                         "invocations are unaffected). The very first update is "
                         "always checkpointed in addition to the regular interval "
                         "whenever this is > 0. Needed to later compute a genuine "
                         "trials-resolved cross-context decode trajectory DURING "
                         "pre-training (see compute_pretrain_crosscontext_decode.py) "
                         "-- the frozen reference classifier is only known once "
                         "training finishes, so that script re-evaluates each saved "
                         "checkpoint's activations against the FINAL model's "
                         "centroids after the fact; this flag just makes sure the "
                         "checkpoints exist to re-evaluate.")
    ap.add_argument("--checkpoint-fine-until", type=int, default=None,
                    help="checkpoint at --checkpoint-fine-every resolution for "
                         "updates 1..N, then fall back to --checkpoint-every. "
                         "Defaults to --checkpoint-every itself (i.e. fills in the "
                         "whole first coarse interval at fine resolution); pass 0 "
                         "to disable fine sampling. (Less critical here than in "
                         "train_reversal.py -- there's no single 'sudden change' "
                         "instant during ordinary pre-training -- but kept for the "
                         "same checkpoint-density control.)")
    ap.add_argument("--checkpoint-fine-every", type=int, default=1,
                    help="fine-window checkpoint interval (default: every update).")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    spec = MODEL_TYPES[args.model_type]

    # The single source of truth for the training call AND the saved config, so the
    # two can never drift apart.
    train_kwargs = dict(
        batch_size=32, n_trials_per_episode=args.n_trials, hidden_size=128,
        vigour_cost=args.cost, cost_type="quadratic", reward_fa=0.0, reward_lick=1.0,
        base_seed=args.seed, lr=5e-4, gamma=0.9, value_coef=0.5,
        action_std=args.action_std, readout_fraction=args.readout_fraction,
        init_scale=args.init, recurrent_gain=0.9, grad_clip=1.0, bptt_len=40,
        stim_timesteps=5, reward_timesteps=3, iti_timesteps=(3, 8),
        readout_mode="linear", policy="score",
        aux_coef=spec["aux_coef"], aux_at=args.aux_at,
        activity_coef=args.activity_coef, activity_at=args.activity_at,
        nonneg_coef=0.0, detach_readout=spec["detach_readout"],
    )
    checkpoint_dir = (out / "checkpoints") if args.checkpoint_every > 0 else None
    fine_until = args.checkpoint_fine_until
    if fine_until is None:
        fine_until = args.checkpoint_every   # default: fill the first coarse interval
    o = train_vigour(VM, device=args.device, probe_every=args.probe_every,
                     probe_fn=make_probe(VM, args.cost) if args.probe_every else None,
                     checkpoint_dir=checkpoint_dir, checkpoint_every=args.checkpoint_every,
                     checkpoint_fine_until=fine_until,
                     checkpoint_fine_every=args.checkpoint_fine_every,
                     **train_kwargs)

    torch.save(o["model"].state_dict(), out / "model.pt")
    torch.save(o["init_state_dict"], out / "model_init.pt")
    if args.probe_every:                                   # time course for the reversal plots
        (out / "history.json").write_text(json.dumps({
            "phase": "original", "n_trials": args.n_trials,
            "total_updates": len(o["history"].get("update", [])),
            "stim_labels": ["0%", "50%", "100%"],
            "probe_update": o["history"].get("probe_update", []),
            "probe_vigour": o["history"].get("probe_vigour", []),
            "probe_pop_activity": o["history"].get("probe_pop_activity", []),
            "probe_frac_responsive": o["history"].get("probe_frac_responsive", []),
            "probe_value": o["history"].get("probe_value", []),
            "probe_stim_decode": o["history"].get("probe_stim_decode", []),
        }, indent=2) + "\n")

    config = dict(
        model_type=args.model_type, label=spec["label"], seed=args.seed,
        diverged=bool(o["diverged"]),
        value_matrix=VM.tolist(), n_stim=int(VM.shape[0]),
        train=dict(train_kwargs, iti_timesteps=list(train_kwargs["iti_timesteps"])),
    )
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "meta.txt").write_text(
        f"model_type={args.model_type} aux_coef={spec['aux_coef']} "
        f"detach_readout={spec['detach_readout']} aux_at={args.aux_at} "
        f"activity_coef={args.activity_coef} activity_at={args.activity_at} "
        f"cost={args.cost} init={args.init} readout_fraction={args.readout_fraction} "
        f"action_std={args.action_std} seed={args.seed} n_trials={args.n_trials}\n")

    acts, vmean = infer_vigour(o["model"], VM, n_eval_episodes=8, n_trials_per_episode=200,
                               vigour_cost=args.cost, device=args.device)
    m = vigour_metrics(acts, vmean, VM)
    print(f"[{args.model_type} seed{args.seed}] diverged={int(o['diverged'])}  "
          f"vigour={m['vig_low']:.2f}/{m['vig_mid']:.2f}/{m['vig_high']:.2f} "
          f"act_ok={int(m['activity_ok'])} resp_ok={int(m['sel_ok'])}")


if __name__ == "__main__":
    main()
