#!/usr/bin/env python
"""Continue-train a trained model on a REVERSAL of the task and record how it recovers.

The reversal swaps the 0% and 100% reward contingencies (value_matrix [0, .5, 1] ->
[1, .5, 0]); the 50% stimulus and ALL stimulus INPUTS are unchanged — only the reward
mapping flips. The model is warm-started from its trained weights and continue-trained
with a fresh optimizer; the auxiliary classification loss stays on (stimulus identity is
unchanged by a contingency reversal).

Reads  <run>/config.json + <run>/model.pt  and writes to --out:
  model.pt       the post-reversal model
  model_init.pt  the pre-reversal model (the starting point)
  config.json    reversal config (reversed value_matrix + the source run + train kwargs)
  history.json   recovery learning curve on the reversed task (updates, mean_reward) plus
                 pre_reversal_reward (trained model on the ORIGINAL task) and
                 post_reversal_reward (reversed model on the REVERSED task); their ratio
                 is the fraction of pre-reversal performance recovered.

Usage:
  python scripts/16_06_26_train_reversal.py --run model_runs/classif_rl/seed42 \
         --out model_runs_reversal/classif_rl/seed42
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cxval.models import RNN
from cxval.batched import generate_batch
from cxval.vigour import (VigourActorCritic, BatchedVigourEnv, train_vigour, infer_vigour,
                         infer_rpe, infer_value)
from cxval.analysis import responsive_proportions_ttest


def build_model(sd, min_vigour=0.0, squash_width=None):
    """Reconstruct a VigourActorCritic from a state_dict (architecture from shapes).
    `min_vigour`/`squash_width` MUST match what the model was actually trained with
    (from its config.json) — build_model is only used for deterministic squash()-based
    eval (eval_reward), and a mismatched floor/squash there would silently corrupt the
    pre/post-reversal reward comparison (recovered_fraction)."""
    H = sd["backbone.h2h.weight"].shape[0]
    obs = sd["backbone.input2h.weight"].shape[1]
    rf = sd["vigour_head.weight"].shape[1] / H
    aux = sd["stim_head.weight"].shape[0] if "stim_head.weight" in sd else 0
    ac = VigourActorCritic(RNN(input_size=obs, hidden_size=H, output_size=1,
                               recurrent_gain=0.9), action_std=0.05,
                           readout_fraction=rf, aux_n_stim=aux)
    ac.min_vigour = min_vigour
    ac.squash_width = squash_width
    ac.load_state_dict(sd); ac.eval()
    return ac


def make_probe(value_matrix, cost):
    """A cheap deterministic probe: per-stimulus vigour, population activity and responder
    fraction, logged periodically during training to trace the reversal time course."""
    def probe(model):
        acts, vmean = infer_vigour(model, value_matrix, n_eval_episodes=4,
                                   n_trials_per_episode=150, vigour_cost=cost)
        R = np.asarray(acts["stim_hidden"]).mean(1); stim = np.asarray(acts["stimulus"])
        tuning = np.stack([R[stim == s].mean(0) for s in range(3)])
        rp = responsive_proportions_ttest(acts, period="stim")
        rpe = infer_rpe(model, value_matrix, n_eval_episodes=4, n_trials_per_episode=150,
                        vigour_cost=cost)
        val = infer_value(model, value_matrix, n_eval_episodes=4, n_trials_per_episode=150,
                          vigour_cost=cost)
        return {"vigour": [float(vmean[s]) for s in range(3)],
                "pop_activity": tuning.mean(1).astype(float).tolist(),
                "frac_responsive": np.asarray(rp["frac_per_stim"], float).tolist(),
                "rpe": [float(rpe[s]) for s in range(3)],
                "value": [float(val[s]) for s in range(3)]}
    return probe


@torch.no_grad()
def eval_reward(model, value_matrix, cost, n_ep=8, n_trials=200, base_seed=20_000):
    """Mean reward per active timestep for `model` on task `value_matrix` (matches the
    training curve's reward definition)."""
    sb, rbv, abv, _ = generate_batch(value_matrix, n_trials, n_ep, base_seed)
    env = BatchedVigourEnv(sb, rbv, abv, vigour_cost=cost)
    obs = torch.as_tensor(env.reset(), dtype=torch.float32); hidden = None
    total, nact, done = 0.0, 0.0, False
    while not done:
        mean, _, hidden = model.step(obs, hidden)
        v = model.squash(mean)
        o, r, done, info = env.step(v.cpu().numpy())
        m = np.asarray(info["active"], float); total += float((r * m).sum()); nact += float(m.sum())
        obs = torch.as_tensor(o, dtype=torch.float32)
    return total / max(nact, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="trained run dir (has config.json + model.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-trials", type=int, default=2500, help="reversal continue-training length")
    ap.add_argument("--probe-every", type=int, default=20,
                    help="log per-stimulus metrics every N updates (0 to disable)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reward-scale", type=float, nargs=3, default=None, metavar=("S0", "S50", "S100"),
                    help="CAUSAL INTERVENTION: multiply the realized reward (payoff net of "
                         "vigour cost) at exactly the trial-by-trial instants each stimulus's "
                         "reward is delivered on the REVERSED task -- e.g. `--reward-scale "
                         "2.0 1.0 1.0` doubles the realized reward (and hence the RPE/advantage "
                         "it feeds into) for the 0%%->100%% stimulus specifically, while the "
                         "100%%->0%% and 50%%->50%% stimuli, and every OTHER timestep for every "
                         "stimulus (ITI/stim-window vigour cost, etc.), are untouched. Use this "
                         "to test whether the SIZE of the RPE signal for a given direction "
                         "causally sets its relearning speed: e.g. does 2x-ing the 0->100 "
                         "reward shrink the observed speed gap vs. the 100->0 direction, and "
                         "does 0.5x-ing the 100->0 reward widen it? Order matches value_matrix "
                         "AFTER the reversal, i.e. [0%%-stim's scale, 50%%-stim's, 100%%-stim's] "
                         "-- since VM_rev = [1,.5,0], index 0 here is the ORIGINALLY-0%%, "
                         "NOW-100%%-valued stimulus (the slow 'newly valuable' direction), and "
                         "index 2 is the ORIGINALLY-100%%, NOW-0%%-valued one (the fast "
                         "'newly worthless' direction) -- default None = no scaling (1,1,1).")
    ap.add_argument("--track-gradients", action="store_true",
                    help="log, every update: per-parameter-group gradient norms (backbone / "
                         "vigour_head / value_head / stim_head, taken PRE grad-clip), the "
                         "scalar policy_loss/value_loss values, and an analytic per-stimulus "
                         "REINFORCE score (exact d(-logpi*A)/d(mean), no extra backward pass) "
                         "-- see track_gradients in cxval.vigour.train_vigour for the exact "
                         "definitions.")
    args = ap.parse_args()
    run = Path(args.run); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((run / "config.json").read_text())
    tr = dict(cfg["train"])                                  # original training kwargs
    tr["iti_timesteps"] = tuple(tr["iti_timesteps"])
    sd = torch.load(run / "model.pt", map_location="cpu")

    VM = np.asarray(cfg["value_matrix"], np.float32)         # [[0],[.5],[1]]
    VM_rev = VM[::-1].copy()                                 # [[1],[.5],[0]]  (swap 0% <-> 100%)
    cost = tr["vigour_cost"]

    min_vigour = tr.get("min_vigour", 0.0)                   # back-compat: old runs lack this key
    squash_width = tr.get("squash_width")                    # back-compat: old runs lack this key
    pre_reward = eval_reward(build_model(sd, min_vigour=min_vigour, squash_width=squash_width),
                             VM, cost)                        # ORIGINAL task

    # continue-train on the reversed task, warm-started from the trained weights
    tr["n_trials_per_episode"] = args.n_trials
    tr["reward_scale_by_stim"] = args.reward_scale
    tr["track_gradients"] = args.track_gradients
    o = train_vigour(VM_rev, device=args.device, init_model=sd,
                     probe_every=args.probe_every, probe_fn=make_probe(VM_rev, cost), **tr)

    post_reward = eval_reward(o["model"].cpu().eval(), VM_rev, cost)  # reversed model, reversed task

    torch.save(o["model"].state_dict(), out / "model.pt")
    torch.save(sd, out / "model_init.pt")                   # pre-reversal weights
    (out / "config.json").write_text(json.dumps({
        "model_type": cfg["model_type"], "label": cfg["label"], "seed": cfg["seed"],
        "phase": "reversal", "source_run": str(run),
        "value_matrix": VM_rev.tolist(), "reversal": "swap 0% <-> 100% (50% fixed)",
        "diverged": bool(o["diverged"]), "train": dict(tr, iti_timesteps=list(tr["iti_timesteps"])),
    }, indent=2) + "\n")
    (out / "history.json").write_text(json.dumps({
        "update": o["history"].get("update", []),
        "mean_reward": o["history"].get("mean_reward", []),
        "pre_reversal_reward": pre_reward,
        "post_reversal_reward": post_reward,
        "recovered_fraction": (post_reward / pre_reward) if pre_reward > 1e-9 else float("nan"),
        "stim_labels": ["0%", "50%", "100%"],
        "n_trials": args.n_trials, "total_updates": len(o["history"].get("update", [])),
        "probe_update": o["history"].get("probe_update", []),
        "probe_vigour": o["history"].get("probe_vigour", []),           # each: [v0, v50, v100]
        "probe_pop_activity": o["history"].get("probe_pop_activity", []),
        "probe_frac_responsive": o["history"].get("probe_frac_responsive", []),
        "probe_rpe": o["history"].get("probe_rpe", []),                 # each: [r0, r50, r100]
        "probe_value": o["history"].get("probe_value", []),             # each: [v0, v50, v100]
        "reward_scale_by_stim": args.reward_scale,
        "grad_norm": o["history"].get("grad_norm", []),
        "grad_norm_backbone": o["history"].get("grad_norm_backbone", []),
        "grad_norm_vigour_head": o["history"].get("grad_norm_vigour_head", []),
        "grad_norm_value_head": o["history"].get("grad_norm_value_head", []),
        "grad_norm_stim_head": o["history"].get("grad_norm_stim_head", []),
        "policy_loss": o["history"].get("policy_loss", []),
        "value_loss": o["history"].get("value_loss", []),
        "policy_grad_by_stim": o["history"].get("policy_grad_by_stim", []),  # each: [g0,g50,g100]
    }, indent=2) + "\n")

    print(f"[{cfg['model_type']} seed{cfg['seed']}] diverged={int(o['diverged'])}  "
          f"pre={pre_reward:.3f} post={post_reward:.3f} "
          f"recovered={100*post_reward/max(pre_reward,1e-9):.0f}%")


if __name__ == "__main__":
    main()
