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
from cxval.vigour import VigourActorCritic, BatchedVigourEnv, train_vigour, infer_vigour, infer_value
from cxval.analysis import responsive_proportions_ttest


def build_model(sd):
    """Reconstruct a VigourActorCritic from a state_dict (architecture from shapes)."""
    H = sd["backbone.h2h.weight"].shape[0]
    obs = sd["backbone.input2h.weight"].shape[1]
    rf = sd["vigour_head.weight"].shape[1] / H
    aux = sd["stim_head.weight"].shape[0] if "stim_head.weight" in sd else 0
    ac = VigourActorCritic(RNN(input_size=obs, hidden_size=H, output_size=1,
                               recurrent_gain=0.9), action_std=0.05,
                           readout_fraction=rf, aux_n_stim=aux)
    ac.load_state_dict(sd); ac.eval()
    return ac



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

def _quick_crosscontext_decode(R, stim, centroids_ref):
    """Fast, dependency-free (numpy only) 3-way nearest-centroid decode of
    the CURRENT rollout's stim_hidden R against a FIXED reference centroid
    set (centroids_ref, built ONCE from a frozen pre-reversal rollout before
    any reversal training begins -- see main()). No train/test split is
    needed here (unlike _quick_stim_decode's within-rollout split): the
    reference centroids come from a completely separate rollout, so there's
    no leakage. This is "cross-context stimulus decoding vs trials": can a
    classifier trained on the ORIGINAL (pre-reversal) representation still
    recognise each physical stimulus in the CURRENT (mid-reversal-training)
    representation, tracked at every probe point through the reversal --
    directly answering whether the stimulus code drifts away from its
    original geometry as the contingency reversal is learned, or stays
    anchored to it. Chance = 1/3, same as _quick_stim_decode."""
    d = np.linalg.norm(R[:, None, :] - centroids_ref[None, :, :], axis=2)
    pred = d.argmin(axis=1)
    return float((pred == stim).mean())


def make_probe(value_matrix, cost, centroids_pre=None):
    """A cheap deterministic probe: per-stimulus vigour, population activity and responder
    fraction, logged periodically during training to trace the reversal time course.

    centroids_pre: optional (3, H) array of FROZEN pre-reversal stim_hidden
    centroids (see main()) -- when given, also logs probe_crosscontext_decode
    (see _quick_crosscontext_decode)."""
    def probe(model):
        acts, vmean = infer_vigour(model, value_matrix, n_eval_episodes=4,
                                   n_trials_per_episode=150, vigour_cost=cost)
        R = np.asarray(acts["stim_hidden"]).mean(1); stim = np.asarray(acts["stimulus"])
        tuning = np.stack([R[stim == s].mean(0) for s in range(3)])
        rp = responsive_proportions_ttest(acts, period="stim")
        # genuine trial-resolved critic value estimate V(s) on the REVERSED contingency
        # (value_matrix here is VM_rev, passed in from main()) -- same rollout cost class
        # as infer_vigour, real training-time curve, not terminal_rpe.py's reconstruction.
        vval = infer_value(model, value_matrix, n_eval_episodes=4,
                           n_trials_per_episode=150, vigour_cost=cost)
        out = {"vigour": [float(vmean[s]) for s in range(3)],
               "pop_activity": tuning.mean(1).astype(float).tolist(),
               "frac_responsive": np.asarray(rp["frac_per_stim"], float).tolist(),
               "value": [float(vval[s]) for s in range(3)],
               "stim_decode": _quick_stim_decode(R, stim)}
        if centroids_pre is not None:
            out["crosscontext_decode"] = _quick_crosscontext_decode(R, stim, centroids_pre)
        return out
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
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="save a full model checkpoint every N updates under "
                         "<out>/checkpoints/ (0 to disable, the default -- "
                         "these are cheap, ~72KB each, but off by default so "
                         "existing invocations are unaffected). The very "
                         "first update is always checkpointed in addition to "
                         "the regular interval whenever this is > 0.")
    ap.add_argument("--checkpoint-fine-until", type=int, default=None,
                    help="checkpoint at --checkpoint-fine-every resolution for "
                         "updates 1..N (covers the window right after the "
                         "reversal begins, where the interesting change "
                         "happens), then fall back to --checkpoint-every. "
                         "Defaults to --checkpoint-every itself (i.e. fills in "
                         "the whole first coarse interval at fine "
                         "resolution); pass 0 to disable fine sampling.")
    ap.add_argument("--checkpoint-fine-every", type=int, default=1,
                    help="fine-window checkpoint interval (default: every "
                         "update).")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run = Path(args.run); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((run / "config.json").read_text())
    tr = dict(cfg["train"])                                  # original training kwargs
    tr["iti_timesteps"] = tuple(tr["iti_timesteps"])
    sd = torch.load(run / "model.pt", map_location="cpu")

    VM = np.asarray(cfg["value_matrix"], np.float32)         # [[0],[.5],[1]]
    VM_rev = VM[::-1].copy()                                 # [[1],[.5],[0]]  (swap 0% <-> 100%)
    cost = tr["vigour_cost"]

    pre_reward = eval_reward(build_model(sd), VM, cost)      # trained model on ORIGINAL task

    # Frozen pre-reversal reference centroids for probe_crosscontext_decode:
    # one rollout of the trained (pre-reversal) model on the ORIGINAL task,
    # computed ONCE here before any reversal training happens -- every probe
    # point during reversal training then tests against this SAME fixed
    # reference, so the resulting curve is genuinely "decode accuracy vs.
    # trials since the reversal started" rather than a moving target.
    _acts_pre, _ = infer_vigour(build_model(sd), VM, n_eval_episodes=4,
                                n_trials_per_episode=150, vigour_cost=cost)
    _R_pre = np.asarray(_acts_pre["stim_hidden"]).mean(1)
    _stim_pre = np.asarray(_acts_pre["stimulus"])
    centroids_pre = np.stack([_R_pre[_stim_pre == s].mean(0) for s in range(3)])

    # continue-train on the reversed task, warm-started from the trained weights
    tr["n_trials_per_episode"] = args.n_trials
    checkpoint_dir = (out / "checkpoints") if args.checkpoint_every > 0 else None
    fine_until = args.checkpoint_fine_until
    if fine_until is None:
        fine_until = args.checkpoint_every   # default: fill the first coarse interval
    o = train_vigour(VM_rev, device=args.device, init_model=sd,
                     probe_every=args.probe_every,
                     probe_fn=make_probe(VM_rev, cost, centroids_pre=centroids_pre),
                     checkpoint_dir=checkpoint_dir, checkpoint_every=args.checkpoint_every,
                     checkpoint_fine_until=fine_until,
                     checkpoint_fine_every=args.checkpoint_fine_every,
                     **tr)

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
        "probe_value": o["history"].get("probe_value", []),
        "probe_stim_decode": o["history"].get("probe_stim_decode", []),
        "probe_crosscontext_decode": o["history"].get("probe_crosscontext_decode", []),
    }, indent=2) + "\n")

    print(f"[{cfg['model_type']} seed{cfg['seed']}] diverged={int(o['diverged'])}  "
          f"pre={pre_reward:.3f} post={post_reward:.3f} "
          f"recovered={100*post_reward/max(pre_reward,1e-9):.0f}%")


if __name__ == "__main__":
    main()
