#!/usr/bin/env python
"""
Hidden-size sweep for the 3-stimulus / 1-context task.

Trains RNN actor-critic models of varying sizes on a 3-stimulus task
(s0=0%, s1=50%, s2=100% reward probability, 1 context).  For each
(hidden_size, seed) combination, saves a vis_data.pkl and model.pt.
An aggregated CSV captures performance and PSA metrics for all runs.

Usage
-----
    python scripts/20_05_26_sweep_3s1c.py
    python scripts/20_05_26_sweep_3s1c.py --seeds 5 --device mps
    python scripts/20_05_26_sweep_3s1c.py --hidden-sizes 2 4 8 16 32 64 128
"""
from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxval.tasks import StimulusSequence, StateSequence
from cxval.envs import TaskEnv
from cxval.models import RNN, ActorCritic
from cxval.agents import Agent
from cxval.analysis import policy_similarity_analysis

# =============================================================================
# TASK
# =============================================================================

VALUE_MATRIX = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)
STIMULI      = ["s0 (0%)", "s1 (50%)", "s2 (100%)"]
CONTEXTS     = ["c0"]
N_STIMULI    = 3
N_CONTEXTS   = 1

STIM_GROUP_INFO = [
    {"name": "low",  "short": "low",  "indices": [0]},
    {"name": "mid",  "short": "mid",  "indices": [1]},
    {"name": "high", "short": "high", "indices": [2]},
]

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

RECURRENT_GAIN   = 0.9
POLICY_CLIP      = 0.1
LICK_COST        = 0.0
BPTT_LEN         = 10
UPDATE_EVERY     = 10
GAMMA            = 0.9
LR               = 9e-4
VALUE_COEF       = 0.5
ENTROPY_COEF     = 0.1
GRAD_CLIP        = 1.0
TRIALS_PER_PHASE = 300
PHASES_PER_CTX   = 1
CONTEXT_REPS     = 30
STIM_TIMESTEPS   = 5
REWARD_TIMESTEPS = 3
ITI_TIMESTEPS    = (3, 8)


# =============================================================================
# HELPERS
# =============================================================================

def compute_returns(rewards, bootstrap_value, gamma):
    returns, R = [], float(bootstrap_value)
    for r in reversed(rewards):
        R = r + gamma * R
        returns.append(R)
    returns.reverse()
    return returns


# =============================================================================
# SINGLE RUN
# =============================================================================

def run_one(hidden_size: int, seed: int, device: torch.device, out_dir: Path) -> dict:
    tag = f"h={hidden_size}  seed={seed}"
    nan_row = dict(
        hidden_size=hidden_size, seed=seed, run_id="",
        spearman_r=np.nan, psa_score=np.nan, psa_delta=np.nan,
        lick_high=np.nan, lick_mid=np.nan, lick_low=np.nan,
        diverged=True,
    )

    # ── task ──────────────────────────────────────────────────────────────
    stim_seq = StimulusSequence(
        value_matrix=VALUE_MATRIX, trials_per_phase=TRIALS_PER_PHASE,
        phases_per_context=PHASES_PER_CTX, context_order="sequential",
        context_reps=CONTEXT_REPS,
    )
    stim_seq.generate(seed=seed)

    state_seq = StateSequence(
        stimulus_sequence=stim_seq, value_matrix=VALUE_MATRIX,
        stim_timesteps=STIM_TIMESTEPS, reward_timesteps=REWARD_TIMESTEPS,
        iti_timesteps=ITI_TIMESTEPS,
    )
    states, _, reward_availability = state_seq.generate(seed=seed)
    obs_dim = states.shape[1] + 2
    n_trials = len(stim_seq.trial_contexts)

    env = TaskEnv(
        states=states, reward_availability=reward_availability,
        reward_lick=1.0, reward_no_lick=0.0, reward_lick_miss=-1.0,
        lick_cost=LICK_COST,
    )

    # ── model ─────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    backbone     = RNN(input_size=obs_dim, hidden_size=hidden_size, output_size=1,
                       recurrent_gain=RECURRENT_GAIN)
    actor_critic = ActorCritic(backbone=backbone, num_actions=2,
                                policy_clip=POLICY_CLIP).to(device)
    optimizer    = torch.optim.Adam(actor_critic.parameters(), lr=LR)

    # ── training ──────────────────────────────────────────────────────────
    actor_critic.train()
    optimizer.zero_grad()
    obs, _  = env.reset()
    hidden  = None
    all_trial_data = []
    log_probs_buf, values_buf, rewards_buf, entropies_buf = [], [], [], []
    action_list, value_ts = [], []
    t_in_window = 0

    try:
        done = False
        while not done:
            obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value, hidden = actor_critic.step(obs_t, hidden)
            dist   = actor_critic.make_dist(logits)
            action = dist.sample()

            log_probs_buf.append(dist.log_prob(action))
            values_buf.append(value)
            entropies_buf.append(dist.entropy())
            value_ts.append(value.detach().item())
            action_list.append(action.item())

            obs, reward, done, _, _ = env.step(action.item())
            rewards_buf.append(reward)
            t_in_window += 1

            if t_in_window % UPDATE_EVERY == 0 or done:
                bootstrap_v = 0.0
                if not done:
                    with torch.no_grad():
                        obs_n = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        _, bv, _ = actor_critic.step(obs_n, hidden)
                        bootstrap_v = bv.item()

                returns_t    = torch.tensor(
                    compute_returns(rewards_buf, bootstrap_v, GAMMA),
                    dtype=torch.float32, device=device,
                )
                log_probs_t  = torch.stack(log_probs_buf).squeeze(-1)
                values_t     = torch.stack(values_buf).squeeze(-1)
                entropy_mean = torch.stack(entropies_buf).mean()

                advantages = returns_t - values_t.detach()
                adv_std    = advantages.std()
                advantages = ((advantages - advantages.mean()) / (adv_std + 1e-8)
                              if adv_std > 1e-4 else advantages - advantages.mean())

                loss = (-(log_probs_t * advantages).mean()
                        + VALUE_COEF   * F.mse_loss(values_t, returns_t)
                        - ENTROPY_COEF * entropy_mean)

                is_last = (t_in_window >= BPTT_LEN) or done
                loss.backward(retain_graph=not is_last)
                log_probs_buf, values_buf, rewards_buf, entropies_buf = [], [], [], []

                if is_last:
                    nn.utils.clip_grad_norm_(actor_critic.parameters(), GRAD_CLIP)
                    optimizer.step()
                    optimizer.zero_grad()
                    hidden = hidden.detach()
                    t_in_window = 0
                    if any(torch.isnan(p).any() for p in actor_critic.parameters()):
                        warnings.warn(f"NaN divergence ({tag})")
                        return nan_row

    except Exception as exc:
        warnings.warn(f"Training crashed ({tag}): {exc}")
        return nan_row

    for ti, trial in enumerate(state_seq.trial_structure):
        rs, re = trial["reward_window"]
        all_trial_data.append({
            "global_trial":     ti,
            "context":          trial["context"],
            "stimulus":         trial["stimulus"],
            "reward_available": trial["reward_available"],
            "licked":           int(action_list[rs] == TaskEnv.LICK),
            "value_estimate":   float(np.mean(value_ts[rs:re])),
            "lick_count":       sum(1 for a in action_list[rs:re] if a == TaskEnv.LICK),
        })

    # ── inference ─────────────────────────────────────────────────────────
    infer_stim_seq = StimulusSequence(
        value_matrix=VALUE_MATRIX, trials_per_phase=TRIALS_PER_PHASE,
        phases_per_context=1, context_order="sequential", context_reps=1,
    )
    infer_stim_seq.generate(seed=seed + 1000)

    infer_state_seq = StateSequence(
        stimulus_sequence=infer_stim_seq, value_matrix=VALUE_MATRIX,
        stim_timesteps=STIM_TIMESTEPS, reward_timesteps=REWARD_TIMESTEPS,
        iti_timesteps=ITI_TIMESTEPS,
    )
    infer_states, _, infer_ravail = infer_state_seq.generate(seed=seed + 1000)

    infer_env = TaskEnv(
        states=infer_states, reward_availability=infer_ravail,
        reward_lick=1.0, reward_no_lick=0.0, reward_lick_miss=-1.0,
        lick_cost=LICK_COST,
    )

    actor_critic.eval()
    agent = Agent(actor_critic, device=device)
    agent.reset()
    obs, _ = infer_env.reset()
    infer_action_seq, infer_value_ts, infer_hidden_list = [], [], []

    done = False
    while not done:
        action, _, value = agent.act(obs)
        infer_action_seq.append(action)
        infer_value_ts.append(value.item())
        infer_hidden_list.append(agent.hidden.detach().squeeze(0))
        obs, _, done, _, _ = infer_env.step(action)

    infer_hidden_np = np.array(torch.stack(infer_hidden_list).tolist(), dtype=np.float32)
    infer_struct    = infer_state_seq.trial_structure

    infer_trial_data = []
    for ti, trial in enumerate(infer_struct):
        rs, re = trial["reward_window"]
        infer_trial_data.append({
            "global_trial":     n_trials + ti,
            "context":          trial["context"],
            "stimulus":         trial["stimulus"],
            "reward_available": trial["reward_available"],
            "licked":           int(infer_action_seq[rs] == TaskEnv.LICK),
            "value_estimate":   float(np.mean(infer_value_ts[rs:re])),
            "lick_count":       sum(1 for a in infer_action_seq[rs:re] if a == TaskEnv.LICK),
        })

    ctx_arr  = np.array([t["context"]          for t in infer_struct])
    stim_arr = np.array([t["stimulus"]         for t in infer_struct])
    ravail   = np.array([t["reward_available"] for t in infer_struct], dtype=bool)

    infer_activations = {
        "hidden_states":    infer_hidden_np,
        "stim_hidden":      np.stack([infer_hidden_np[t["stim_window"][0]:t["stim_window"][1]]
                                       for t in infer_struct]),
        "reward_hidden":    np.stack([infer_hidden_np[t["reward_window"][0]:t["reward_window"][1]]
                                       for t in infer_struct]),
        "context":          ctx_arr,
        "stimulus":         stim_arr,
        "reward_available": ravail,
        "trial_structure":  infer_struct,
    }

    # ── metrics ───────────────────────────────────────────────────────────
    inf_lick = np.array([d["licked"] for d in infer_trial_data])
    lick_sc  = np.full((N_STIMULI, N_CONTEXTS), np.nan)
    for si in range(N_STIMULI):
        for ci in range(N_CONTEXTS):
            m = (stim_arr == si) & (ctx_arr == ci)
            if m.sum():
                lick_sc[si, ci] = inf_lick[m].mean()

    psa     = policy_similarity_analysis(lick_sc, VALUE_MATRIX)
    lp_flat = [lick_sc[si, 0] for si in range(N_STIMULI) if not np.isnan(lick_sc[si, 0])]
    rp_flat = [float(VALUE_MATRIX[si, 0]) for si in range(N_STIMULI) if not np.isnan(lick_sc[si, 0])]
    spear_r = float(spearmanr(rp_flat, lp_flat)[0]) if len(lp_flat) > 2 else np.nan

    # ── save vis_data.pkl ─────────────────────────────────────────────────
    run_id  = f"RNN_3s1c_h{hidden_size}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    vis_data = {
        # task
        "value_matrix":               VALUE_MATRIX,
        "n_stimuli":                  N_STIMULI,
        "n_contexts":                 N_CONTEXTS,
        "stimuli":                    STIMULI,
        "contexts":                   CONTEXTS,
        "stim_group_info":            STIM_GROUP_INFO,
        "stim_timesteps":             STIM_TIMESTEPS,
        "reward_timesteps":           REWARD_TIMESTEPS,
        "reward_lick":                1.0,
        "reward_lick_miss":           -1.0,
        "reward_no_lick":             0.0,
        "lick_cost":                  LICK_COST,
        # training
        "n_trials":                   n_trials,
        "trials_per_phase":           TRIALS_PER_PHASE,
        "context_reps":               CONTEXT_REPS,
        "all_trial_data":             all_trial_data,
        # inference
        "infer_trial_data":           infer_trial_data,
        "infer_action_seq":           infer_action_seq,
        "infer_trial_structure":      infer_struct,
        "infer_activations":          infer_activations,
        # saved for injection experiments
        "infer_states":               infer_states,
        "infer_reward_availability":  infer_ravail,
        # decoding helpers
        "stim_groups":                {"all": None},
        "group_colors":               {"all": "black"},
        "pooling":                    "average",
        "n_folds":                    5,
        # per-run identity
        "hidden_size":                hidden_size,
        "seed":                       seed,
        "run_id":                     run_id,
        "lick_sc":                    lick_sc,
        "psa_results":                psa,
        "spearman_r":                 spear_r,
        "env_kwargs": {
            "reward_lick": 1.0, "reward_no_lick": 0.0,
            "reward_lick_miss": -1.0, "lick_cost": LICK_COST,
        },
    }

    with open(run_dir / "vis_data.pkl", "wb") as f:
        pickle.dump(vis_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(actor_critic.state_dict(), run_dir / "model.pt")

    return dict(
        hidden_size=hidden_size, seed=seed, run_id=run_id,
        spearman_r=spear_r,
        psa_score=float(psa[0]["psa_score"]),
        psa_delta=float(psa[0]["psa_delta"]),
        lick_high=float(psa[0]["high_lick"]),
        lick_mid=float(psa[0]["mid_lick"]),
        lick_low=float(psa[0]["low_lick"]),
        diverged=False,
    )


# =============================================================================
# DIAGNOSTIC PLOTS
# =============================================================================

def plot_sweep_results(df: pd.DataFrame, out_dir: Path, ts: str) -> None:
    hidden_sizes = sorted(df["hidden_size"].unique())
    palette      = plt.cm.plasma(np.linspace(0.15, 0.85, len(hidden_sizes)))
    h_color      = {h: c for h, c in zip(hidden_sizes, palette)}

    metrics = [
        ("spearman_r",  "Lick–value calibration (Spearman r)", -1,   1,    None),
        ("lick_high",   "Lick rate — high stim (→ 1)",          0,   1,    1.0),
        ("lick_mid",    "Lick rate — mid stim  (→ 0.5)",         0,   1,    0.5),
        ("lick_low",    "Lick rate — low stim  (→ 0)",           0,   1,    0.0),
        ("psa_score",   "PSA score  (1 − MAE, → 1)",             0,   1,    1.0),
        ("psa_delta",   "PSA delta  (high − low, → 1)",          -1,  1,    1.0),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    x_pos = np.arange(len(hidden_sizes))

    for ax, (col, title, ymin, ymax, ref) in zip(axes.flat, metrics):
        means = [df[df["hidden_size"] == h][col].mean() for h in hidden_sizes]
        stds  = [df[df["hidden_size"] == h][col].std()  for h in hidden_sizes]
        colors = [h_color[h] for h in hidden_sizes]

        bars = ax.bar(x_pos, means, color=colors, alpha=0.85, zorder=2)
        ax.errorbar(x_pos, means, yerr=stds, fmt="none", color="black",
                    capsize=4, lw=1.5, zorder=3)

        # Individual seed points
        for xi, h in enumerate(hidden_sizes):
            vals = df[df["hidden_size"] == h][col].dropna()
            ax.scatter(np.full(len(vals), xi), vals, color="black",
                       s=18, alpha=0.6, zorder=4)

        if ref is not None:
            ax.axhline(ref, color="gray", lw=0.8, linestyle="--", alpha=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(h) for h in hidden_sizes])
        ax.set_xlabel("Hidden size")
        ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin))
        ax.set_title(title, fontsize=9)

    n_seeds = len(df["seed"].unique())
    fig.suptitle(
        f"3s1c hidden-size sweep  (n_seeds={n_seeds})  —  inference metrics  [plasticity off]",
        y=1.01, fontsize=10,
    )
    plt.tight_layout()
    p = out_dir / f"sweep_3s1c_metrics_{ts}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Metrics figure → {p}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--hidden-sizes", type=int, nargs="+",
                        default=[2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--seeds",     type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--device",    default="cpu")
    parser.add_argument("--out-dir",   default="results/20_05_26_sweep_3s1c")
    args = parser.parse_args()

    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [(h, args.base_seed + s)
               for h in args.hidden_sizes
               for s in range(args.seeds)]
    n_runs  = len(configs)

    print(f"3s1c hidden-size sweep: {len(args.hidden_sizes)} sizes × {args.seeds} seeds = {n_runs} runs")
    print(f"Hidden sizes: {args.hidden_sizes}  |  Device: {device}\n")

    rows   = []
    for n_done, (h, seed) in enumerate(configs, 1):
        print(f"[{n_done:>3}/{n_runs}]  h={h:<4}  seed={seed}", end="  ", flush=True)
        row = run_one(h, seed, device, out_dir)
        rows.append(row)
        if row["diverged"]:
            print("DIVERGED")
        else:
            print(f"spearman={row['spearman_r']:.3f}  "
                  f"psa={row['psa_score']:.3f}  "
                  f"delta={row['psa_delta']:.3f}  "
                  f"mid={row['lick_mid']:.3f}")

    df  = pd.DataFrame(rows)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = out_dir / f"sweep_3s1c_{ts}.csv"
    df.to_csv(csv, index=False)
    print(f"\nResults → {csv}")

    # Summary table
    print("\n" + "=" * 75)
    print(f"{'h':>5}  {'spearman':>10}  {'psa_score':>10}  {'lick_high':>10}  "
          f"{'lick_mid':>10}  {'lick_low':>10}")
    print("=" * 75)
    for h in sorted(df["hidden_size"].unique()):
        sub = df[df["hidden_size"] == h]
        print(f"  h={h:<4}  "
              f"{sub.spearman_r.mean():+.3f}±{sub.spearman_r.std():.3f}  "
              f"{sub.psa_score.mean():.3f}±{sub.psa_score.std():.3f}  "
              f"{sub.lick_high.mean():.3f}±{sub.lick_high.std():.3f}  "
              f"{sub.lick_mid.mean():.3f}±{sub.lick_mid.std():.3f}  "
              f"{sub.lick_low.mean():.3f}±{sub.lick_low.std():.3f}")

    plot_sweep_results(df, out_dir, ts)


if __name__ == "__main__":
    main()
