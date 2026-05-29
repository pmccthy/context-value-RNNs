#!/usr/bin/env python
"""
Multi-run analysis for the 3-stimulus / 1-context task.

Trains N_RUNS independent models using the best params from the mid-lick
calibration sweep (combo 12: entropy_coef=0.1, lick_cost=0.0) and produces:
  - Per-stimulus lick probability (mean ± SEM across runs)
  - Selective unit proportions by value group (mean ± SEM across runs)

Usage:
    python scripts/13_05_26_multi_run_3s1c.py [--n-runs 10] [--out-dir results]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxval.tasks import StimulusSequence, StateSequence
from cxval.envs import TaskEnv
from cxval.models import RNN, ActorCritic
from cxval.agents import Agent
from cxval.analysis import (
    compute_unit_tuning, selectivity_index, preferred_value_proportions,
)

# =============================================================================
# PARAMS — exact match to mid sweep combo 12
# =============================================================================
ENTROPY_COEF = 0.1
LICK_COST    = 0.0
POLICY_CLIP  = 0.1

HIDDEN_SIZE    = 64
RECURRENT_GAIN = 0.9
N_EPISODES     = 1
BPTT_LEN       = 10
UPDATE_EVERY   = 10
GAMMA          = 0.9
LR             = 9e-4
VALUE_COEF     = 0.5
GRAD_CLIP      = 1.0

TRIALS_PER_PHASE   = 300
PHASES_PER_CONTEXT = 1
CONTEXT_REPS       = 30
STIM_TIMESTEPS     = 5
REWARD_TIMESTEPS   = 3
ITI_TIMESTEPS      = (3, 8)

VALUE_MATRIX = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)
N_STIMULI    = 3
N_CONTEXTS   = 1
STIMULI      = ["s0 (low)", "s1 (mid)", "s2 (high)"]

SI_THRESHOLD    = 0.1
VALUE_THRESHOLD = 0.25


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

def run_one(seed: int, device: torch.device) -> dict | None:
    """Train one model and return lick probs + selectivity proportions.

    Args:
        seed: RNG seed for task generation and model initialisation.
        device: Torch device.

    Returns:
        Dict with lick_sc, props, si, or None if the run diverged.
    """
    # ── task ──────────────────────────────────────────────────────────────
    stim_seq = StimulusSequence(
        value_matrix=VALUE_MATRIX,
        trials_per_phase=TRIALS_PER_PHASE,
        phases_per_context=PHASES_PER_CONTEXT,
        context_order="sequential",
        context_reps=CONTEXT_REPS,
    )
    stim_seq.generate(seed=seed)

    state_seq = StateSequence(
        stimulus_sequence=stim_seq,
        value_matrix=VALUE_MATRIX,
        stim_timesteps=STIM_TIMESTEPS,
        reward_timesteps=REWARD_TIMESTEPS,
        iti_timesteps=ITI_TIMESTEPS,
    )
    states, _, reward_availability = state_seq.generate(seed=seed)
    obs_dim = states.shape[1] + 2

    env = TaskEnv(
        states=states,
        reward_availability=reward_availability,
        reward_lick=1.0,
        reward_no_lick=0.0,
        reward_lick_miss=-1.0,
        lick_cost=LICK_COST,
    )

    # ── model ─────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    backbone     = RNN(input_size=obs_dim, hidden_size=HIDDEN_SIZE, output_size=1,
                       recurrent_gain=RECURRENT_GAIN)
    actor_critic = ActorCritic(backbone=backbone, num_actions=2,
                               policy_clip=POLICY_CLIP).to(device)
    optimizer    = torch.optim.Adam(actor_critic.parameters(), lr=LR)

    def _weights_nan():
        return any(torch.isnan(p).any() for p in actor_critic.parameters())

    # ── training ──────────────────────────────────────────────────────────
    actor_critic.train()
    optimizer.zero_grad()
    obs, _ = env.reset()
    hidden = None
    log_probs_buf, values_buf, rewards_buf, entropies_buf = [], [], [], []
    t_in_window = 0
    done = False

    try:
        while not done:
            obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value, hidden = actor_critic.step(obs_t, hidden)
            dist   = actor_critic.make_dist(logits)
            action = dist.sample()

            log_probs_buf.append(dist.log_prob(action))
            values_buf.append(value)
            entropies_buf.append(dist.entropy())

            obs, reward, done, _, _ = env.step(action.item())
            rewards_buf.append(reward)
            t_in_window += 1

            if t_in_window % UPDATE_EVERY == 0 or done:
                bv = 0.0
                if not done:
                    with torch.no_grad():
                        obs_next = torch.tensor(obs, dtype=torch.float32,
                                                device=device).unsqueeze(0)
                        _, bv_t, _ = actor_critic.step(obs_next, hidden)
                        bv = bv_t.item()

                returns_t    = torch.tensor(
                    compute_returns(rewards_buf, bv, GAMMA),
                    dtype=torch.float32, device=device)
                log_probs_t  = torch.stack(log_probs_buf).squeeze(-1)
                values_t     = torch.stack(values_buf).squeeze(-1)
                entropy_mean = torch.stack(entropies_buf).mean()

                advantages = returns_t - values_t.detach()
                adv_std = advantages.std()
                advantages = ((advantages - advantages.mean()) / (adv_std + 1e-8)
                              if adv_std > 1e-4 else advantages - advantages.mean())

                loss = (-(log_probs_t * advantages).mean()
                        + VALUE_COEF * F.mse_loss(values_t, returns_t)
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
                    if _weights_nan():
                        return None

    except Exception:
        return None

    if _weights_nan():
        return None

    # ── inference ─────────────────────────────────────────────────────────
    infer_stim_seq = StimulusSequence(
        value_matrix=VALUE_MATRIX,
        trials_per_phase=TRIALS_PER_PHASE,
        phases_per_context=1,
        context_order="sequential",
        context_reps=1,
    )
    infer_stim_seq.generate(seed=seed + 1000)

    infer_state_seq = StateSequence(
        stimulus_sequence=infer_stim_seq,
        value_matrix=VALUE_MATRIX,
        stim_timesteps=STIM_TIMESTEPS,
        reward_timesteps=REWARD_TIMESTEPS,
        iti_timesteps=ITI_TIMESTEPS,
    )
    infer_states, _, infer_reward_avail = infer_state_seq.generate(seed=seed + 1000)

    infer_env = TaskEnv(
        states=infer_states,
        reward_availability=infer_reward_avail,
        reward_lick=1.0,
        reward_no_lick=0.0,
        reward_lick_miss=-1.0,
        lick_cost=LICK_COST,
    )

    actor_critic.eval()
    agent = Agent(actor_critic, device=device)
    agent.reset()
    obs, _ = infer_env.reset()
    action_seq, hidden_list = [], []
    done = False

    try:
        while not done:
            action, _, _ = agent.act(obs)
            action_seq.append(action)
            hidden_list.append(agent.hidden.detach().squeeze(0).cpu())
            obs, _, done, _, _ = infer_env.step(action)
    except Exception:
        return None

    hidden_np = np.array(torch.stack(hidden_list).tolist(), dtype=np.float32)  # (T, H)

    # ── lick probabilities per stimulus ───────────────────────────────────
    stim_arr, lick_arr = [], []
    for trial in infer_state_seq.trial_structure:
        rs = trial["reward_window"][0]
        stim_arr.append(trial["stimulus"])
        lick_arr.append(float(action_seq[rs] == TaskEnv.LICK))

    stim_arr = np.array(stim_arr)
    lick_arr = np.array(lick_arr)
    lick_sc  = np.full((N_STIMULI, N_CONTEXTS), np.nan)
    for si in range(N_STIMULI):
        m = stim_arr == si
        if m.sum() > 0:
            lick_sc[si, 0] = lick_arr[m].mean()

    # ── selectivity ───────────────────────────────────────────────────────
    ctx_arr  = np.array([t["context"]  for t in infer_state_seq.trial_structure])
    stim_arr_full = np.array([t["stimulus"] for t in infer_state_seq.trial_structure])
    stim_hidden = np.stack([
        hidden_np[t["stim_window"][0]:t["stim_window"][1]]
        for t in infer_state_seq.trial_structure
    ])

    activations = {
        "stim_hidden":   stim_hidden,
        "reward_hidden": np.stack([
            hidden_np[t["reward_window"][0]:t["reward_window"][1]]
            for t in infer_state_seq.trial_structure
        ]),
        "context":  ctx_arr,
        "stimulus": stim_arr_full,
    }

    tuning           = compute_unit_tuning(activations, period="stim")
    props, si_values = preferred_value_proportions(
        tuning, VALUE_MATRIX,
        si_threshold=SI_THRESHOLD, value_threshold=VALUE_THRESHOLD,
    )

    return dict(lick_sc=lick_sc, props=props, si=si_values)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs",  type=int, default=10)
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    device  = torch.device(args.device)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"multi_run_3s1c_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output → {out_dir}\n")

    lick_sc_runs  = []   # list of (N_STIMULI, N_CONTEXTS)
    frac_low_runs, frac_mid_runs, frac_high_runs = [], [], []
    n_sel_runs = []

    for run_i in range(args.n_runs):
        seed = 42 + run_i
        print(f"[{run_i + 1:>2}/{args.n_runs}]  seed={seed} ...", end="  ", flush=True)
        result = run_one(seed=seed, device=device)
        if result is None:
            print("DIVERGED — skipped")
            continue

        lick_sc_runs.append(result["lick_sc"])
        p = result["props"][0]   # single context
        frac_low_runs.append(p["frac_low"])
        frac_mid_runs.append(p["frac_mid"])
        frac_high_runs.append(p["frac_high"])
        n_sel_runs.append(p["n_selective"])

        lp = result["lick_sc"][:, 0]
        print(f"lick = [{lp[0]:.2f}, {lp[1]:.2f}, {lp[2]:.2f}]  "
              f"mid_err={abs(lp[1] - 0.5):.3f}  "
              f"n_sel={p['n_selective']}")

    n_good = len(lick_sc_runs)
    print(f"\n{n_good}/{args.n_runs} runs completed successfully.")

    if n_good == 0:
        print("No valid runs — exiting.")
        return

    # ── aggregate ─────────────────────────────────────────────────────────
    lick_arr    = np.array(lick_sc_runs)            # (n_good, N_STIMULI, 1)
    lick_mean   = np.nanmean(lick_arr[:, :, 0], 0)  # (N_STIMULI,)
    lick_sem    = np.nanstd(lick_arr[:, :, 0],  0) / np.sqrt(n_good)

    def _agg(vals):
        a = np.array(vals, dtype=float)
        return np.nanmean(a), np.nanstd(a) / np.sqrt(np.sum(~np.isnan(a)))

    grp_stats = {
        "low":  _agg(frac_low_runs),
        "mid":  _agg(frac_mid_runs),
        "high": _agg(frac_high_runs),
    }
    mean_n_sel = np.nanmean(n_sel_runs)

    # ── figure ────────────────────────────────────────────────────────────
    stim_colors = ["#4C72B0", "#8C8C8C", "#DD8452"]
    grp_colors  = {"low": "#4C72B0", "mid": "#8C8C8C", "high": "#DD8452"}
    x_stim      = np.arange(N_STIMULI)
    x_grp       = np.arange(3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── left: per-stimulus lick probability ───────────────────────────────
    ax = axes[0]
    bars = ax.bar(x_stim, lick_mean, color=stim_colors, alpha=0.85, width=0.5, zorder=2)
    ax.errorbar(x_stim, lick_mean, yerr=lick_sem,
                fmt="none", color="black", capsize=6, lw=1.5, zorder=3)
    for si in range(N_STIMULI):
        ax.plot([si - 0.25, si + 0.25], [VALUE_MATRIX[si, 0]] * 2,
                color="black", lw=2.5, zorder=5)
    ax.axhline(0.5, color="gray", lw=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(x_stim)
    ax.set_xticklabels(STIMULI)
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel("Lick probability  (mean ± SEM)")
    ax.set_title(
        f"Per-stimulus lick probability\n"
        f"(black lines = GT reward prob, n={n_good} runs)"
    )

    # ── right: selectivity proportions ────────────────────────────────────
    ax = axes[1]
    grp_labels = list(grp_stats.keys())
    means = [grp_stats[g][0] * 100 for g in grp_labels]
    sems  = [grp_stats[g][1] * 100 for g in grp_labels]
    colors = [grp_colors[g] for g in grp_labels]

    ax.bar(x_grp, means, color=colors, alpha=0.85, width=0.5, zorder=2)
    ax.errorbar(x_grp, means, yerr=sems,
                fmt="none", color="black", capsize=6, lw=1.5, zorder=3)
    ax.set_xticks(x_grp)
    ax.set_xticklabels(["Prefer low\n(v=0)", "Prefer mid\n(v=0.5)", "Prefer high\n(v=1)"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("% selective units  (mean ± SEM)")
    ax.set_title(
        f"Value preference of selective units\n"
        f"(SI ≥ {SI_THRESHOLD},  mean n_selective = {mean_n_sel:.0f},  n={n_good} runs)"
    )

    fig.suptitle(
        f"3s1c task  ·  entropy_coef={ENTROPY_COEF}  lick_cost={LICK_COST}  "
        f"policy_clip={POLICY_CLIP}  ·  {n_good} runs",
        y=1.02,
    )
    plt.tight_layout()
    fig_path = out_dir / f"multi_run_summary_{ts}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {fig_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
