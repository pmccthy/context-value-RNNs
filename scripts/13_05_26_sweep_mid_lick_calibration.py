#!/usr/bin/env python
"""
Sweep to find params that produce lick probability ≈ 0.5 for the mid-value
stimulus in a 3-stimulus / 1-context task (v = 0.0, 0.5, 1.0), AND
monotonic selectivity (frac_low < frac_mid < frac_high).

Primary metrics:
  mid_lick_error     = |lick_prob(s_mid) − 0.5|  (lower = better)
  sel_monotonicity   = frac_low < frac_mid < frac_high  (higher = better)
  combined_score     = −mid_lick_error + sel_monotonicity  (higher = better)

Edit SWEEP and N_SEEDS at the top, then run:
    python scripts/13_05_26_sweep_mid_lick_calibration.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from collections import defaultdict
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
from cxval.analysis import compute_unit_tuning, preferred_value_proportions

SI_THRESHOLD    = 0.1
VALUE_THRESHOLD = 0.25

# =============================================================================
# SWEEP CONFIGURATION  ← edit here
# =============================================================================
SWEEP: dict[str, list] = {
    "entropy_coef": [0.001, 0.01, 0.05, 0.1, 0.2],
    "lick_cost":    [0.0, -0.1, -0.2, -0.3],
}
N_SEEDS  = 3
BASE_SEED = 42

# =============================================================================
# FIXED HYPERPARAMETERS  (match 13_05_26_sweep_training_hyperparams.py)
# =============================================================================
HIDDEN_SIZE    = 64
RECURRENT_GAIN = 0.9
N_EPISODES     = 1
BPTT_LEN       = 10
UPDATE_EVERY   = 10
GAMMA          = 0.9
LR             = 9e-4
VALUE_COEF     = 0.5
GRAD_CLIP      = 1.0
POLICY_CLIP    = 0.1   # best value from training hyperparams sweep

TRIALS_PER_PHASE   = 300
PHASES_PER_CONTEXT = 1
CONTEXT_REPS       = 30
STIM_TIMESTEPS     = 5
REWARD_TIMESTEPS   = 3
ITI_TIMESTEPS      = (3, 8)

# 3-stimulus / 1-context task
VALUE_MATRIX = np.array([
    [0.0],   # s0: low
    [0.5],   # s1: mid  ← want lick_prob ≈ 0.5
    [1.0],   # s2: high
], dtype=np.float32)

N_STIMULI  = VALUE_MATRIX.shape[0]
N_CONTEXTS = VALUE_MATRIX.shape[1]
MID_STIM   = 1   # index of mid-value stimulus


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

def run_one(entropy_coef: float, lick_cost: float, seed: int,
            device: torch.device) -> dict:
    """Train one model and return per-stimulus inference metrics.

    Args:
        entropy_coef: Entropy regularisation coefficient.
        lick_cost: Penalty applied when licking on an unrewarded trial.
        seed: RNG seed for task generation and model init.
        device: Torch device.

    Returns:
        Dict of scalar metrics plus lick_sc array.
    """
    _nan_row = dict(
        spearman_r=np.nan, mid_lick_prob=np.nan, mid_lick_error=np.nan,
        low_lick_prob=np.nan, high_lick_prob=np.nan,
        pct_reward_consumed=np.nan, false_alarm_rate=np.nan,
        monotonicity=np.nan, diverged=True,
        frac_low=np.nan, frac_mid=np.nan, frac_high=np.nan,
        n_selective=np.nan, sel_monotonicity=np.nan, combined_score=np.nan,
        lick_sc=np.full((N_STIMULI, N_CONTEXTS), np.nan),
    )

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
        lick_cost=lick_cost,
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
    diverged = False
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
                bootstrap_v = 0.0
                if not done:
                    with torch.no_grad():
                        obs_next = torch.tensor(obs, dtype=torch.float32,
                                                device=device).unsqueeze(0)
                        _, bv, _ = actor_critic.step(obs_next, hidden)
                        bootstrap_v = bv.item()

                returns_t    = torch.tensor(
                    compute_returns(rewards_buf, bootstrap_v, GAMMA),
                    dtype=torch.float32, device=device)
                log_probs_t  = torch.stack(log_probs_buf).squeeze(-1)
                values_t     = torch.stack(values_buf).squeeze(-1)
                entropy_mean = torch.stack(entropies_buf).mean()

                advantages = returns_t - values_t.detach()
                adv_std = advantages.std()
                if adv_std > 1e-4:
                    advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
                else:
                    advantages = advantages - advantages.mean()

                actor_loss  = -(log_probs_t * advantages).mean()
                critic_loss = F.mse_loss(values_t, returns_t)
                loss        = (actor_loss + VALUE_COEF * critic_loss
                               - entropy_coef * entropy_mean)

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
                        diverged = True
                        break

    except Exception as exc:
        warnings.warn(f"Training crashed (seed={seed}): {exc}")
        return _nan_row

    if diverged or _weights_nan():
        warnings.warn(f"NaN weights (seed={seed})")
        return _nan_row

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
        lick_cost=lick_cost,
    )

    actor_critic.eval()
    agent = Agent(actor_critic, device=device)
    agent.reset()
    obs, _ = infer_env.reset()
    action_seq, hidden_list, done = [], [], False

    try:
        while not done:
            action, _, _ = agent.act(obs)
            action_seq.append(action)
            hidden_list.append(agent.hidden.detach().squeeze(0).cpu())
            obs, _, done, _, _ = infer_env.step(action)
    except Exception as exc:
        warnings.warn(f"Inference crashed (seed={seed}): {exc}")
        return _nan_row

    hidden_np = np.array(torch.stack(hidden_list).tolist(), dtype=np.float32)

    # ── metrics ───────────────────────────────────────────────────────────
    lick_sc    = np.full((N_STIMULI, N_CONTEXTS), np.nan)
    ravail_arr = []
    stim_arr   = []
    licked_arr = []

    for trial in infer_state_seq.trial_structure:
        rs = trial["reward_window"][0]
        si = trial["stimulus"]
        ci = trial["context"]
        ra = trial["reward_available"]
        lp = float(action_seq[rs] == TaskEnv.LICK)

        stim_arr.append(si)
        ravail_arr.append(ra)
        licked_arr.append(lp)

        m = np.array(stim_arr) == si
        # accumulate incrementally — recompute below after loop

    stim_arr   = np.array(stim_arr)
    ravail_arr = np.array(ravail_arr, dtype=bool)
    licked_arr = np.array(licked_arr)

    lp_flat, rp_flat = [], []
    for si in range(N_STIMULI):
        for ci in range(N_CONTEXTS):
            m = stim_arr == si
            if m.sum() == 0:
                continue
            lp = float(licked_arr[m].mean())
            lick_sc[si, ci] = lp
            lp_flat.append(lp)
            rp_flat.append(float(VALUE_MATRIX[si, ci]))

    spear_r    = spearmanr(rp_flat, lp_flat)[0] if len(lp_flat) > 2 else np.nan
    mid_lp     = float(lick_sc[MID_STIM, 0]) if not np.isnan(lick_sc[MID_STIM, 0]) else np.nan
    low_lp     = float(lick_sc[0, 0]) if not np.isnan(lick_sc[0, 0]) else np.nan
    high_lp    = float(lick_sc[2, 0]) if not np.isnan(lick_sc[2, 0]) else np.nan
    mid_err    = float(abs(mid_lp - 0.5)) if not np.isnan(mid_lp) else np.nan
    pct_reward = float(licked_arr[ravail_arr].mean() * 100) if ravail_arr.any() else np.nan
    false_alarm = float(licked_arr[~ravail_arr].mean() * 100) if (~ravail_arr).any() else np.nan

    # monotonicity: low < mid < high
    mono = float(low_lp < mid_lp < high_lp) if not any(
        np.isnan(x) for x in [low_lp, mid_lp, high_lp]
    ) else np.nan

    # ── selectivity ───────────────────────────────────────────────────────
    ctx_arr       = np.array([t["context"]  for t in infer_state_seq.trial_structure])
    stim_arr_full = np.array([t["stimulus"] for t in infer_state_seq.trial_structure])
    stim_hidden   = np.array([
        hidden_np[t["stim_window"][0]:t["stim_window"][1]].tolist()
        for t in infer_state_seq.trial_structure
    ], dtype=np.float32)
    rew_hidden = np.array([
        hidden_np[t["reward_window"][0]:t["reward_window"][1]].tolist()
        for t in infer_state_seq.trial_structure
    ], dtype=np.float32)

    activations = {
        "stim_hidden":   stim_hidden,
        "reward_hidden": rew_hidden,
        "context":       ctx_arr,
        "stimulus":      stim_arr_full,
    }
    tuning           = compute_unit_tuning(activations, period="stim")
    props, _si_vals  = preferred_value_proportions(
        tuning, VALUE_MATRIX,
        si_threshold=SI_THRESHOLD, value_threshold=VALUE_THRESHOLD,
    )
    p           = props[0]   # single context
    frac_low    = float(p["frac_low"])
    frac_mid    = float(p["frac_mid"])
    frac_high   = float(p["frac_high"])
    n_selective = int(p["n_selective"])
    sel_mono    = float(frac_low < frac_mid < frac_high)
    combined    = float(sel_mono - mid_err)   # higher is better

    return dict(
        spearman_r=spear_r,
        mid_lick_prob=mid_lp,
        mid_lick_error=mid_err,
        low_lick_prob=low_lp,
        high_lick_prob=high_lp,
        pct_reward_consumed=pct_reward,
        false_alarm_rate=false_alarm,
        monotonicity=mono,
        diverged=False,
        frac_low=frac_low, frac_mid=frac_mid, frac_high=frac_high,
        n_selective=n_selective, sel_monotonicity=sel_mono,
        combined_score=combined,
        lick_sc=lick_sc,
    )


# =============================================================================
# PER-COMBO FIGURE
# =============================================================================

def _plot_per_combo(combo_params, lick_sc_runs, sel_runs, out_dir, ts, combo_idx):
    """Lick probability and selectivity bar charts for one combo averaged across seeds.

    Args:
        combo_params: dict of param name → value.
        lick_sc_runs: list of (N_STIMULI, N_CONTEXTS) arrays, one per seed.
        sel_runs: list of dicts with frac_low/mid/high and n_selective, one per seed.
        out_dir: save directory.
        ts: timestamp string.
        combo_idx: integer index for filename ordering.
    """
    lick_sc_arr  = np.array(lick_sc_runs, dtype=float)
    lick_sc_mean = np.nanmean(lick_sc_arr, axis=0)
    lick_sc_std  = np.nanstd(lick_sc_arr,  axis=0)

    stimuli    = [f"s{i}" for i in range(N_STIMULI)]
    x          = np.arange(N_STIMULI)
    bar_colors = ["#4C72B0", "#8C8C8C", "#DD8452"]

    lp_flat = [lick_sc_mean[si, 0] for si in range(N_STIMULI)
               if not np.isnan(lick_sc_mean[si, 0])]
    rp_flat = [float(VALUE_MATRIX[si, 0]) for si in range(N_STIMULI)
               if not np.isnan(lick_sc_mean[si, 0])]
    r_sp   = spearmanr(rp_flat, lp_flat)[0] if len(lp_flat) > 2 else np.nan
    mid_lp = lick_sc_mean[MID_STIM, 0]

    # selectivity aggregation
    sel_keys  = ["frac_low", "frac_mid", "frac_high"]
    sel_mean  = {k: np.nanmean([s[k] for s in sel_runs]) for k in sel_keys}
    sel_std   = {k: np.nanstd( [s[k] for s in sel_runs]) for k in sel_keys}
    mean_nsel = np.nanmean([s["n_selective"] for s in sel_runs])
    sel_mono  = bool(sel_mean["frac_low"] < sel_mean["frac_mid"] < sel_mean["frac_high"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # ── calibration scatter ───────────────────────────────────────────────
    ax = axes[0]
    for si in range(N_STIMULI):
        lp = lick_sc_mean[si, 0]
        rp = float(VALUE_MATRIX[si, 0])
        if not np.isnan(lp):
            ax.scatter(rp, lp, color=bar_colors[si], s=120, zorder=3,
                       label=stimuli[si])
    ax.axhline(0.5, color="gray", lw=0.8, linestyle="--", alpha=0.6)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Reward probability")
    ax.set_ylabel("Lick probability")
    ax.set_title(f"Calibration  (Spearman r = {r_sp:.3f})")
    ax.legend(fontsize=9)

    # ── per-stim lick bars ────────────────────────────────────────────────
    ax = axes[1]
    ax.bar(x, lick_sc_mean[:, 0], color=bar_colors, alpha=0.85, width=0.5, zorder=2)
    ax.errorbar(x, lick_sc_mean[:, 0], yerr=lick_sc_std[:, 0],
                fmt="none", color="black", capsize=5, lw=1.2, zorder=3)
    for si in range(N_STIMULI):
        ax.plot([si - 0.25, si + 0.25], [VALUE_MATRIX[si, 0]] * 2,
                color="black", lw=2.5, zorder=5)
    ax.axhline(0.5, color="gray", lw=0.8, linestyle="--", alpha=0.6,
               label="target mid = 0.5")
    ax.set_xticks(x); ax.set_xticklabels(["low (v=0)", "mid (v=0.5)", "high (v=1)"])
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel("Lick probability  (mean ± SD across seeds)")
    ax.set_title(
        f"Per-stimulus lick prob  (black lines = GT)\n"
        f"mid lick prob = {mid_lp:.3f}  |  error from 0.5 = {abs(mid_lp - 0.5):.3f}"
    )
    ax.legend(fontsize=8)

    # ── selectivity proportions ───────────────────────────────────────────
    ax = axes[2]
    x3    = np.arange(3)
    means = [sel_mean[k] * 100 for k in sel_keys]
    stds  = [sel_std[k]  * 100 for k in sel_keys]
    ax.bar(x3, means, color=bar_colors, alpha=0.85, width=0.5, zorder=2)
    ax.errorbar(x3, means, yerr=stds,
                fmt="none", color="black", capsize=5, lw=1.2, zorder=3)
    ax.set_xticks(x3)
    ax.set_xticklabels(["Prefer low\n(v=0)", "Prefer mid\n(v=0.5)", "Prefer high\n(v=1)"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("% selective units  (mean ± SD)")
    mono_str = "monotonic" if sel_mono else "NOT monotonic"
    ax.set_title(
        f"Value selectivity  [{mono_str}]\n"
        f"SI ≥ {SI_THRESHOLD}  ·  mean n_sel = {mean_nsel:.0f}"
    )

    n_seeds   = len(lick_sc_runs)
    param_str = "  ".join(f"{k}={v}" for k, v in combo_params.items())
    fig.suptitle(
        f"Combo {combo_idx:02d}: {param_str}   [{n_seeds} seed{'s' if n_seeds != 1 else ''}]",
        y=1.02,
    )
    plt.tight_layout()

    safe_tag = "_".join(
        f"{k}{str(v).replace('-', 'm').replace('.', 'p')}"
        for k, v in combo_params.items()
    )
    fig_path = out_dir / f"mid_sweep_{ts}_combo{combo_idx:02d}_{safe_tag}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


# =============================================================================
# SUMMARY FIGURE
# =============================================================================

def _plot_summary(df, param_keys, out_dir, ts):
    """2-D heatmaps of key metrics across the 2-param sweep grid.

    Args:
        df: Full results DataFrame.
        param_keys: List of swept parameter names.
        out_dir: Save directory.
        ts: Timestamp string.
    """
    metric_cols = [
        "mid_lick_error", "mid_lick_prob", "spearman_r",
        "monotonicity", "pct_reward_consumed", "false_alarm_rate",
        "sel_monotonicity", "frac_low", "frac_mid", "frac_high",
        "combined_score",
    ]
    metric_labels = {
        "mid_lick_error":       "|mid lick − 0.5|  (↓ better)",
        "mid_lick_prob":        "Mid lick prob  (target = 0.5)",
        "spearman_r":           "Spearman r  (↑ better)",
        "monotonicity":         "Lick monotonicity  (↑ better)",
        "pct_reward_consumed":  "Reward consumed %  (↑ better)",
        "false_alarm_rate":     "False alarm %  (↓ better)",
        "sel_monotonicity":     "Selectivity monotonic  (↑ better)",
        "frac_low":             "Frac prefer low  (↓ better)",
        "frac_mid":             "Frac prefer mid",
        "frac_high":            "Frac prefer high  (↑ better)",
        "combined_score":       "Combined score  (↑ better)",
    }
    better_high = {
        "spearman_r", "monotonicity", "pct_reward_consumed",
        "sel_monotonicity", "frac_high", "frac_mid", "combined_score",
    }

    mean_df = df.groupby(param_keys)[metric_cols].mean().reset_index()

    if len(param_keys) == 2:
        pk0, pk1 = param_keys
        v0 = sorted(mean_df[pk0].unique())
        v1 = sorted(mean_df[pk1].unique())
        n_met = len(metric_cols)
        fig, axes = plt.subplots(2, (n_met + 1) // 2,
                                 figsize=(4 * ((n_met + 1) // 2), 7))
        axes = axes.flat
        for ax, mc in zip(axes, metric_cols):
            grid = np.full((len(v0), len(v1)), np.nan)
            for i, a in enumerate(v0):
                for j, b in enumerate(v1):
                    row = mean_df[(mean_df[pk0] == a) & (mean_df[pk1] == b)]
                    if len(row):
                        grid[i, j] = row[mc].values[0]
            # for mid_lick_error: green = low (good); for others standard
            if mc == "mid_lick_error":
                cmap = "RdYlGn_r"
            elif mc == "mid_lick_prob":
                # diverging around 0.5 — use custom normalisation below
                cmap = "RdYlGn"
            else:
                cmap = "RdYlGn" if mc in better_high else "RdYlGn_r"
            im = ax.imshow(grid, cmap=cmap, aspect="auto")
            ax.set_xticks(range(len(v1)))
            ax.set_xticklabels([str(b) for b in v1], fontsize=8)
            ax.set_yticks(range(len(v0)))
            ax.set_yticklabels([str(a) for a in v0], fontsize=8)
            ax.set_xlabel(pk1, fontsize=8); ax.set_ylabel(pk0, fontsize=8)
            ax.set_title(metric_labels.get(mc, mc), fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.8)
            for i in range(len(v0)):
                for j in range(len(v1)):
                    if not np.isnan(grid[i, j]):
                        ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                                fontsize=7, color="white")
        for ax in list(axes)[n_met:]:
            ax.set_visible(False)
        fig.suptitle(
            f"Mid-lick calibration sweep  (mean over {N_SEEDS} seeds)\n"
            f"{pk0} × {pk1}",
            y=1.02, fontsize=10,
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        mean_df["label"] = mean_df[param_keys].astype(str).agg(" | ".join, axis=1)
        mean_df_s = mean_df.sort_values("mid_lick_error")
        ax.bar(range(len(mean_df_s)), mean_df_s["mid_lick_error"], color="steelblue")
        ax.set_xticks(range(len(mean_df_s)))
        ax.set_xticklabels(mean_df_s["label"], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("|mid lick − 0.5|")
        ax.set_title("Mid-lick calibration sweep — ranked by error")

    plt.tight_layout()
    fig_path = out_dir / f"mid_sweep_{ts}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Summary figure  → {fig_path}")


# =============================================================================
# LICK + SELECTIVITY GRID HEATMAPS
# =============================================================================

def _plot_lick_sel_grid(df, lick_sc_by_combo, sel_by_combo,
                        param_keys, out_dir, ts):
    """Grid of mini bar charts: lick probs and selectivity proportions per combo.

    Rows = values of param_keys[0], columns = values of param_keys[1].
    Each cell contains two small bar charts (lick probs on top, selectivity on
    the bottom).

    Args:
        df: Full results DataFrame.
        lick_sc_by_combo: defaultdict mapping combo tuple → list of lick_sc arrays.
        sel_by_combo: defaultdict mapping combo tuple → list of sel dicts.
        param_keys: List of the two swept parameter names.
        out_dir: Save directory.
        ts: Timestamp string.
    """
    if len(param_keys) != 2:
        return   # only makes sense for a 2-D sweep

    pk0, pk1 = param_keys
    mean_df  = df.groupby(param_keys).mean().reset_index()
    v0       = sorted(mean_df[pk0].unique())
    v1       = sorted(mean_df[pk1].unique())
    nr, nc   = len(v0), len(v1)

    bar_colors = ["#4C72B0", "#8C8C8C", "#DD8452"]
    cell_h, cell_w = 2.8, 2.6

    fig, axes = plt.subplots(
        nr * 2, nc,
        figsize=(cell_w * nc, cell_h * nr),
        gridspec_kw={"hspace": 0.05, "wspace": 0.35},
        squeeze=False,
    )

    for ri, a in enumerate(v0):
        for ci, b in enumerate(v1):
            combo = (a, b)
            ax_lick = axes[ri * 2,     ci]
            ax_sel  = axes[ri * 2 + 1, ci]

            # ── lick probs ────────────────────────────────────────────────
            lick_runs = lick_sc_by_combo.get(combo, [])
            if lick_runs:
                arr = np.array([r[:, 0].tolist() for r in lick_runs], dtype=float)
                lp_mean = np.nanmean(arr, 0)
                lp_sem  = np.nanstd(arr, 0) / max(np.sqrt(len(arr)), 1)
            else:
                lp_mean = np.full(N_STIMULI, np.nan)
                lp_sem  = np.zeros(N_STIMULI)

            x = np.arange(N_STIMULI)
            ax_lick.bar(x, lp_mean, color=bar_colors, alpha=0.85, width=0.6, zorder=2)
            ax_lick.errorbar(x, lp_mean, yerr=lp_sem,
                             fmt="none", color="black", capsize=3, lw=1, zorder=3)
            for si in range(N_STIMULI):
                ax_lick.plot([si - 0.3, si + 0.3], [VALUE_MATRIX[si, 0]] * 2,
                             color="black", lw=2, zorder=5)
            ax_lick.axhline(0.5, color="gray", lw=0.6, linestyle="--", alpha=0.5)
            ax_lick.set_ylim(-0.05, 1.15)
            ax_lick.set_xticks(x)
            ax_lick.set_xticklabels(["lo", "mi", "hi"], fontsize=7)
            ax_lick.tick_params(axis="y", labelsize=6)
            mid_lp = lp_mean[MID_STIM]
            ax_lick.set_title(f"mid={mid_lp:.2f}", fontsize=7, pad=2)

            # ── selectivity ───────────────────────────────────────────────
            sel_runs = sel_by_combo.get(combo, [])
            if sel_runs:
                fracs = np.array(
                    [[s["frac_low"], s["frac_mid"], s["frac_high"]] for s in sel_runs],
                    dtype=float,
                )
                fr_mean = np.nanmean(fracs, 0) * 100
                fr_sem  = np.nanstd(fracs, 0)  / max(np.sqrt(len(fracs)), 1) * 100
                sel_mono = bool(fr_mean[0] < fr_mean[1] < fr_mean[2])
            else:
                fr_mean  = np.full(3, np.nan)
                fr_sem   = np.zeros(3)
                sel_mono = False

            x3 = np.arange(3)
            edge_colors = ["#222222" if sel_mono else "red"] * 3
            ax_sel.bar(x3, fr_mean, color=bar_colors, alpha=0.85, width=0.6, zorder=2,
                       edgecolor=edge_colors, linewidth=1.2 if sel_mono else 1.8)
            ax_sel.errorbar(x3, fr_mean, yerr=fr_sem,
                            fmt="none", color="black", capsize=3, lw=1, zorder=3)
            ax_sel.set_ylim(0, 105)
            ax_sel.set_xticks(x3)
            ax_sel.set_xticklabels(["lo", "mi", "hi"], fontsize=7)
            ax_sel.tick_params(axis="y", labelsize=6)
            mono_str = "✓" if sel_mono else "✗"
            ax_sel.set_title(f"sel {mono_str}", fontsize=7, pad=2)

            # row/col labels on edge cells only
            if ci == 0:
                ax_lick.set_ylabel(f"{pk0}={a}\nlick p", fontsize=7)
                ax_sel.set_ylabel("sel %", fontsize=7)
            if ri == 0:
                ax_lick.set_title(f"{pk1}={b}\nmid={mid_lp:.2f}", fontsize=7, pad=2)

    fig.suptitle(
        f"Lick calibration (top) & Selectivity (bottom)  ·  mean ± SEM over {N_SEEDS} seeds\n"
        f"rows = {pk0}   ·   cols = {pk1}   ·  black lines = GT reward prob  ·  sel border: green=monotonic, red=not",
        fontsize=9, y=1.01,
    )
    plt.tight_layout()
    fig_path = out_dir / f"mid_sweep_{ts}_grid.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grid figure      → {fig_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    device  = torch.device(args.device)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"mid_sweep_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output → {out_dir}")

    param_keys   = list(SWEEP.keys())
    param_values = list(SWEEP.values())
    combos       = list(itertools.product(*param_values))
    n_runs       = len(combos) * N_SEEDS

    print(f"Task: {N_STIMULI}s{N_CONTEXTS}c  |  target: lick_prob(s_mid) ≈ 0.5")
    print(f"Sweep: {param_keys}")
    print(f"Combos: {len(combos)}  ×  Seeds: {N_SEEDS}  =  {n_runs} runs")
    print(f"Device: {device}\n")

    rows             = []
    lick_sc_by_combo = defaultdict(list)
    sel_by_combo     = defaultdict(list)

    run_i = 0
    for combo in combos:
        params = dict(zip(param_keys, combo))
        tag    = "  ".join(f"{k}={v}" for k, v in params.items())
        for s in range(N_SEEDS):
            seed  = BASE_SEED + s
            run_i += 1
            print(f"[{run_i:>3}/{n_runs}]  {tag}  seed={seed}", end="  ", flush=True)
            metrics = run_one(**params, seed=seed, device=device)
            lick_sc_by_combo[combo].append(metrics.pop("lick_sc"))
            sel_by_combo[combo].append({
                k: metrics[k]
                for k in ("frac_low", "frac_mid", "frac_high", "n_selective")
            })
            row = {**params, "seed": seed, **metrics}
            rows.append(row)
            if metrics["diverged"]:
                print("DIVERGED")
            else:
                print(f"mid={metrics['mid_lick_prob']:.3f}  "
                      f"err={metrics['mid_lick_error']:.3f}  "
                      f"r={metrics['spearman_r']:.3f}  "
                      f"sel_mono={int(metrics['sel_monotonicity'])}  "
                      f"[lo={metrics['frac_low']:.2f} "
                      f"mi={metrics['frac_mid']:.2f} "
                      f"hi={metrics['frac_high']:.2f}]")

    df = pd.DataFrame(rows)

    # ── summary table ─────────────────────────────────────────────────────
    table_cols = [
        "mid_lick_error", "mid_lick_prob", "spearman_r",
        "monotonicity", "pct_reward_consumed", "false_alarm_rate",
        "sel_monotonicity", "frac_low", "frac_mid", "frac_high",
        "combined_score",
    ]
    summary = (
        df.groupby(param_keys)[table_cols]
        .agg(["mean", "std"])
        .round(3)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary["n_diverged"] = df.groupby(param_keys)["diverged"].sum().values

    # rank by combined_score descending (higher = better)
    summary = summary.sort_values("combined_score_mean", ascending=False)

    print("\n" + "=" * 72)
    print("SUMMARY  (sorted by combined_score = sel_monotonicity − mid_lick_error, descending)")
    print("=" * 72)

    display_rows = []
    for idx, row in summary.iterrows():
        combo_label = "  ".join(
            f"{k}={v}" for k, v in zip(
                param_keys, (idx if isinstance(idx, tuple) else (idx,))
            )
        )
        entry = {"params": combo_label}
        for mc in table_cols:
            mu = row[f"{mc}_mean"]
            sd = row[f"{mc}_std"]
            entry[mc] = f"{mu:.3f} ± {sd:.3f}" if not np.isnan(sd) else f"{mu:.3f}"
        entry["n_diverged"] = int(row["n_diverged"])
        display_rows.append(entry)

    disp_df = pd.DataFrame(display_rows).set_index("params")
    pd.set_option("display.max_colwidth", 20)
    pd.set_option("display.width", 220)
    print(disp_df.to_string())

    # ── save CSV ──────────────────────────────────────────────────────────
    csv_path = out_dir / f"mid_sweep_{ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull results → {csv_path}")

    # ── summary figure ────────────────────────────────────────────────────
    _plot_summary(df, param_keys, out_dir, ts)

    # ── lick + selectivity grid ───────────────────────────────────────────
    _plot_lick_sel_grid(
        df, lick_sc_by_combo, sel_by_combo, param_keys, out_dir, ts
    )

    # ── per-combo figures ─────────────────────────────────────────────────
    print("\nSaving per-combo figures...")
    for ci, combo in enumerate(combos):
        params   = dict(zip(param_keys, combo))
        fig_path = _plot_per_combo(
            params, lick_sc_by_combo[combo], sel_by_combo[combo], out_dir, ts, ci
        )
        print(f"  combo {ci:02d} → {fig_path.name}")


if __name__ == "__main__":
    main()
