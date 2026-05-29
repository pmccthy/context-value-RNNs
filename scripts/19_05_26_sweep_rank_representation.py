#!/usr/bin/env python
"""
Rank × hidden-size sweep measuring value representation quality.

Representation metrics (key outcomes)
    held_out_r        — held-out cross-context decode: train on anchor stimuli,
                        test on swap stimuli; mean off-diagonal Pearson r
    ccgp_ridge_anchor — train=all stim, test=anchor only, mean cross-context Pearson r
    ccgp_ridge_swap   — train=all stim, test=swap only,   mean cross-context Pearson r
    ccgp_bin_anchor   — binary SVM, train=all, test=anchor, mean cross-context accuracy
    ccgp_bin_swap     — binary SVM, train=all, test=swap,   mean cross-context accuracy
    rsa_identity_r    — RSA Spearman r (neural RDM vs identity model)
    rsa_value_r       — RSA Spearman r (neural RDM vs value model)

Performance check (gate: ensure model is performing before reading representations)
    spearman_r          — Spearman r lick prob vs reward prob (calibration)
    pct_reward_consumed — % rewarded trials where the agent licked
    false_alarm_rate    — % unrewarded trials where the agent licked

Usage
-----
    python scripts/19_05_26_sweep_rank_representation.py
    python scripts/19_05_26_sweep_rank_representation.py --seeds 5 --device mps
    python scripts/19_05_26_sweep_rank_representation.py --out-dir results/rank_sweep
"""
from __future__ import annotations

import argparse
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
from cxval.models import RNN, LowRankRNN, ActorCritic
from cxval.agents import Agent
from cxval.analysis import (
    mean_offdiag,
    held_out_stim_decode,
    cross_group_decode,
    cross_group_binary_decode,
    compute_neural_rdm,
    compute_model_rdms,
    rsa_compare,
)

# =============================================================================
# SWEEP CONFIGURATION  ← edit here
# =============================================================================

# (hidden_size, rank) — rank=None uses a full-rank RNN as baseline
GRID: list[tuple[int, int | None]] = [
    (32,   1), (32,   2), (32,   4), (32,   8), (32,  16), (32,  None),
    (64,   1), (64,   2), (64,   4), (64,   8), (64,  16), (64,  None),
    (128,  1), (128,  2), (128,  4), (128,  8), (128, 16), (128, None),
]

# Trials per phase per hidden size — smaller models may need more experience
TRIALS_PER_HIDDEN: dict[int, int] = {
    32:  600,
    64:  400,
    128: 400,
}
TRIALS_PER_HIDDEN_DEFAULT = 400

# =============================================================================
# TASK  (15-stimulus / 2-context swap+anchors)
# =============================================================================

VALUE_MATRIX = np.array([
    [0.0, 1.0],   # s0:  swap low→high
    [0.0, 1.0],   # s1:  swap low→high
    [0.0, 1.0],   # s2:  swap low→high
    [0.5, 0.5],   # s3:  mid
    [0.5, 0.5],   # s4:  mid
    [0.5, 0.5],   # s5:  mid
    [1.0, 0.0],   # s6:  swap high→low
    [1.0, 0.0],   # s7:  swap high→low
    [1.0, 0.0],   # s8:  swap high→low
    [0.0, 0.0],   # s9:  anchor low
    [0.0, 0.0],   # s10: anchor low
    [0.0, 0.0],   # s11: anchor low
    [1.0, 1.0],   # s12: anchor high
    [1.0, 1.0],   # s13: anchor high
    [1.0, 1.0],   # s14: anchor high
], dtype=np.float32)

ANCHOR_IDX  = [9, 10, 11, 12, 13, 14]
SWAP_IDX    = [0, 1, 2, 6, 7, 8]
N_STIMULI   = VALUE_MATRIX.shape[0]
N_CONTEXTS  = VALUE_MATRIX.shape[1]

# =============================================================================
# FIXED HYPERPARAMETERS
# =============================================================================

RECURRENT_GAIN     = 0.9   # full-rank RNN only
GAIN               = 0.0   # LowRankRNN J_0 scale (0 = pure low-rank)
POLICY_CLIP        = 0.05
LICK_COST          = 0.0
BPTT_LEN           = 10
UPDATE_EVERY       = 10
GAMMA              = 0.9
LR                 = 9e-4
VALUE_COEF         = 0.5
ENTROPY_COEF       = 0.01
GRAD_CLIP          = 1.0
PHASES_PER_CONTEXT = 1
CONTEXT_REPS       = 30
STIM_TIMESTEPS     = 5
REWARD_TIMESTEPS   = 3
ITI_TIMESTEPS      = (3, 8)

FULL_RANK_KEY = -1   # integer sentinel stored in CSV for rank=None

# =============================================================================
# HELPERS
# =============================================================================

def compute_returns(rewards: list, bootstrap_value: float, gamma: float) -> list:
    returns, R = [], float(bootstrap_value)
    for r in reversed(rewards):
        R = r + gamma * R
        returns.append(R)
    returns.reverse()
    return returns


def build_backbone(obs_dim: int, hidden_size: int, rank: int | None) -> nn.Module:
    if rank is None:
        return RNN(input_size=obs_dim, hidden_size=hidden_size, output_size=1,
                   recurrent_gain=RECURRENT_GAIN)
    return LowRankRNN(input_size=obs_dim, hidden_size=hidden_size, output_size=1,
                      rank=rank, gain=GAIN)


def weights_nan(model: nn.Module) -> bool:
    return any(torch.isnan(p).any() for p in model.parameters())


# =============================================================================
# SINGLE RUN
# =============================================================================

def run_one(hidden_size: int, rank: int | None, seed: int, device: torch.device) -> dict:
    nan_row = dict(
        held_out_r=np.nan, held_out_r_swap_to_anchor=np.nan,
        ccgp_ridge_anchor=np.nan, ccgp_ridge_swap=np.nan,
        ccgp_bin_anchor=np.nan,   ccgp_bin_swap=np.nan,
        rsa_identity_r=np.nan,    rsa_value_r=np.nan,
        spearman_r=np.nan,
        lick_rate_high=np.nan, lick_rate_mid=np.nan, lick_rate_low=np.nan,
        pct_reward_consumed=np.nan, false_alarm_rate=np.nan,
        diverged=True,
    )

    trials_per_phase = TRIALS_PER_HIDDEN.get(hidden_size, TRIALS_PER_HIDDEN_DEFAULT)

    # ── task ──────────────────────────────────────────────────────────────
    stim_seq = StimulusSequence(
        value_matrix=VALUE_MATRIX,
        trials_per_phase=trials_per_phase,
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
        states=states, reward_availability=reward_availability,
        reward_lick=1.0, reward_no_lick=0.0, reward_lick_miss=-1.0,
        lick_cost=LICK_COST,
    )

    # ── model ─────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    backbone     = build_backbone(obs_dim, hidden_size, rank)
    actor_critic = ActorCritic(backbone=backbone, num_actions=2,
                               policy_clip=POLICY_CLIP).to(device)
    optimizer    = torch.optim.Adam(actor_critic.parameters(), lr=LR)

    # ── training ──────────────────────────────────────────────────────────
    actor_critic.train()
    optimizer.zero_grad()
    obs, _  = env.reset()
    hidden  = None
    log_probs_buf, values_buf, rewards_buf, entropies_buf = [], [], [], []
    t_in_window = 0
    diverged    = False
    t_global    = 0
    train_rows: list[dict] = []
    first_rew_ts_lookup = {
        trial["reward_window"][0]: ti
        for ti, trial in enumerate(state_seq.trial_structure)
    }

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

            if t_global in first_rew_ts_lookup:
                ti_tr = first_rew_ts_lookup[t_global]
                tr    = state_seq.trial_structure[ti_tr]
                train_rows.append({
                    "trial_idx": ti_tr,
                    "stimulus":  tr["stimulus"],
                    "context":   tr["context"],
                    "value_gt":  float(VALUE_MATRIX[tr["stimulus"], tr["context"]]),
                    "licked":    int(action.item() == TaskEnv.LICK),
                    "value_est": float(value.item()),
                })

            obs, reward, done, _, _ = env.step(action.item())
            rewards_buf.append(reward)
            t_in_window += 1
            t_global    += 1

            if t_in_window % UPDATE_EVERY == 0 or done:
                bootstrap_v = 0.0
                if not done:
                    with torch.no_grad():
                        obs_n = torch.tensor(obs, dtype=torch.float32,
                                             device=device).unsqueeze(0)
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
                    if weights_nan(actor_critic):
                        diverged = True
                        break

    except Exception as exc:
        warnings.warn(f"Training crashed (h={hidden_size}, r={rank}, seed={seed}): {exc}")
        return nan_row, []

    if diverged or weights_nan(actor_critic):
        warnings.warn(f"NaN divergence (h={hidden_size}, r={rank}, seed={seed})")
        return nan_row, []

    # ── inference ─────────────────────────────────────────────────────────
    infer_stim_seq = StimulusSequence(
        value_matrix=VALUE_MATRIX,
        trials_per_phase=trials_per_phase,
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
        states=infer_states, reward_availability=infer_reward_avail,
        reward_lick=1.0, reward_no_lick=0.0, reward_lick_miss=-1.0,
        lick_cost=LICK_COST,
    )

    actor_critic.eval()
    agent = Agent(actor_critic, device=device)
    agent.reset()
    obs, _ = infer_env.reset()
    action_seq, hidden_list = [], []
    done = False
    while not done:
        action, _, _ = agent.act(obs)
        action_seq.append(action)
        hidden_list.append(agent.hidden.detach().squeeze(0))
        obs, _, done, _, _ = infer_env.step(action)

    infer_hidden = np.array(torch.stack(hidden_list).cpu().tolist(), dtype=np.float32)
    trial_struct = infer_state_seq.trial_structure
    ctx_arr      = np.array([t["context"]          for t in trial_struct])
    stim_arr     = np.array([t["stimulus"]         for t in trial_struct])
    ravail_arr   = np.array([t["reward_available"] for t in trial_struct], dtype=bool)

    activations = {
        "stim_hidden":     np.stack([
            infer_hidden[t["stim_window"][0]:t["stim_window"][1]] for t in trial_struct
        ]),
        "reward_hidden":   np.stack([
            infer_hidden[t["reward_window"][0]:t["reward_window"][1]] for t in trial_struct
        ]),
        "context":         ctx_arr,
        "stimulus":        stim_arr,
        "trial_structure": trial_struct,
    }

    # ── representation metrics ────────────────────────────────────────────

    # Held-out cross-context decode: train on anchors, test on swaps (and reverse)
    try:
        r_mat      = held_out_stim_decode(
            activations, "stim", "average", VALUE_MATRIX,
            train_stim_idx=ANCHOR_IDX, test_stim_idx=SWAP_IDX,
        )
        held_out_r = float(mean_offdiag(r_mat))
    except Exception:
        held_out_r = np.nan

    try:
        r_mat_rev              = held_out_stim_decode(
            activations, "stim", "average", VALUE_MATRIX,
            train_stim_idx=SWAP_IDX, test_stim_idx=ANCHOR_IDX,
        )
        held_out_r_swap_to_anchor = float(mean_offdiag(r_mat_rev))
    except Exception:
        held_out_r_swap_to_anchor = np.nan

    # CCGP Ridge: train=all, test={anchor, swap}
    try:
        cgd = cross_group_decode(
            activations, "stim", "average", VALUE_MATRIX,
            train_groups={"all": None},
            test_groups={"anchor": ANCHOR_IDX, "swap": SWAP_IDX},
            n_folds=5,
        )
        ccgp_ridge_anchor = float(mean_offdiag(cgd[("all", "anchor")]))
        ccgp_ridge_swap   = float(mean_offdiag(cgd[("all", "swap")]))
    except Exception:
        ccgp_ridge_anchor = ccgp_ridge_swap = np.nan

    # CCGP binary SVM: train=all, test={anchor, swap}
    try:
        cgd_bin = cross_group_binary_decode(
            activations, "stim", "average", VALUE_MATRIX,
            train_groups={"all": None},
            test_groups={"anchor": ANCHOR_IDX, "swap": SWAP_IDX},
            n_folds=5,
        )
        ccgp_bin_anchor = float(mean_offdiag(cgd_bin[("all", "anchor")]))
        ccgp_bin_swap   = float(mean_offdiag(cgd_bin[("all", "swap")]))
    except Exception:
        ccgp_bin_anchor = ccgp_bin_swap = np.nan

    # RSA: stim period
    try:
        neural_rdm, conditions = compute_neural_rdm(activations, "stim")
        id_rdm, val_rdm        = compute_model_rdms(conditions, VALUE_MATRIX)
        rsa_res                = rsa_compare(neural_rdm, {"identity": id_rdm, "value": val_rdm})
        rsa_identity_r         = float(rsa_res["identity"])
        rsa_value_r            = float(rsa_res["value"])
    except Exception:
        rsa_identity_r = rsa_value_r = np.nan

    # ── performance check ─────────────────────────────────────────────────
    VALUE_THRESHOLD = 0.25
    lp_flat, rp_flat = [], []
    rew_r_rates, rew_u_rates = [], []
    lick_high_rates, lick_mid_rates, lick_low_rates = [], [], []

    for ti, trial in enumerate(trial_struct):
        s, e = trial["reward_window"]
        rr   = float(np.mean([action_seq[t] == TaskEnv.LICK for t in range(s, e)])) \
               if e > s else np.nan
        if ravail_arr[ti]:
            rew_r_rates.append(rr)
        else:
            rew_u_rates.append(rr)

        val = float(VALUE_MATRIX[stim_arr[ti], ctx_arr[ti]])
        if val >= 1 - VALUE_THRESHOLD:
            lick_high_rates.append(rr)
        elif val <= VALUE_THRESHOLD:
            lick_low_rates.append(rr)
        else:
            lick_mid_rates.append(rr)

    for si in range(N_STIMULI):
        for ci in range(N_CONTEXTS):
            m = (stim_arr == si) & (ctx_arr == ci)
            if not m.any():
                continue
            lp = float(np.mean([
                action_seq[t["reward_window"][0]] == TaskEnv.LICK
                for t, flag in zip(trial_struct, m) if flag
            ]))
            lp_flat.append(lp)
            rp_flat.append(float(VALUE_MATRIX[si, ci]))

    spear_r        = float(spearmanr(rp_flat, lp_flat)[0]) if len(lp_flat) > 2 else np.nan
    lick_rate_high = float(np.nanmean(lick_high_rates))    if lick_high_rates else np.nan
    lick_rate_mid  = float(np.nanmean(lick_mid_rates))     if lick_mid_rates  else np.nan
    lick_rate_low  = float(np.nanmean(lick_low_rates))     if lick_low_rates  else np.nan
    pct_reward     = float(np.nanmean(rew_r_rates) * 100)  if rew_r_rates     else np.nan
    false_alarm    = float(np.nanmean(rew_u_rates) * 100)  if rew_u_rates     else np.nan

    return dict(
        held_out_r=held_out_r, held_out_r_swap_to_anchor=held_out_r_swap_to_anchor,
        ccgp_ridge_anchor=ccgp_ridge_anchor, ccgp_ridge_swap=ccgp_ridge_swap,
        ccgp_bin_anchor=ccgp_bin_anchor,     ccgp_bin_swap=ccgp_bin_swap,
        rsa_identity_r=rsa_identity_r,       rsa_value_r=rsa_value_r,
        spearman_r=spear_r,
        lick_rate_high=lick_rate_high, lick_rate_mid=lick_rate_mid, lick_rate_low=lick_rate_low,
        pct_reward_consumed=pct_reward, false_alarm_rate=false_alarm,
        diverged=False,
    ), train_rows


# =============================================================================
# PLOTTING
# =============================================================================

# (key, ylabel, ymin, ymax, chance_line)
REPR_METRICS = [
    ("held_out_r",        "Cross-ctx Pearson r",  -1,   1,    0.0),
    ("ccgp_ridge_anchor", "Cross-ctx Pearson r",  -1,   1,    0.0),
    ("ccgp_ridge_swap",   "Cross-ctx Pearson r",  -1,   1,    0.0),
    ("ccgp_bin_anchor",   "Cross-ctx accuracy",    0,   1,    0.5),
    ("ccgp_bin_swap",     "Cross-ctx accuracy",    0,   1,    0.5),
    ("rsa_identity_r",    "Spearman r",           -1,   1,    0.0),
    ("rsa_value_r",       "Spearman r",           -1,   1,    0.0),
]

REPR_TITLES = [
    "Held-out decode\n(train anchor → test swap)",
    "CCGP Ridge\n(train=all, test=anchor)",
    "CCGP Ridge\n(train=all, test=swap)",
    "CCGP SVM\n(train=all, test=anchor)",
    "CCGP SVM\n(train=all, test=swap)",
    "RSA — identity model",
    "RSA — value model",
]

PERF_METRICS = [
    ("spearman_r",          "Lick–value calibration\n(Spearman r, ↑ better → 1)",  -1,   1),
    ("lick_rate_high",      "Lick rate — high-value stim\n(↑ better → 1.0)",         0,   1),
    ("lick_rate_low",       "Lick rate — low-value stim\n(↓ better → 0.0)",          0,   1),
    ("pct_reward_consumed", "Reward consumed (%)\n(↑ better)",                        0, 100),
    ("false_alarm_rate",    "False alarm rate (%)\n(↓ better)",                       0, 100),
]


def _stats(df, h, rk, metric):
    vals = df.loc[(df["hidden_size"] == h) & (df["rank_key"] == rk), metric].dropna()
    return (float(vals.mean()), float(vals.std())) if len(vals) else (np.nan, np.nan)


def plot_results(df, hidden_sizes, rank_nums, out_dir, ts, n_seeds):
    h_colors = plt.cm.plasma(np.linspace(0.15, 0.80, len(hidden_sizes)))

    # ── representation metrics (2×4 grid, last panel = legend) ────────────
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axlist = list(axes.flat)

    for ax, (mc, ylabel, ymin, ymax, chance), title in zip(
        axlist, REPR_METRICS, REPR_TITLES
    ):
        for h, color in zip(hidden_sizes, h_colors):
            ys, es = [], []
            for rk in rank_nums:
                mu, sd = _stats(df, h, rk, mc)
                ys.append(mu); es.append(sd if not np.isnan(sd) else 0.0)
            ax.errorbar(rank_nums, ys, yerr=es, marker="o", color=color,
                        label=f"h={h}", linewidth=2, capsize=4, zorder=3)

            # Full-rank shown as a diamond at a gap to the right
            mu_fr, sd_fr = _stats(df, h, FULL_RANK_KEY, mc)
            if not np.isnan(mu_fr):
                x_fr = max(rank_nums) * 1.5
                ax.errorbar([x_fr], [mu_fr],
                            yerr=[sd_fr if not np.isnan(sd_fr) else 0.0],
                            marker="D", color=color, markersize=8,
                            linestyle=":", linewidth=1.5, capsize=4, zorder=3)

        ax.axhline(chance, color="gray", linestyle="--", lw=0.8, alpha=0.7)
        ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin))
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=8)

        xtick_vals = rank_nums[:]
        xtick_lbls = [str(r) for r in rank_nums]
        x_fr = max(rank_nums) * 1.5
        xtick_vals.append(x_fr)
        xtick_lbls.append("full")
        ax.set_xticks(xtick_vals)
        ax.set_xticklabels(xtick_lbls, fontsize=8)
        ax.set_xlabel("Rank")

    # Legend panel
    ax_leg = axlist[len(REPR_METRICS)]
    for h, color in zip(hidden_sizes, h_colors):
        ax_leg.plot([], [], color=color, lw=2,   marker="o", label=f"h={h} (low-rank)")
        ax_leg.plot([], [], color=color, lw=1.5, marker="D", linestyle=":", label=f"h={h} (full-rank)")
    ax_leg.legend(fontsize=8, loc="center", frameon=False)
    ax_leg.axis("off")

    fig.suptitle(
        f"Rank × hidden-size sweep  —  representation quality  (n={n_seeds} seeds)\n"
        "15-stimulus / 2-context swap+anchors task  ·  stim period  ·  average pooling",
        y=1.01, fontsize=10,
    )
    plt.tight_layout()
    p_repr = out_dir / f"rank_sweep_repr_{ts}.png"
    fig.savefig(p_repr, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Repr figure  → {p_repr}")

    # ── performance check ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(PERF_METRICS), figsize=(4 * len(PERF_METRICS), 3.5))
    all_rk_keys = rank_nums + [FULL_RANK_KEY]
    x_pos       = list(range(len(all_rk_keys)))
    x_lbls      = [str(r) for r in rank_nums] + ["full"]

    for ax, (mc, title, ymin, ymax) in zip(axes, PERF_METRICS):
        for h, color in zip(hidden_sizes, h_colors):
            ys, es = [], []
            for rk in all_rk_keys:
                mu, sd = _stats(df, h, rk, mc)
                ys.append(mu); es.append(sd if not np.isnan(sd) else 0.0)
            ax.errorbar(x_pos, ys, yerr=es, marker="o", color=color,
                        label=f"h={h}", linewidth=2, capsize=4)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_lbls, fontsize=8)
        ax.set_ylim(ymin, ymax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Rank")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Performance check  (n={n_seeds} seeds)  —  models below Spearman r ≈ 0.8 should be excluded",
        y=1.03, fontsize=9,
    )
    plt.tight_layout()
    p_perf = out_dir / f"rank_sweep_perf_{ts}.png"
    fig.savefig(p_perf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Perf figure  → {p_perf}")


# =============================================================================
# TRAINING CURVES
# =============================================================================

def plot_training_curves(train_data, hidden_sizes, rank_nums, out_dir, ts):
    """Per-config training curves grouped into high/mid/low value stimuli."""
    all_rk  = rank_nums + [FULL_RANK_KEY]
    rk_lbls = [str(r) for r in rank_nums] + ["full"]

    group_specs = [
        ("high", lambda v: v >= 1 - VALUE_THRESHOLD, "forestgreen", 1.0),
        ("mid",  lambda v: VALUE_THRESHOLD < v < 1 - VALUE_THRESHOLD, "darkorange", 0.5),
        ("low",  lambda v: v <= VALUE_THRESHOLD,      "crimson",      0.0),
    ]
    smooth_w = 30

    for h in hidden_sizes:
        ncols = len(all_rk)
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 3.5), sharey=True)
        axes = list(axes) if hasattr(axes, "__len__") else [axes]

        for ci, (rk, rk_lbl) in enumerate(zip(all_rk, rk_lbls)):
            ax       = axes[ci]
            seeds_td = train_data.get((h, rk), [])

            for grp_lbl, grp_fn, color, target in group_specs:
                seed_curves = []
                for seed_rows in seeds_td:
                    rows = [r for r in seed_rows if grp_fn(r["value_gt"])]
                    if not rows:
                        continue
                    trial_idxs = np.array([r["trial_idx"] for r in rows])
                    licked     = np.array([r["licked"] for r in rows], dtype=float)
                    if len(licked) >= smooth_w:
                        s = np.convolve(licked, np.ones(smooth_w) / smooth_w, mode="valid")
                        x = trial_idxs[smooth_w - 1:]
                    else:
                        s, x = licked, trial_idxs
                    ax.plot(x, s, color=color, alpha=0.3, lw=0.8)
                    seed_curves.append((x, s))

                if seed_curves:
                    n  = min(len(x) for x, _ in seed_curves)
                    mu = np.mean([y[:n] for _, y in seed_curves], axis=0)
                    ax.plot(seed_curves[0][0][:n], mu, color=color, lw=2.0, label=grp_lbl)

                ax.axhline(target, color=color, lw=0.6, ls=":", alpha=0.5)

            ax.set_title(rk_lbl, fontsize=9)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Training trial", fontsize=8)
            if ci == 0:
                ax.set_ylabel("Lick rate (smoothed)", fontsize=8)
            ax.legend(fontsize=7, frameon=False)

        fig.suptitle(
            f"Training curves — h={h}  (faint = per seed, bold = mean, dotted = target)",
            fontsize=9, y=1.01,
        )
        plt.tight_layout()
        p = out_dir / f"rank_sweep_train_curves_h{h}_{ts}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"Training curves h={h} → {p}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device",    default="cpu", help="torch device (cpu / cuda / mps)")
    parser.add_argument("--seeds",     type=int, default=3, help="Seeds per config")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--out-dir",   default="results/19_05_26_sweep_rank_representation", help="Output directory")
    args = parser.parse_args()

    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_runs = len(GRID) * args.seeds
    print(f"Rank × hidden-size sweep: {len(GRID)} configs × {args.seeds} seeds = {n_runs} runs")
    print(f"Device: {device}\n")
    for h, r in GRID:
        tpp = TRIALS_PER_HIDDEN.get(h, TRIALS_PER_HIDDEN_DEFAULT)
        print(f"  h={h:>3}  {'full RNN' if r is None else f'rank={r}':<10}  trials/phase={tpp}")
    print()

    rows       = []
    train_data = defaultdict(list)
    n_done     = 0
    for hidden_size, rank in GRID:
        tag = f"h={hidden_size}  {'full RNN' if rank is None else f'rank={rank}'}"
        for s in range(args.seeds):
            seed    = args.base_seed + s
            n_done += 1
            print(f"[{n_done:>3}/{n_runs}]  {tag:<22}  seed={seed}", end="  ", flush=True)
            metrics, t_rows = run_one(hidden_size, rank, seed, device)
            rk = FULL_RANK_KEY if rank is None else rank
            rows.append({
                "hidden_size": hidden_size,
                "rank":        rank,
                "rank_key":    rk,
                "seed":        seed,
                **metrics,
            })
            train_data[(hidden_size, rk)].append(t_rows)
            if metrics["diverged"]:
                print("DIVERGED")
            else:
                print(
                    f"held_out={metrics['held_out_r']:+.3f}  "
                    f"rsa_val={metrics['rsa_value_r']:+.3f}  "
                    f"cal={metrics['spearman_r']:.3f}"
                )

    df = pd.DataFrame(rows)

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_out = out_dir / f"rank_sweep_{ts}.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nResults → {csv_out}")

    # Summary table
    repr_keys = [m for m, *_ in REPR_METRICS]
    perf_keys = [m for m, *_ in PERF_METRICS]
    all_keys  = repr_keys + perf_keys
    summary   = df.groupby(["hidden_size", "rank_key"])[all_keys].agg(["mean", "std"]).round(3)
    summary.columns = ["_".join(c) for c in summary.columns]

    print("\n" + "=" * 90)
    print(f"{'Config':<24}  " + "  ".join(f"{k:<14}" for k in repr_keys + ["spearman_r"]))
    print("=" * 90)
    for (h, rk), row in summary.iterrows():
        tag   = f"h={h}  {'full' if rk == FULL_RANK_KEY else f'rank={rk}'}"
        vals  = "  ".join(
            f"{row[f'{k}_mean']:+.3f}±{row[f'{k}_std']:.3f}"
            if not np.isnan(row[f"{k}_mean"]) else "   nan        "
            for k in repr_keys + ["spearman_r"]
        )
        print(f"  {tag:<22}  {vals}")

    hidden_sizes = sorted(df["hidden_size"].unique().tolist())
    rank_nums    = sorted(r for r in df["rank_key"].unique() if r != FULL_RANK_KEY)
    plot_results(df, hidden_sizes, rank_nums, out_dir, ts, args.seeds)
    plot_training_curves(train_data, hidden_sizes, rank_nums, out_dir, ts)


if __name__ == "__main__":
    main()
