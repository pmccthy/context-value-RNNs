#!/usr/bin/env python
"""
Architecture sweep: hidden_size × rank for the 9-stimulus / 2-context lick task.

Investigates whether conjunctive vs factorised value representations arise as a
function of model capacity (hidden_size) and structural constraint (rank).

Key outputs
-----------
Value CCGP  — cross-context generalisation of value representations:
    * Ridge regression cross-context Pearson r  (all / swap / anchor stimuli)
    * Binary SVM cross-context accuracy         (all / swap / anchor stimuli)

Performance checks (gate for "model is actually working"):
    * Spearman r    — lick prob ↔ reward prob calibration
    * Monotonicity  — fraction of contexts where lick(low) < lick(mid) < lick(high)
    * ITI / stim lick rate  — selectivity outside reward window
    * % reward consumed / false alarm rate

Configuration
-------------
Edit GRID and TRIALS_PER_HIDDEN at the top, then run:
    python scripts/architecture_sweep.py [--device cpu|cuda] [--seeds 3] [--out-dir results]
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
from cxval.models import RNN, LowRankRNN, ActorCritic
from cxval.agents import Agent
from cxval.analysis import (
    filter_act_dict, mean_offdiag,
    value_decode_within, value_decode_cross,
    binary_value_decode_within, binary_value_decode_cross,
)

# =============================================================================
# SWEEP CONFIGURATION  ← edit here
# =============================================================================

# (hidden_size, rank) — rank=None uses a full-rank RNN as baseline
GRID: list[tuple[int, int | None]] = [
    (16,  1), (16,  2), (16,  4), (16,  None),
    (32,  1), (32,  2), (32,  4), (32,  None),
    (64,  1), (64,  2), (64,  4), (64,  None),
]

# Trials per phase for each hidden size.
# Smaller models learn slower — give them more experience to reach comparable
# asymptotic performance before declaring CCGP results.
TRIALS_PER_HIDDEN: dict[int, int] = {
    16: 600,
    32: 450,
    64: 300,
}
# Fallback if a hidden_size is not in the dict:
TRIALS_PER_HIDDEN_DEFAULT = 300

# =============================================================================
# FIXED HYPERPARAMETERS
# =============================================================================
RECURRENT_GAIN     = 0.9    # only used for full-rank RNN
GAIN               = 0.0    # J_0 scale for LowRankRNN (0 = pure low-rank)
POLICY_CLIP        = 0.05
LICK_COST          = 0.0
N_EPISODES         = 1
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

VALUE_MATRIX = np.array([
    [0.0, 1.0],   # s0: swap  low→high
    [0.5, 0.5],   # s1: fixed mid
    [1.0, 0.0],   # s2: swap  high→low
    [0.0, 1.0],   # s3: swap  low→high
    [1.0, 0.0],   # s4: swap  high→low
    [0.0, 0.0],   # s5: anchor low
    [0.0, 0.0],   # s6: anchor low
    [1.0, 1.0],   # s7: anchor high
    [1.0, 1.0],   # s8: anchor high
], dtype=np.float32)

N_STIMULI  = VALUE_MATRIX.shape[0]
N_CONTEXTS = VALUE_MATRIX.shape[1]

# stimulus groups for CCGP split
STIM_GROUPS = {
    "all":    None,
    "swap":   [0, 2, 3, 4],
    "anchor": [5, 6, 7, 8],
}

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


def monotonicity_score(lick_sc: np.ndarray, vm: np.ndarray) -> float:
    """Fraction of contexts where mean lick(low) < lick(mid) < lick(high)."""
    scores = []
    for ci in range(N_CONTEXTS):
        vals  = vm[:, ci]
        lows  = lick_sc[vals <= 0.25, ci]
        mids  = lick_sc[(vals > 0.4) & (vals < 0.6), ci]
        highs = lick_sc[vals >= 0.75, ci]
        if not (lows.size and mids.size and highs.size):
            continue
        scores.append(float(np.nanmean(lows) < np.nanmean(mids) < np.nanmean(highs)))
    return float(np.mean(scores)) if scores else np.nan


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
        ccgp_ridge_all=np.nan, ccgp_ridge_swap=np.nan, ccgp_ridge_anchor=np.nan,
        ccgp_bin_all=np.nan,   ccgp_bin_swap=np.nan,   ccgp_bin_anchor=np.nan,
        within_ridge=np.nan,   within_bin_all=np.nan,
        spearman_r=np.nan,     monotonicity=np.nan,
        iti_lick_rate=np.nan,  stim_lick_rate=np.nan,
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
    obs, _ = env.reset()
    hidden = None
    log_probs_buf, values_buf, rewards_buf, entropies_buf = [], [], [], []
    t_in_window = 0
    diverged = False

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

                returns_t    = torch.tensor(compute_returns(rewards_buf, bootstrap_v, GAMMA),
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
                loss        = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy_mean

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
        return nan_row

    if diverged or weights_nan(actor_critic):
        warnings.warn(f"NaN divergence (h={hidden_size}, r={rank}, seed={seed})")
        return nan_row

    # ── inference run ─────────────────────────────────────────────────────
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

    # ── build activations dict ────────────────────────────────────────────
    trial_struct = infer_state_seq.trial_structure
    ctx_arr   = np.array([t["context"]          for t in trial_struct])
    stim_arr  = np.array([t["stimulus"]         for t in trial_struct])
    ravail_arr = np.array([t["reward_available"] for t in trial_struct], dtype=bool)

    stim_hidden = np.stack([
        infer_hidden[t["stim_window"][0]:t["stim_window"][1]] for t in trial_struct
    ])

    activations = {
        "stim_hidden":      stim_hidden,
        "reward_hidden":    np.stack([
            infer_hidden[t["reward_window"][0]:t["reward_window"][1]] for t in trial_struct
        ]),
        "context":          ctx_arr,
        "stimulus":         stim_arr,
        "reward_available": ravail_arr,
        "trial_structure":  trial_struct,
    }

    # ── value CCGP (Ridge) ────────────────────────────────────────────────
    r_within_arr = value_decode_within(activations, period="stim", pooling="average",
                                       value_matrix=VALUE_MATRIX, n_folds=5)
    within_ridge = float(np.nanmean(r_within_arr))

    def _ccgp_ridge(act):
        try:
            r_cross = value_decode_cross(act, period="stim", pooling="average",
                                         value_matrix=VALUE_MATRIX)
            return float(mean_offdiag(r_cross))
        except Exception:
            return np.nan

    def _ccgp_bin(act):
        try:
            bc = binary_value_decode_cross(act, period="stim", pooling="average",
                                           value_matrix=VALUE_MATRIX)
            return float(mean_offdiag(bc))
        except Exception:
            return np.nan

    def _within_bin(act):
        try:
            bw = binary_value_decode_within(act, period="stim", pooling="average",
                                            value_matrix=VALUE_MATRIX, n_folds=5)
            return float(np.nanmean(bw))
        except Exception:
            return np.nan

    ccgp_ridge, ccgp_bin, within_bin = {}, {}, {}
    for gname, gidx in STIM_GROUPS.items():
        act_g = (filter_act_dict(activations, np.isin(activations["stimulus"], gidx))
                 if gidx is not None else activations)
        ccgp_ridge[gname] = _ccgp_ridge(act_g)
        ccgp_bin[gname]   = _ccgp_bin(act_g)
        if gname == "all":
            within_bin["all"] = _within_bin(act_g)

    # ── lick-value calibration & period selectivity ────────────────────────
    lick_sc = np.full((N_STIMULI, N_CONTEXTS), np.nan)
    lp_flat, rp_flat = [], []
    iti_rates, stim_rates, rew_r, rew_u = [], [], [], []

    for ti, trial in enumerate(trial_struct):
        si    = stim_arr[ti]
        ci    = ctx_arr[ti]
        ravail = ravail_arr[ti]

        def _lr(s, e):
            seg = [a == TaskEnv.LICK for a in action_seq[s:e]]
            return float(np.mean(seg)) if seg else np.nan

        iti_rates.append(_lr(*trial["iti_window"]))
        stim_rates.append(_lr(*trial["stim_window"]))
        rr = _lr(*trial["reward_window"])
        rew_r.append(rr if ravail  else np.nan)
        rew_u.append(rr if not ravail else np.nan)

    # lick_sc from first reward-window step (consistent with notebook)
    for si in range(N_STIMULI):
        for ci in range(N_CONTEXTS):
            m = (stim_arr == si) & (ctx_arr == ci)
            if not m.any():
                continue
            lp = float(np.mean([action_seq[t["reward_window"][0]] == TaskEnv.LICK
                                 for t, ms in zip(trial_struct, m) if ms]))
            lick_sc[si, ci] = lp
            lp_flat.append(lp)
            rp_flat.append(float(VALUE_MATRIX[si, ci]))

    spear_r = spearmanr(rp_flat, lp_flat)[0] if len(lp_flat) > 2 else np.nan
    mono    = monotonicity_score(lick_sc, VALUE_MATRIX)

    ravail_arr_np = ravail_arr
    pct_reward  = float(np.nanmean(np.array(rew_r, float)[ravail_arr_np])  * 100) \
                  if ravail_arr_np.any() else np.nan
    false_alarm = float(np.nanmean(np.array(rew_u, float)[~ravail_arr_np]) * 100) \
                  if (~ravail_arr_np).any() else np.nan

    return dict(
        ccgp_ridge_all=ccgp_ridge["all"],
        ccgp_ridge_swap=ccgp_ridge["swap"],
        ccgp_ridge_anchor=ccgp_ridge["anchor"],
        ccgp_bin_all=ccgp_bin["all"],
        ccgp_bin_swap=ccgp_bin["swap"],
        ccgp_bin_anchor=ccgp_bin["anchor"],
        within_ridge=within_ridge,
        within_bin_all=within_bin["all"],
        spearman_r=spear_r,
        monotonicity=mono,
        iti_lick_rate=float(np.nanmean(iti_rates)),
        stim_lick_rate=float(np.nanmean(stim_rates)),
        pct_reward_consumed=pct_reward,
        false_alarm_rate=false_alarm,
        diverged=False,
    )


# =============================================================================
# SUMMARY FIGURES
# =============================================================================

CCGP_METRICS = [
    ("ccgp_ridge_all",    "Ridge CCGP — all stim\n(Pearson r, ↑ better)"),
    ("ccgp_ridge_swap",   "Ridge CCGP — swap stim\n(Pearson r, ↑ better)"),
    ("ccgp_ridge_anchor", "Ridge CCGP — anchor stim\n(Pearson r, ↑ better)"),
    ("ccgp_bin_all",      "Binary SVM CCGP — all\n(accuracy, ↑ better, chance=0.5)"),
    ("ccgp_bin_swap",     "Binary SVM CCGP — swap\n(accuracy, ↑ better, chance=0.5)"),
    ("ccgp_bin_anchor",   "Binary SVM CCGP — anchor\n(accuracy, ↑ better, chance=0.5)"),
]

PERF_METRICS = [
    ("spearman_r",          "Spearman r\n(lick↔value, ↑ better)"),
    ("monotonicity",        "Monotonicity\nlow<mid<high (↑ better)"),
    ("pct_reward_consumed", "Reward consumed (%)\n(↑ better)"),
    ("iti_lick_rate",       "ITI lick rate\n(↓ better → 0)"),
    ("stim_lick_rate",      "Stim lick rate\n(↓ better → 0)"),
    ("false_alarm_rate",    "False alarm rate (%)\n(↓ better → 0)"),
]


def _cell_mean(df, h, rk, metric):
    """Mean of metric for a given (hidden_size, rank_key) cell, skipping NaN."""
    vals = df.loc[(df["hidden_size"] == h) & (df["rank_key"] == rk), metric].dropna()
    return float(vals.mean()) if len(vals) else np.nan


def _heatmap_panel(ax, df, hidden_sizes, ranks_plot, metric, title,
                   vmin, vmax, cmap, fmt=".2f"):
    """Draw one heatmap: rows=hidden_size, cols=rank label.

    rank_key=-1 means full-rank RNN (integer sentinel that survives CSV round-trips).
    """
    rank_labels = ["full\nRNN" if r == -1 else f"rank {r}" for r in ranks_plot]
    grid = np.full((len(hidden_sizes), len(ranks_plot)), np.nan)
    for i, h in enumerate(hidden_sizes):
        for j, r in enumerate(ranks_plot):
            grid[i, j] = _cell_mean(df, h, r, metric)

    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(ranks_plot))); ax.set_xticklabels(rank_labels, fontsize=7)
    ax.set_yticks(range(len(hidden_sizes)))
    ax.set_yticklabels([f"h={h}" for h in hidden_sizes], fontsize=7)
    ax.set_title(title, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    for i in range(len(hidden_sizes)):
        for j in range(len(ranks_plot)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                        fontsize=7, color="white" if abs(v - (vmin + vmax) / 2) > (vmax - vmin) * 0.3
                        else "black")


def plot_results(df, hidden_sizes, ranks_all, out_dir, ts, n_seeds):
    # -1 = full-rank RNN sentinel; sort numerically, full-rank last
    ranks_plot = sorted(set(ranks_all), key=lambda r: 9999 if r == -1 else r)

    # ── CCGP figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (mc, label) in zip(axes.flat, CCGP_METRICS):
        is_ridge = "ridge" in mc
        vmin, vmax = (-1, 1) if is_ridge else (0, 1)
        cmap = "RdYlGn"
        _heatmap_panel(ax, df, hidden_sizes, ranks_plot, mc, label, vmin, vmax, cmap)

    fig.suptitle(
        f"Value CCGP by architecture  (mean over {n_seeds} seeds)\n"
        f"Inference activations, stim period, average pooling",
        y=1.02, fontsize=11,
    )
    try:
        plt.tight_layout()
    except Exception:
        pass
    p = out_dir / f"arch_sweep_ccgp_{ts}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"CCGP figure  → {p}")

    # ── performance figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    perf_cfg = [
        ("spearman_r",          -1, 1,   "RdYlGn"),
        ("monotonicity",         0, 1,   "RdYlGn"),
        ("pct_reward_consumed",  0, 100, "RdYlGn"),
        ("iti_lick_rate",        0, 1,   "RdYlGn_r"),
        ("stim_lick_rate",       0, 1,   "RdYlGn_r"),
        ("false_alarm_rate",     0, 100, "RdYlGn_r"),
    ]
    for ax, (mc, label), (_, vmin, vmax, cmap) in zip(axes.flat, PERF_METRICS, perf_cfg):
        _heatmap_panel(ax, df, hidden_sizes, ranks_plot, mc, label, vmin, vmax, cmap)

    fig.suptitle(
        f"Performance metrics by architecture  (mean over {n_seeds} seeds)\n"
        f"Higher is better for top row; lower is better for bottom row",
        y=1.02, fontsize=11,
    )
    plt.tight_layout()
    p = out_dir / f"arch_sweep_perf_{ts}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Perf figure  → {p}")

    # ── per-group line plot: CCGP Ridge by rank for each hidden_size ───────
    # (good for seeing the rank effect within each capacity)
    rank_nums = [r for r in ranks_plot if r != -1]
    if rank_nums:
        fig, axes = plt.subplots(1, len(hidden_sizes), figsize=(5 * len(hidden_sizes), 4),
                                 sharey=True)
        if len(hidden_sizes) == 1:
            axes = [axes]
        colors = {"all": "black", "swap": "tomato", "anchor": "steelblue"}
        for ax, h in zip(axes, hidden_sizes):
            trials = TRIALS_PER_HIDDEN.get(h, TRIALS_PER_HIDDEN_DEFAULT)
            for gname, gc in colors.items():
                mc = f"ccgp_ridge_{gname}"
                ys, es = [], []
                for r in rank_nums:
                    ys.append(_cell_mean(df, h, r, mc))
                    es.append(df.loc[(df["hidden_size"] == h) & (df["rank_key"] == r), mc].std())
                ax.errorbar(rank_nums, ys, yerr=es, marker="o", color=gc,
                            label=gname, linewidth=1.5, capsize=4)
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Rank")
            ax.set_title(f"h={h}  (trials/phase={trials})")
            ax.set_xticks(rank_nums)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Cross-context Pearson r  (Ridge CCGP)")
        fig.suptitle("Ridge CCGP by rank — stim group comparison", y=1.02)
        plt.tight_layout()
        p = out_dir / f"arch_sweep_rank_effect_{ts}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Rank effect  → {p}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--seeds",   type=int, default=3,
                        help="Number of seeds per (hidden_size, rank) combo")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_runs = len(GRID) * args.seeds
    print(f"Architecture sweep: {len(GRID)} combos × {args.seeds} seeds = {n_runs} runs")
    print(f"Device: {device}\n")
    for h, r in GRID:
        label  = f"rank={r}" if r is not None else "full RNN"
        trials = TRIALS_PER_HIDDEN.get(h, TRIALS_PER_HIDDEN_DEFAULT)
        print(f"  h={h:>3}  {label:<10}  trials/phase={trials}")
    print()

    rows = []
    run_i = 0
    for hidden_size, rank in GRID:
        tag = f"h={hidden_size}  {'full' if rank is None else f'rank={rank}'}"
        for s in range(args.seeds):
            seed = args.base_seed + s
            run_i += 1
            print(f"[{run_i:>3}/{n_runs}]  {tag:<22}  seed={seed}", end="  ", flush=True)
            metrics = run_one(hidden_size, rank, seed, device)
            row = {
                "hidden_size": hidden_size,
                "rank":        rank,
                "rank_key":    -1 if rank is None else rank,  # int sentinel, survives CSV
                "seed":        seed,
                **metrics,
            }
            rows.append(row)
            status = "DIVERGED" if metrics["diverged"] else \
                     f"CCGP={metrics['ccgp_ridge_all']:.3f}  r={metrics['spearman_r']:.3f}"
            print(status)

    df = pd.DataFrame(rows)

    # ── summary table ─────────────────────────────────────────────────────
    all_metrics = [m for m, _ in CCGP_METRICS] + [m for m, _ in PERF_METRICS]
    summary = df.groupby(["hidden_size", "rank_key"])[all_metrics].agg(["mean", "std"]).round(3)
    summary.columns = ["_".join(c) for c in summary.columns]
    summary["n_diverged"] = df.groupby(["hidden_size", "rank_key"])["diverged"].sum().values

    print("\n" + "=" * 80)
    print("SUMMARY  (mean ± std across seeds)")
    print("=" * 80)

    disp_rows = []
    for (h, rk), row in summary.iterrows():
        entry = {"arch": f"h={h}  {'full' if rk == -1 else f'rank={rk}'}",
                 "n_div": int(row["n_diverged"])}
        for mc in all_metrics:
            mu = row[f"{mc}_mean"]; sd = row[f"{mc}_std"]
            entry[mc] = f"{mu:.3f}±{sd:.3f}" if not np.isnan(sd) else f"{mu:.3f}"
        disp_rows.append(entry)

    disp_df = pd.DataFrame(disp_rows).set_index("arch")
    # Print CCGP columns first, then performance
    ccgp_cols = [m for m, _ in CCGP_METRICS]
    perf_cols = [m for m, _ in PERF_METRICS]
    pd.set_option("display.max_colwidth", 14); pd.set_option("display.width", 200)
    print("\n— CCGP metrics —")
    print(disp_df[ccgp_cols + ["n_div"]].to_string())
    print("\n— Performance metrics —")
    print(disp_df[perf_cols].to_string())

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_out = out_dir / f"arch_sweep_{ts}.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nFull results → {csv_out}")

    hidden_sizes = sorted(df["hidden_size"].unique())
    # rank_key is always an int (-1 for full-rank); derive directly from rank_key column
    ranks_all = sorted(int(r) for r in df["rank_key"].unique())
    plot_results(df, hidden_sizes, ranks_all, out_dir, ts, args.seeds)


if __name__ == "__main__":
    main()
