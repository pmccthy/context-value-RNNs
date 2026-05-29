#!/usr/bin/env python
"""
Parameter sweep for the 9-stimulus / 2-context lick task.

Edit SWEEP and N_SEEDS at the top, then run:
    python scripts/param_sweep_13_05_26.py

Results are printed as a summary table, saved to results/sweep_<timestamp>.csv,
a heatmap summary figure, and per-combo lick-value calibration plots are saved
alongside it.
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

# ── repo root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxval.tasks import StimulusSequence, StateSequence
from cxval.envs import TaskEnv
from cxval.models import RNN, ActorCritic
from cxval.agents import Agent

# =============================================================================
# SWEEP CONFIGURATION  ← edit here
# =============================================================================
SWEEP: dict[str, list] = {
    "lick_cost":   [0.0, -0.1, -0.2, -0.5],
    "policy_clip": [0.0, 0.05, 0.1],
}
N_SEEDS = 3      # number of seeds per parameter combo
BASE_SEED = 42   # seeds will be BASE_SEED, BASE_SEED+1, …

# =============================================================================
# FIXED HYPERPARAMETERS  (match notebook)
# =============================================================================
HIDDEN_SIZE    = 64
RECURRENT_GAIN = 0.9
N_EPISODES     = 1
BPTT_LEN       = 10
UPDATE_EVERY   = 10
GAMMA          = 0.9
LR             = 9e-4
VALUE_COEF     = 0.5
ENTROPY_COEF   = 0.01
GRAD_CLIP      = 1.0

TRIALS_PER_PHASE   = 300
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
    """
    Fraction of contexts where mean lick prob is correctly ordered:
        low-value stimuli < mid-value stimuli < high-value stimuli.

    Uses value classes defined by vm: v<=0.25 → low, 0.4<v<0.6 → mid, v>=0.75 → high.
    Returns NaN if any class is empty in a context.
    """
    scores = []
    for ci in range(N_CONTEXTS):
        vals = vm[:, ci]
        lows  = lick_sc[vals <= 0.25, ci]
        mids  = lick_sc[(vals > 0.4) & (vals < 0.6), ci]
        highs = lick_sc[vals >= 0.75, ci]
        if lows.size == 0 or mids.size == 0 or highs.size == 0:
            continue
        l, m, h = np.nanmean(lows), np.nanmean(mids), np.nanmean(highs)
        scores.append(float(l < m < h))
    return float(np.mean(scores)) if scores else np.nan


# =============================================================================
# SINGLE RUN
# =============================================================================

def run_one(
    lick_cost: float,
    policy_clip: float,
    seed: int,
    device: torch.device,
    verbose: bool = False,
) -> dict:
    rng = np.random.default_rng(seed)

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
                               policy_clip=policy_clip).to(device)
    optimizer    = torch.optim.Adam(actor_critic.parameters(), lr=LR)

    # ── training ──────────────────────────────────────────────────────────
    _nan_row = dict(
        spearman_r=np.nan, iti_lick_rate=np.nan, stim_lick_rate=np.nan,
        pct_reward_consumed=np.nan, false_alarm_rate=np.nan, monotonicity=np.nan,
        diverged=True,
        lick_sc=np.full((N_STIMULI, N_CONTEXTS), np.nan),
    )

    def _weights_nan():
        return any(torch.isnan(p).any() for p in actor_critic.parameters())

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
                        obs_next = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
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

                    if _weights_nan():
                        diverged = True
                        break

    except Exception as exc:
        warnings.warn(f"Training crashed (seed={seed}, lick_cost={lick_cost}): {exc}")
        return _nan_row

    if diverged or _weights_nan():
        warnings.warn(f"NaN weights after training (seed={seed}, lick_cost={lick_cost})")
        return _nan_row

    # ── inference (one full block per context, plasticity off) ────────────
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
    action_seq = []
    done = False
    try:
        while not done:
            action, _, _ = agent.act(obs)
            action_seq.append(action)
            obs, _, done, _, _ = infer_env.step(action)
    except Exception as exc:
        warnings.warn(f"Inference crashed (seed={seed}, lick_cost={lick_cost}): {exc}")
        return _nan_row

    # ── metrics ───────────────────────────────────────────────────────────
    lick_sc   = np.full((N_STIMULI, N_CONTEXTS), np.nan)
    iti_rates, stim_rates, rew_r_rates, rew_u_rates = [], [], [], []
    ctx_arr, stim_arr, ravail_arr = [], [], []

    for trial in infer_state_seq.trial_structure:
        si     = trial["stimulus"]
        ci     = trial["context"]
        ravail = trial["reward_available"]
        iti_s,  iti_e  = trial["iti_window"]
        stim_s, stim_e = trial["stim_window"]
        rew_s,  rew_e  = trial["reward_window"]

        def _lr(s, e):
            seg = [a == TaskEnv.LICK for a in action_seq[s:e]]
            return float(np.mean(seg)) if seg else np.nan

        iti_rates.append(_lr(iti_s, iti_e))
        stim_rates.append(_lr(stim_s, stim_e))
        rr = _lr(rew_s, rew_e)
        rew_r_rates.append(rr if ravail  else np.nan)
        rew_u_rates.append(rr if not ravail else np.nan)
        ctx_arr.append(ci); stim_arr.append(si); ravail_arr.append(ravail)

    ctx_arr   = np.array(ctx_arr)
    stim_arr  = np.array(stim_arr)
    ravail_arr = np.array(ravail_arr, dtype=bool)

    lp_flat, rp_flat = [], []
    for si in range(N_STIMULI):
        for ci in range(N_CONTEXTS):
            m = (stim_arr == si) & (ctx_arr == ci)
            if m.sum() == 0:
                continue
            # lick prob = fraction of reward-window timesteps where agent licked
            rew_seg = [action_seq[trial["reward_window"][0]]
                       for trial, ms in zip(infer_state_seq.trial_structure, m) if ms]
            lp = float(np.mean([a == TaskEnv.LICK for a in rew_seg]))
            lick_sc[si, ci] = lp
            lp_flat.append(lp)
            rp_flat.append(float(VALUE_MATRIX[si, ci]))

    spear_r = spearmanr(rp_flat, lp_flat)[0] if len(lp_flat) > 2 else np.nan

    iti_mean   = float(np.nanmean(iti_rates))
    stim_mean  = float(np.nanmean(stim_rates))
    pct_reward = float(np.nanmean(np.array(rew_r_rates, float)[ravail_arr]) * 100) \
                 if ravail_arr.any() else np.nan
    false_alarm = float(np.nanmean(np.array(rew_u_rates, float)[~ravail_arr]) * 100) \
                  if (~ravail_arr).any() else np.nan
    mono       = monotonicity_score(lick_sc, VALUE_MATRIX)

    return dict(
        spearman_r=spear_r,
        iti_lick_rate=iti_mean,
        stim_lick_rate=stim_mean,
        pct_reward_consumed=pct_reward,
        false_alarm_rate=false_alarm,
        monotonicity=mono,
        diverged=False,
        lick_sc=lick_sc,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    device  = torch.device(args.device)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"param_sweep_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output → {out_dir}")

    param_keys   = list(SWEEP.keys())
    param_values = list(SWEEP.values())
    combos       = list(itertools.product(*param_values))
    n_runs       = len(combos) * N_SEEDS

    print(f"Sweep: {param_keys}")
    print(f"Combos: {len(combos)}  ×  Seeds: {N_SEEDS}  =  {n_runs} runs")
    print(f"Device: {device}\n")

    rows = []
    # lick_sc_by_combo[combo_tuple] = list of (N_STIMULI, N_CONTEXTS) arrays
    lick_sc_by_combo: dict[tuple, list] = defaultdict(list)

    run_i = 0
    for combo in combos:
        params = dict(zip(param_keys, combo))
        tag = "  ".join(f"{k}={v}" for k, v in params.items())
        for s in range(N_SEEDS):
            seed = BASE_SEED + s
            run_i += 1
            print(f"[{run_i:>3}/{n_runs}]  {tag}  seed={seed}", end="  ", flush=True)
            metrics = run_one(**params, seed=seed, device=device)
            lick_sc_by_combo[combo].append(metrics.pop("lick_sc"))
            row = {**params, "seed": seed, **metrics}
            rows.append(row)
            status = "DIVERGED" if metrics["diverged"] else f"r={metrics['spearman_r']:.3f}"
            print(status)

    df = pd.DataFrame(rows)

    # ── summary table ─────────────────────────────────────────────────────
    metric_cols = ["spearman_r", "iti_lick_rate", "stim_lick_rate",
                   "pct_reward_consumed", "false_alarm_rate", "monotonicity"]

    summary = (
        df.groupby(param_keys)[metric_cols]
        .agg(["mean", "std"])
        .round(3)
    )

    # Flatten multi-level columns for display: "spearman_r_mean", etc.
    summary.columns = ["_".join(c) for c in summary.columns]

    # Add divergence count
    summary["n_diverged"] = df.groupby(param_keys)["diverged"].sum().values

    print("\n" + "=" * 72)
    print("SUMMARY  (mean ± std across seeds)")
    print("=" * 72)

    # Pretty-print: show mean ± std side by side for each metric
    display_rows = []
    for idx, row in summary.iterrows():
        combo_label = "  ".join(
            f"{k}={v}" for k, v in zip(param_keys, (idx if isinstance(idx, tuple) else (idx,)))
        )
        entry = {"params": combo_label}
        for mc in metric_cols:
            mu  = row[f"{mc}_mean"]
            sd  = row[f"{mc}_std"]
            entry[mc] = f"{mu:.3f} ± {sd:.3f}" if not np.isnan(sd) else f"{mu:.3f}"
        entry["n_diverged"] = int(row["n_diverged"])
        display_rows.append(entry)

    disp_df = pd.DataFrame(display_rows).set_index("params")
    pd.set_option("display.max_colwidth", 20)
    pd.set_option("display.width", 160)
    print(disp_df.to_string())

    # ── save CSV ──────────────────────────────────────────────────────────
    csv_path = out_dir / f"sweep_{ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull results → {csv_path}")

    # ── heatmap summary figure ────────────────────────────────────────────
    _plot_summary(df, param_keys, metric_cols, out_dir, ts)

    # ── per-combo lick visualisation ──────────────────────────────────────
    print("\nSaving per-combo lick figures...")
    for ci, combo in enumerate(combos):
        params = dict(zip(param_keys, combo))
        fig_path = _plot_per_combo(params, lick_sc_by_combo[combo], out_dir, ts, ci)
        print(f"  combo {ci:02d} → {fig_path.name}")


def _plot_per_combo(combo_params, lick_sc_runs, out_dir, ts, combo_idx):
    """
    Two-panel figure for one parameter combo averaged across seeds.

    Left:  lick–value calibration scatter (lick prob vs reward prob).
    Right: per-stimulus per-context bar chart with ±1 SD error bars
           and ground-truth reward probability overlaid as black lines.

    Args:
        combo_params: dict of parameter name → value for this combo.
        lick_sc_runs: list of (N_STIMULI, N_CONTEXTS) arrays, one per seed.
        out_dir: directory to save the figure.
        ts: timestamp string used in the filename.
        combo_idx: integer index of this combo (for filename ordering).
    """
    lick_sc_arr  = np.array(lick_sc_runs, dtype=float)   # (n_seeds, S, C)
    lick_sc_mean = np.nanmean(lick_sc_arr, axis=0)        # (S, C)
    lick_sc_std  = np.nanstd(lick_sc_arr, axis=0)         # (S, C)

    n_stimuli  = N_STIMULI
    n_contexts = N_CONTEXTS
    stimuli    = [f"s{i}" for i in range(n_stimuli)]
    contexts   = [f"c{i}" for i in range(n_contexts)]

    stim_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ctx_markers = ["o", "s", "^", "D", "v"]
    ctx_colors  = ["#5C6BC0", "#FFA726", "#66BB6A", "#EF5350", "#26C6DA"]

    # calibration scatter data
    lp_flat, rp_flat, si_flat, ci_flat = [], [], [], []
    for si in range(n_stimuli):
        for ci in range(n_contexts):
            v = lick_sc_mean[si, ci]
            if not np.isnan(v):
                lp_flat.append(v)
                rp_flat.append(float(VALUE_MATRIX[si, ci]))
                si_flat.append(si)
                ci_flat.append(ci)

    lp_flat = np.array(lp_flat)
    rp_flat = np.array(rp_flat)
    r_sp = spearmanr(rp_flat, lp_flat)[0] if len(lp_flat) > 2 else np.nan

    x = np.arange(n_stimuli)
    w = min(0.8 / n_contexts, 0.4)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── left: calibration scatter ────────────────────────────────────────
    ax = axes[0]
    for lp, rp, si, ci in zip(lp_flat, rp_flat, si_flat, ci_flat):
        ax.scatter(rp, lp,
                   color=stim_colors[si % len(stim_colors)],
                   marker=ctx_markers[ci % len(ctx_markers)],
                   s=90, zorder=3)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Reward probability")
    ax.set_ylabel("Lick probability  (mean across seeds)")
    ax.set_title(f"Lick–value calibration  (Spearman r = {r_sp:.3f})")

    # ── right: per-stim per-context bars ─────────────────────────────────
    ax = axes[1]
    for ci in range(n_contexts):
        offset = (ci - (n_contexts - 1) / 2) * w
        ax.bar(x + offset, lick_sc_mean[:, ci], width=w,
               color=ctx_colors[ci % len(ctx_colors)], alpha=0.85,
               label=contexts[ci], zorder=2)
        ax.errorbar(x + offset, lick_sc_mean[:, ci], yerr=lick_sc_std[:, ci],
                    fmt="none", color="black", capsize=3, lw=1.0, zorder=3)
        for si in range(n_stimuli):
            ax.plot([x[si] + offset - w / 2, x[si] + offset + w / 2],
                    [VALUE_MATRIX[si, ci]] * 2,
                    color="black", lw=2.0, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(stimuli)
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(0, color="gray", lw=0.5, linestyle=":")
    ax.set_xlabel("Stimulus")
    ax.set_ylabel("Lick probability  (mean ± SD across seeds)")
    ax.set_title("Per-stim lick prob  (black lines = GT reward prob)")
    ax.legend(fontsize=8)

    n_seeds = len(lick_sc_runs)
    param_str = "  ".join(f"{k}={v}" for k, v in combo_params.items())
    fig.suptitle(f"Combo {combo_idx:02d}: {param_str}   [{n_seeds} seed{'s' if n_seeds != 1 else ''}]",
                 y=1.02)
    plt.tight_layout()

    safe_tag = "_".join(
        f"{k}{str(v).replace('-', 'm').replace('.', 'p')}"
        for k, v in combo_params.items()
    )
    fig_path = out_dir / f"sweep_{ts}_combo{combo_idx:02d}_{safe_tag}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def _plot_summary(df, param_keys, metric_cols, out_dir, ts):
    """
    If exactly 2 sweep params: 2-D heatmaps (one per metric).
    If 1 param: line/bar plots.
    Otherwise: ranked bar of spearman_r only.
    """
    metric_labels = {
        "spearman_r":          "Spearman r\n(lick↔value, ↑ better)",
        "iti_lick_rate":       "ITI lick rate\n(↓ better)",
        "stim_lick_rate":      "Stim lick rate\n(↓ better)",
        "pct_reward_consumed": "Reward consumed (%)\n(↑ better)",
        "false_alarm_rate":    "False alarm rate (%)\n(↓ better)",
        "monotonicity":        "Monotonicity\nlow<mid<high (↑ better)",
    }
    better_high = {"spearman_r", "pct_reward_consumed", "monotonicity"}

    mean_df = df.groupby(param_keys)[metric_cols].mean().reset_index()

    if len(param_keys) == 2:
        pk0, pk1 = param_keys
        v0 = sorted(mean_df[pk0].unique())
        v1 = sorted(mean_df[pk1].unique())

        n_metrics = len(metric_cols)
        fig, axes = plt.subplots(2, (n_metrics + 1) // 2,
                                 figsize=(4 * ((n_metrics + 1) // 2), 7))
        axes = axes.flat

        for ax, mc in zip(axes, metric_cols):
            grid = np.full((len(v0), len(v1)), np.nan)
            for i, a in enumerate(v0):
                for j, b in enumerate(v1):
                    row = mean_df[(mean_df[pk0] == a) & (mean_df[pk1] == b)]
                    if len(row):
                        grid[i, j] = row[mc].values[0]

            cmap = "RdYlGn" if mc in better_high else "RdYlGn_r"
            im = ax.imshow(grid, cmap=cmap, aspect="auto")
            ax.set_xticks(range(len(v1))); ax.set_xticklabels([f"{b}" for b in v1], fontsize=8)
            ax.set_yticks(range(len(v0))); ax.set_yticklabels([f"{a}" for a in v0], fontsize=8)
            ax.set_xlabel(pk1, fontsize=8); ax.set_ylabel(pk0, fontsize=8)
            ax.set_title(metric_labels.get(mc, mc), fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.8)
            for i in range(len(v0)):
                for j in range(len(v1)):
                    val = grid[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=7, color="black")

        # hide any unused axes
        for ax in list(axes)[n_metrics:]:
            ax.set_visible(False)

        fig.suptitle(
            f"Param sweep summary  (mean over {N_SEEDS} seeds)\n"
            f"{param_keys[0]} × {param_keys[1]}",
            y=1.02, fontsize=10,
        )

    elif len(param_keys) == 1:
        pk = param_keys[0]
        vals = sorted(mean_df[pk].unique())
        n_metrics = len(metric_cols)
        fig, axes = plt.subplots(2, (n_metrics + 1) // 2,
                                 figsize=(4 * ((n_metrics + 1) // 2), 6))
        axes = axes.flat
        for ax, mc in zip(axes, metric_cols):
            y = [mean_df[mean_df[pk] == v][mc].values[0] for v in vals]
            std_df = df.groupby(param_keys)[metric_cols].std().reset_index()
            e = [std_df[std_df[pk] == v][mc].values[0] for v in vals]
            ax.bar([str(v) for v in vals], y, yerr=e, capsize=4, color="steelblue", alpha=0.8)
            ax.set_xlabel(pk, fontsize=8)
            ax.set_title(metric_labels.get(mc, mc), fontsize=8)
        for ax in list(axes)[n_metrics:]:
            ax.set_visible(False)
        fig.suptitle(f"Param sweep — {pk}  (mean ± std, {N_SEEDS} seeds)", fontsize=10)

    else:
        # >2 params: just rank by spearman_r
        mean_df["label"] = mean_df[param_keys].astype(str).agg(" | ".join, axis=1)
        mean_df_s = mean_df.sort_values("spearman_r", ascending=False)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(len(mean_df_s)), mean_df_s["spearman_r"], color="steelblue")
        ax.set_xticks(range(len(mean_df_s)))
        ax.set_xticklabels(mean_df_s["label"], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Mean Spearman r")
        ax.set_title("Param sweep — ranked by lick–value calibration")

    plt.tight_layout()
    fig_path = out_dir / f"sweep_{ts}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figure       → {fig_path}")


if __name__ == "__main__":
    main()
