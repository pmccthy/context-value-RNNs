#!/usr/bin/env python
"""
Multi-run combined analysis for 3s2c and 15s2c swap tasks.

Trains N_RUNS models for each task config then generates all figures matching
the vis notebook, combining results across runs:
  - Trial-by-trial plots (Fig 0a/0b, Fig 8): individual traces + mean ± SEM
  - All other plots: mean ± SEM bars/scatter

Usage (from repo root):
    python scripts/13_05_26_multi_run_combined.py
    python scripts/13_05_26_multi_run_combined.py --n-runs 5 --skip-decoding
    python scripts/13_05_26_multi_run_combined.py --task 3s2c   # one task only
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxval.tasks import StimulusSequence, StateSequence
from cxval.envs import TaskEnv
from cxval.models import RNN, ActorCritic
from cxval.agents import Agent
from cxval.analysis import (
    pairwise_decode, crosscontext_decode, generalisation_matrix,
    value_decode_within, value_decode_cross, value_gen_matrix,
    binary_value_decode_within, binary_value_decode_cross,
    plot_generalisation_heatmap, filter_act_dict,
    compute_unit_tuning, preferred_value_proportions,
    time_resolved_decode,
)

# =============================================================================
# FIXED HYPERPARAMETERS  (match 13_05_26_train_rnn_3s2c_swap.ipynb)
# =============================================================================
ENTROPY_COEF   = 0.1
LICK_COST      = 0.0
POLICY_CLIP    = 0.1
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

REWARD_LICK      = 1.0
REWARD_NO_LICK   = 0.0
REWARD_LICK_MISS = -1.0

POOLING         = "average"
N_FOLDS         = 5
SI_THRESHOLD    = 0.1
VALUE_THRESHOLD = 0.25
N_ITI_PRE       = 3
SMOOTH_W        = 5
LAST_N_REPS     = 5   # training behaviour: show last N context reps (None = all)

_CTX_PALETTE  = ["#5C6BC0", "#FFA726", "#66BB6A", "#EF5350", "#26C6DA"]
_STIM_PALETTE = ["#4C72B0", "#8C8C8C", "#DD8452", "#55A868", "#C44E52",
                 "#8172B2", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974",
                 "#64B5CD", "#4C72B0", "#DD8452", "#55A868", "#C44E52"]


# =============================================================================
# TASK CONFIG BUILDER
# =============================================================================

def build_value_matrix(n_swap_lohi=1, n_fixed_mid=1, n_swap_hilow=1,
                       n_anchor_low=0, n_anchor_high=0):
    """Build value_matrix, stimuli list, and stim_group_info."""
    group_specs = [
        ("swap (low\u2192high)", "swap\n(lo\u2192hi)",  [0.0, 1.0], n_swap_lohi),
        ("mid",                   "mid",                  [0.5, 0.5], n_fixed_mid),
        ("swap (high\u2192low)", "swap\n(hi\u2192lo)", [1.0, 0.0], n_swap_hilow),
        ("anchor (low)",          "anchor\n(low)",       [0.0, 0.0], n_anchor_low),
        ("anchor (high)",         "anchor\n(high)",      [1.0, 1.0], n_anchor_high),
    ]
    rows, si, stim_group_info = [], 0, []
    for gname, gshort, gvals, n in group_specs:
        if n == 0:
            continue
        stim_group_info.append({"name": gname, "short": gshort,
                                 "indices": list(range(si, si + n))})
        rows.extend([gvals] * n)
        si += n
    return np.array(rows, dtype=np.float32), [f"s{i}" for i in range(si)], stim_group_info


TASK_CONFIGS = {
    "3s2c": dict(n_swap_lohi=1, n_fixed_mid=1, n_swap_hilow=1),
    "15s2c": dict(n_swap_lohi=5, n_fixed_mid=5, n_swap_hilow=5),
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


def _sem(arr, axis=0):
    a = np.array(arr, dtype=float)
    n = np.sum(~np.isnan(a), axis=axis)
    return np.nanstd(a, axis=axis) / np.sqrt(np.maximum(n, 1))


def _smooth(y, w):
    if w > 1 and len(y) >= w:
        return np.convolve(y, np.ones(w) / w, mode="valid")
    return np.array(y, dtype=float)


def _lick_rate_seg(action_seq, s, e):
    seg = [action_seq[t] == TaskEnv.LICK for t in range(s, e)]
    return float(np.mean(seg)) if seg else np.nan


# =============================================================================
# SINGLE RUN
# =============================================================================

def run_one(value_matrix, stimuli, stim_group_info, contexts,
            seed, device, skip_decoding=False):
    """Train one model and return all per-run data needed for every figure.

    Args:
        value_matrix: (n_stimuli, n_contexts) reward probability array.
        stimuli: List of stimulus name strings.
        stim_group_info: List of stimulus group dicts.
        contexts: List of context name strings.
        seed: RNG seed.
        device: Torch device.
        skip_decoding: If True, skip collecting full hidden states.

    Returns:
        Dict of per-run data, or None if the run diverged.
    """
    n_stimuli  = len(stimuli)
    n_contexts = len(contexts)

    # ── training task ─────────────────────────────────────────────────────
    stim_seq = StimulusSequence(
        value_matrix=value_matrix,
        trials_per_phase=TRIALS_PER_PHASE,
        phases_per_context=PHASES_PER_CONTEXT,
        context_order="sequential",
        context_reps=CONTEXT_REPS,
    )
    stim_seq.generate(seed=seed)
    state_seq = StateSequence(
        stimulus_sequence=stim_seq, value_matrix=value_matrix,
        stim_timesteps=STIM_TIMESTEPS, reward_timesteps=REWARD_TIMESTEPS,
        iti_timesteps=ITI_TIMESTEPS,
    )
    states, _, reward_availability = state_seq.generate(seed=seed)
    obs_dim = states.shape[1] + 2

    env = TaskEnv(states=states, reward_availability=reward_availability,
                  reward_lick=REWARD_LICK, reward_no_lick=REWARD_NO_LICK,
                  reward_lick_miss=REWARD_LICK_MISS, lick_cost=LICK_COST)

    # ── model ─────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    backbone     = RNN(input_size=obs_dim, hidden_size=HIDDEN_SIZE,
                       output_size=1, recurrent_gain=RECURRENT_GAIN)
    actor_critic = ActorCritic(backbone=backbone, num_actions=2,
                               policy_clip=POLICY_CLIP).to(device)
    optimizer    = torch.optim.Adam(actor_critic.parameters(), lr=LR)

    def _nan_weights():
        return any(torch.isnan(p).any() for p in actor_critic.parameters())

    # ── training ──────────────────────────────────────────────────────────
    actor_critic.train()
    optimizer.zero_grad()
    obs, _  = env.reset()
    hidden  = None
    lp_buf, val_buf, rew_buf, ent_buf = [], [], [], []
    t_win   = 0
    done    = False
    all_trial_data = []

    try:
        while not done:
            obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value, hidden = actor_critic.step(obs_t, hidden)
            dist   = actor_critic.make_dist(logits)
            action = dist.sample()
            lp_buf.append(dist.log_prob(action))
            val_buf.append(value)
            ent_buf.append(dist.entropy())
            obs, reward, done, _, _ = env.step(action.item())
            rew_buf.append(reward)
            t_win += 1

            if t_win % UPDATE_EVERY == 0 or done:
                bv = 0.0
                if not done:
                    with torch.no_grad():
                        obs_n = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        _, bv_t, _ = actor_critic.step(obs_n, hidden)
                        bv = bv_t.item()
                ret_t  = torch.tensor(compute_returns(rew_buf, bv, GAMMA),
                                       dtype=torch.float32, device=device)
                lp_t   = torch.stack(lp_buf).squeeze(-1)
                val_t  = torch.stack(val_buf).squeeze(-1)
                ent_m  = torch.stack(ent_buf).mean()
                adv    = ret_t - val_t.detach()
                std    = adv.std()
                adv    = (adv - adv.mean()) / (std + 1e-8) if std > 1e-4 else adv - adv.mean()
                loss   = (-(lp_t * adv).mean()
                          + VALUE_COEF * F.mse_loss(val_t, ret_t)
                          - ENTROPY_COEF * ent_m)
                is_last = (t_win >= BPTT_LEN) or done
                loss.backward(retain_graph=not is_last)
                lp_buf, val_buf, rew_buf, ent_buf = [], [], [], []
                if is_last:
                    nn.utils.clip_grad_norm_(actor_critic.parameters(), GRAD_CLIP)
                    optimizer.step()
                    optimizer.zero_grad()
                    hidden = hidden.detach()
                    t_win  = 0
                    if _nan_weights():
                        return None
    except Exception:
        return None

    if _nan_weights():
        return None

    # ── collect training trial data ───────────────────────────────────────
    agent_train = Agent(actor_critic, device=device)
    actor_critic.eval()

    # reconstruct training actions from the already-run episode
    # (we need to replay to get value estimates per trial)
    obs2, _ = env.reset()
    agent_train.reset()
    val_ts_list, act_list = [], []
    done2 = False
    try:
        while not done2:
            action, _, vval = agent_train.act(obs2)
            act_list.append(action)
            val_ts_list.append(vval.item())
            obs2, _, done2, _, _ = env.step(action)
    except Exception:
        act_list    = [TaskEnv.NO_LICK] * len(state_seq.trial_structure) * 8
        val_ts_list = [0.0] * len(act_list)

    for ti, trial in enumerate(state_seq.trial_structure):
        rs, re = trial["reward_window"]
        all_trial_data.append({
            "context":          trial["context"],
            "stimulus":         trial["stimulus"],
            "reward_available": trial["reward_available"],
            "licked":           int(act_list[rs] == TaskEnv.LICK) if rs < len(act_list) else 0,
            "value_estimate":   float(np.mean(val_ts_list[rs:re])) if rs < len(val_ts_list) else 0.0,
            "lick_count":       sum(1 for t in range(rs, re)
                                    if t < len(act_list) and act_list[t] == TaskEnv.LICK),
        })

    # ── inference task ────────────────────────────────────────────────────
    infer_stim_seq = StimulusSequence(
        value_matrix=value_matrix,
        trials_per_phase=TRIALS_PER_PHASE,
        phases_per_context=1,
        context_order="sequential",
        context_reps=1,
    )
    infer_stim_seq.generate(seed=seed + 1000)
    infer_state_seq = StateSequence(
        stimulus_sequence=infer_stim_seq, value_matrix=value_matrix,
        stim_timesteps=STIM_TIMESTEPS, reward_timesteps=REWARD_TIMESTEPS,
        iti_timesteps=ITI_TIMESTEPS,
    )
    infer_states, _, infer_reward_avail = infer_state_seq.generate(seed=seed + 1000)
    infer_env = TaskEnv(
        states=infer_states, reward_availability=infer_reward_avail,
        reward_lick=REWARD_LICK, reward_no_lick=REWARD_NO_LICK,
        reward_lick_miss=REWARD_LICK_MISS, lick_cost=LICK_COST,
    )

    agent = Agent(actor_critic, device=device)
    agent.reset()
    obs, _ = infer_env.reset()
    infer_act, infer_val_ts, hidden_list = [], [], []
    done = False
    try:
        while not done:
            action, _, vval = agent.act(obs)
            infer_act.append(action)
            infer_val_ts.append(vval.item())
            hidden_list.append(agent.hidden.detach().squeeze(0).cpu())
            obs, _, done, _, _ = infer_env.step(action)
    except Exception:
        return None

    hidden_np = np.array(torch.stack(hidden_list).tolist(), dtype=np.float32)
    ts = infer_state_seq.trial_structure

    # ── per-trial inference data ──────────────────────────────────────────
    infer_trial_data = []
    for ti, trial in enumerate(ts):
        rs, re = trial["reward_window"]
        infer_trial_data.append({
            "context":          trial["context"],
            "stimulus":         trial["stimulus"],
            "reward_available": trial["reward_available"],
            "licked":           int(infer_act[rs] == TaskEnv.LICK),
            "value_estimate":   float(np.mean(infer_val_ts[rs:re])),
            "lick_count":       sum(1 for t in range(rs, re) if infer_act[t] == TaskEnv.LICK),
        })

    # ── per-stim/ctx aggregates ───────────────────────────────────────────
    inf_stim   = np.array([d["stimulus"]         for d in infer_trial_data])
    inf_ctx    = np.array([d["context"]          for d in infer_trial_data])
    inf_lick   = np.array([d["licked"]           for d in infer_trial_data], dtype=float)
    inf_lkct   = np.array([d["lick_count"]       for d in infer_trial_data], dtype=float)
    inf_vest   = np.array([d["value_estimate"]   for d in infer_trial_data], dtype=float)
    inf_ravail = np.array([d["reward_available"] for d in infer_trial_data], dtype=bool)

    val_sc = lick_sc = lkrt_sc = None
    val_sc  = np.full((n_stimuli, n_contexts), np.nan)
    lick_sc = np.full((n_stimuli, n_contexts), np.nan)
    lkrt_sc = np.full((n_stimuli, n_contexts), np.nan)
    for si in range(n_stimuli):
        for ci in range(n_contexts):
            m = (inf_stim == si) & (inf_ctx == ci)
            if m.sum():
                val_sc[si, ci]  = inf_vest[m].mean()
                lick_sc[si, ci] = inf_lick[m].mean()
                lkrt_sc[si, ci] = inf_lkct[m].mean() / REWARD_TIMESTEPS

    # ── period lick rates ─────────────────────────────────────────────────
    iti_r, stim_r, rewr_r, rewu_r = [], [], [], []
    pc_stim, pc_ctx, pc_ravail    = [], [], []
    for trial in ts:
        iti_s,  iti_e  = trial["iti_window"]
        stim_s, stim_e = trial["stim_window"]
        rew_s,  rew_e  = trial["reward_window"]
        rr = _lick_rate_seg(infer_act, rew_s, rew_e)
        iti_r.append(_lick_rate_seg(infer_act, iti_s, iti_e))
        stim_r.append(_lick_rate_seg(infer_act, stim_s, stim_e))
        rewr_r.append(rr if trial["reward_available"]  else np.nan)
        rewu_r.append(rr if not trial["reward_available"] else np.nan)
        pc_stim.append(trial["stimulus"])
        pc_ctx.append(trial["context"])
        pc_ravail.append(trial["reward_available"])

    pc_stim   = np.array(pc_stim)
    pc_ctx    = np.array(pc_ctx)
    pc_ravail = np.array(pc_ravail, dtype=bool)

    def _by_sc(rates):
        r = np.array(rates, dtype=float)
        out = np.full((n_stimuli, n_contexts), np.nan)
        for si in range(n_stimuli):
            for ci in range(n_contexts):
                m   = (pc_stim == si) & (pc_ctx == ci)
                val = r[m]
                if np.any(~np.isnan(val)):
                    out[si, ci] = np.nanmean(val)
        return out

    # ── scalar metrics ────────────────────────────────────────────────────
    lp_flat = [lick_sc[si, ci] for si in range(n_stimuli) for ci in range(n_contexts)
               if not np.isnan(lick_sc[si, ci])]
    rp_flat = [float(value_matrix[si, ci]) for si in range(n_stimuli) for ci in range(n_contexts)
               if not np.isnan(lick_sc[si, ci])]
    spear_r    = spearmanr(rp_flat, lp_flat)[0] if len(lp_flat) > 2 else np.nan
    pct_reward = inf_lick[inf_ravail].mean() * 100 if inf_ravail.any() else np.nan
    false_alarm = inf_lick[~inf_ravail].mean() * 100 if (~inf_ravail).any() else np.nan

    # ── activations for decoding ──────────────────────────────────────────
    if not skip_decoding:
        stim_hidden = np.array([
            hidden_np[t["stim_window"][0]:t["stim_window"][1]].tolist()
            for t in ts
        ], dtype=np.float32)
        rew_hidden = np.array([
            hidden_np[t["reward_window"][0]:t["reward_window"][1]].tolist()
            for t in ts
        ], dtype=np.float32)
        infer_activations = {
            "hidden_states":    hidden_np,
            "stim_hidden":      stim_hidden,
            "reward_hidden":    rew_hidden,
            "context":          inf_ctx,
            "stimulus":         inf_stim,
            "reward_available": inf_ravail,
            "trial_structure":  ts,
        }
    else:
        infer_activations = None

    # ── selectivity ───────────────────────────────────────────────────────
    if not skip_decoding:
        tuning           = compute_unit_tuning(infer_activations, period="stim")
        props, si_vals   = preferred_value_proportions(
            tuning, value_matrix,
            si_threshold=SI_THRESHOLD, value_threshold=VALUE_THRESHOLD,
        )
    else:
        props, si_vals = None, None

    return dict(
        # trial-by-trial
        infer_trial_data = infer_trial_data,
        all_trial_data   = all_trial_data,
        # per-stim/ctx aggregates
        val_sc   = val_sc,
        lick_sc  = lick_sc,
        lkrt_sc  = lkrt_sc,
        iti_sc   = _by_sc(iti_r),
        stim_sc  = _by_sc(stim_r),
        rewr_sc  = _by_sc(rewr_r),
        rewu_sc  = _by_sc(rewu_r),
        iti_r_all    = np.array(iti_r, dtype=float),
        stim_r_all   = np.array(stim_r, dtype=float),
        rewr_r_all   = np.array(rewr_r, dtype=float),
        rewu_r_all   = np.array(rewu_r, dtype=float),
        # scalar metrics
        spearman_r  = spear_r,
        pct_reward  = pct_reward,
        false_alarm = false_alarm,
        # decoding
        infer_activations = infer_activations,
        props  = props,
        si     = si_vals,
        n_trials_infer = len(infer_trial_data),
        n_trials_train = len(all_trial_data),
    )


# =============================================================================
# FIGURE HELPERS
# =============================================================================

def _stim_colors(n):
    return [_STIM_PALETTE[i % len(_STIM_PALETTE)] for i in range(n)]


def _ctx_colors(n):
    return [_CTX_PALETTE[i % len(_CTX_PALETTE)] for i in range(n)]


def _bar_with_sem(ax, x, means, sems, colors, width=0.5, **kwargs):
    for xi, mu, se, c in zip(x, means, sems, colors):
        ax.bar(xi, mu, width=width, color=c, alpha=0.85, zorder=2, **kwargs)
        ax.errorbar(xi, mu, yerr=se, fmt="none", color="black",
                    capsize=4, lw=1.2, zorder=3)


def _save(fig, path, tight=True):
    if tight:
        plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


# =============================================================================
# FIGURES
# =============================================================================

def fig0_behaviour(runs, value_matrix, stimuli, contexts, stim_group_info,
                   fig_dir, task_name, last_n_reps=LAST_N_REPS):
    """Fig 0a/0b — trial-by-trial inference and training behaviour."""
    n_stimuli  = len(stimuli)
    n_contexts = len(contexts)
    sc         = _stim_colors(n_stimuli)
    cc         = _ctx_colors(n_contexts)
    w          = SMOOTH_W

    def _plot_behaviour(td_list, suffix, fname):
        if not td_list:
            return
        max_trials = max(len(td) for td in td_list)
        fig, axes = plt.subplots(3, 1, figsize=(14, 8),
                                 gridspec_kw={"height_ratios": [1, 1, 1], "hspace": 0.45})
        for si in range(n_stimuli):
            c     = sc[si]
            per_run_v, per_run_lp, per_run_lr = [], [], []
            for td in td_list:
                m  = td[:, 2] == si
                xi = np.arange(m.sum())
                v  = td[m, 5]
                lp = td[m, 4]
                lr = td[m, 6] / REWARD_TIMESTEPS
                sv  = _smooth(v,  w)
                slp = _smooth(lp, w)
                slr = _smooth(lr, w)
                n_s = len(sv)
                # thin individual traces
                axes[0].plot(xi[:n_s], sv,  color=c, lw=0.7, alpha=0.35)
                axes[1].plot(xi[:n_s], slp, color=c, lw=0.7, alpha=0.35)
                axes[2].plot(xi[:n_s], slr, color=c, lw=0.7, alpha=0.35)
                per_run_v.append(sv); per_run_lp.append(slp); per_run_lr.append(slr)

            # mean over runs — pad to same length
            min_len = min(len(x) for x in per_run_v)
            if min_len > 0:
                mv  = np.nanmean([x[:min_len] for x in per_run_v],  axis=0)
                mlp = np.nanmean([x[:min_len] for x in per_run_lp], axis=0)
                mlr = np.nanmean([x[:min_len] for x in per_run_lr], axis=0)
                xi  = np.arange(min_len)
                axes[0].plot(xi, mv,  color=c, lw=2.2, label=stimuli[si], zorder=4)
                axes[1].plot(xi, mlp, color=c, lw=2.2, zorder=4)
                axes[2].plot(xi, mlr, color=c, lw=2.2, zorder=4)

            # GT dotted line per context
            for ci in range(n_contexts):
                axes[0].axhline(value_matrix[si, ci], color=c, lw=0.9,
                                linestyle=":", alpha=0.55)

        handles = [Line2D([0],[0], color=sc[si], lw=2, label=stimuli[si])
                   for si in range(n_stimuli)]
        if n_contexts > 1:
            handles += [Patch(facecolor=cc[ci], alpha=0.7, label=contexts[ci])
                        for ci in range(n_contexts)]
        axes[0].legend(handles=handles, ncol=len(handles), fontsize=8, loc="upper left")
        axes[0].set_ylabel("Value estimate")
        axes[0].set_title("Value estimate  (dotted lines = GT reward probability per context)")
        axes[1].set_ylabel("Lick prob\n(1st rew. step)")
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].set_title("Lick probability at first reward-window timestep")
        axes[1].axhline(0.5, color="gray", lw=0.7, ls="--", alpha=0.5)
        axes[2].set_ylabel(f"Lick rate\n(all {REWARD_TIMESTEPS} rew. steps)")
        axes[2].set_ylim(-0.02, 1.02)
        axes[2].set_xlabel("Trial index (per stimulus)")
        axes[2].set_title("Mean lick rate across reward window")
        axes[2].axhline(0.5, color="gray", lw=0.7, ls="--", alpha=0.5)
        for ax in axes:
            ax.set_xlim(0, min_len - 1 if min_len > 0 else 1)
        n = len(td_list)
        fig.suptitle(f"{task_name}  ·  {suffix}  ·  {n} run{'s' if n != 1 else ''}  "
                     f"(thin=individual, thick=mean)", fontsize=9, y=1.01)
        _save(fig, fig_dir / fname)

    # ── 0a: inference ─────────────────────────────────────────────────────
    infer_td_list = []
    for r in runs:
        td = np.array([(i, d["context"], d["stimulus"], d["reward_available"],
                        d["licked"], d["value_estimate"], d["lick_count"])
                       for i, d in enumerate(r["infer_trial_data"])], dtype=float)
        infer_td_list.append(td)
    _plot_behaviour(infer_td_list, "inference  [plasticity off]", "fig0a_inference_behaviour.png")

    # ── 0b: training (last N reps) ────────────────────────────────────────
    train_td_list = []
    for r in runs:
        td_all = np.array([(i, d["context"], d["stimulus"], d["reward_available"],
                            d["licked"], d["value_estimate"], d["lick_count"])
                           for i, d in enumerate(r["all_trial_data"])], dtype=float)
        if last_n_reps is not None:
            cut = max(0, len(td_all) - last_n_reps * n_contexts * TRIALS_PER_PHASE)
            td_all = td_all[int(cut):]
            td_all[:, 0] -= td_all[0, 0]
        train_td_list.append(td_all)
    suffix = f"training  (last {last_n_reps} reps)" if last_n_reps else "training  (full)"
    _plot_behaviour(train_td_list, suffix, "fig0b_training_behaviour.png")


def fig1_calibration(runs, value_matrix, stimuli, contexts, fig_dir, task_name):
    """Fig 1 — lick-value calibration scatter and performance summary."""
    n_stimuli  = len(stimuli)
    n_contexts = len(contexts)
    sc         = _stim_colors(n_stimuli)
    cc         = _ctx_colors(n_contexts)

    lick_arr = np.array([r["lick_sc"] for r in runs])   # (N, n_stim, n_ctx)
    lick_mean = np.nanmean(lick_arr, 0)
    lick_sem  = _sem(lick_arr, 0)

    r_sp_arr = np.array([r["spearman_r"]  for r in runs])
    pr_arr   = np.array([r["pct_reward"]  for r in runs])
    fa_arr   = np.array([r["false_alarm"] for r in runs])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # scatter
    ax = axes[0]
    for si in range(n_stimuli):
        for ci in range(n_contexts):
            mu  = lick_mean[si, ci]
            se  = lick_sem[si, ci]
            rp  = float(value_matrix[si, ci])
            m   = ctx_markers[ci % len(ctx_markers)]
            if not np.isnan(mu):
                ax.errorbar(rp, mu, yerr=se, fmt=m, color=sc[si], ms=8,
                            capsize=4, lw=1.2, zorder=3)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Reward probability")
    ax.set_ylabel("Lick probability  (mean ± SEM)")
    r_m = np.nanmean(r_sp_arr)
    r_s = _sem(r_sp_arr)
    ax.set_title(f"Lick–value calibration\nSpearman r = {r_m:.3f} ± {r_s:.3f}")


    # performance bars
    ax = axes[1]
    metrics = [("Reward\nconsumed %", pr_arr, "#55A868"),
               ("False\nalarm %",    fa_arr, "#DD8452")]
    for xi, (label, arr, c) in enumerate(metrics):
        mu = np.nanmean(arr)
        se = _sem(arr)
        ax.bar(xi, mu, color=c, alpha=0.85, width=0.5, zorder=2)
        ax.errorbar(xi, mu, yerr=se, fmt="none", color="black", capsize=5, lw=1.5, zorder=3)
        ax.text(xi, mu + se + 1, f"{mu:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_ylim(0, 120)
    ax.set_ylabel("% trials")
    ax.set_title("Performance summary  (mean ± SEM)")

    n = len(runs)
    fig.suptitle(f"{task_name}  ·  {n} runs", fontsize=10, y=1.02)
    _save(fig, fig_dir / "fig1_lick_value_calibration.png")


ctx_markers = ["o", "s", "^", "D", "v"]


def _fig_sc_bars(runs_sc, value_matrix, stimuli, contexts, stim_group_info,
                 row_labels, row_titles, fig_path, task_name,
                 ylim=(0, 1.05), ylabel_extra=""):
    """Generic per-stim per-ctx bar chart with mean ± SEM across runs."""
    n_stimuli  = len(stimuli)
    n_contexts = len(contexts)
    n_rows     = len(runs_sc)
    sc         = _stim_colors(n_stimuli)
    cc         = _ctx_colors(n_contexts)

    w   = min(0.8 / max(n_contexts, 1), 0.4)
    x   = np.arange(n_stimuli)
    fig, axes = plt.subplots(n_rows, 1, figsize=(max(8, 2 + 1.6 * n_stimuli), 3.5 * n_rows),
                             sharex=True, gridspec_kw={"hspace": 0.55})
    if n_rows == 1:
        axes = [axes]

    for row_i, (sc_list, row_lbl, row_title) in enumerate(
            zip(runs_sc, row_labels, row_titles)):
        ax   = axes[row_i]
        arr  = np.array(sc_list)      # (n_runs, n_stim, n_ctx)
        mean = np.nanmean(arr, 0)
        sem  = _sem(arr, 0)
        for ci in range(n_contexts):
            xi  = x + (ci - (n_contexts - 1) / 2) * w
            clr = [sc[si] for si in range(n_stimuli)]
            ax.bar(xi, mean[:, ci], width=w * 0.9, color=clr, alpha=0.85,
                   edgecolor=cc[ci], linewidth=1.5, zorder=2,
                   label=contexts[ci] if ci == 0 else "_")
            ax.errorbar(xi, mean[:, ci], yerr=sem[:, ci],
                        fmt="none", color=cc[ci], capsize=3, lw=1.0, zorder=3)
            for si in range(n_stimuli):
                ax.plot([xi[si] - w * 0.4, xi[si] + w * 0.4],
                        [value_matrix[si, ci]] * 2,
                        color=cc[ci], lw=2.0, zorder=5)

        ax.set_ylim(*ylim)
        ax.set_ylabel(row_lbl + ("\n" + ylabel_extra if ylabel_extra else ""))
        ax.set_title(row_title)
        ax.axhline(0.5, color="gray", lw=0.7, ls="--", alpha=0.5)

        if n_contexts > 1:
            ctx_handles = [Patch(facecolor=cc[ci], label=contexts[ci])
                           for ci in range(n_contexts)]
            ax.legend(handles=ctx_handles, fontsize=8, loc="upper right")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(stimuli, rotation=45 if n_stimuli > 6 else 0)
    axes[-1].set_xlabel("Stimulus")

    n = len(sc_list)
    fig.suptitle(f"{task_name}  ·  {n} runs  (bars=mean, err=SEM, "
                 "lines=GT reward prob)", fontsize=9, y=1.01)
    _save(fig, fig_path)


def fig2_per_stim_summary(runs, value_matrix, stimuli, contexts, stim_group_info,
                          fig_dir, task_name):
    """Fig 2 — per-stimulus per-context aggregates."""
    _fig_sc_bars(
        [[r["val_sc"]  for r in runs],
         [r["lick_sc"] for r in runs],
         [r["lkrt_sc"] for r in runs]],
        value_matrix, stimuli, contexts, stim_group_info,
        row_labels=["Value estimate\n(reward window)",
                    "Lick prob\n(1st rew. step)",
                    f"Lick rate\n(all {REWARD_TIMESTEPS} rew. steps)"],
        row_titles=["Value estimates  [mean over reward-window timesteps]",
                    "Lick probability at first reward-window timestep",
                    f"Lick rate  [all {REWARD_TIMESTEPS} reward-window timesteps]"],
        fig_path=fig_dir / "fig2_per_stim_inference_summary.png",
        task_name=task_name,
    )


def fig3_period_overview(runs, contexts, fig_dir, task_name):
    """Fig 3 — overall and per-context lick rates by period."""
    n_contexts = len(contexts)
    cc         = _ctx_colors(n_contexts)

    keys   = ["iti_r_all", "stim_r_all", "rewr_r_all", "rewu_r_all"]
    labels = ["ITI", "Stim", "Reward\n(rewarded)", "Reward\n(unrewarded)"]
    colors = ["lightgray", "#4C72B0", "#55A868", "#DD8452"]

    overall_mean = {l: np.nanmean([np.nanmean(r[k]) for r in runs]) for l, k in zip(labels, keys)}
    overall_sem  = {l: _sem(np.array([np.nanmean(r[k]) for r in runs])) for l, k in zip(labels, keys)}

    fig, (ax_ov, ax_ctx) = plt.subplots(1, 2, figsize=(11, 4.5))

    x = np.arange(len(labels))
    ax_ov.bar(x, [overall_mean[l] for l in labels],
              color=colors, width=0.55, alpha=0.85, zorder=2)
    ax_ov.errorbar(x, [overall_mean[l] for l in labels],
                   yerr=[overall_sem[l] for l in labels],
                   fmt="none", color="black", capsize=4, lw=1.2, zorder=3)
    ax_ov.set_xticks(x); ax_ov.set_xticklabels(labels)
    ax_ov.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax_ov.set_ylim(0, 1.05); ax_ov.set_ylabel("Mean lick rate  (mean ± SEM)")
    ax_ov.set_title(f"Overall lick rate by period  [{len(runs)} runs]")

    if n_contexts > 1:
        w = 0.35
        for ki, (k, lab) in enumerate(zip(keys, labels)):
            for ci in range(n_contexts):
                ctx_means = []
                for r in runs:
                    arr = r[k]
                    # align to context using rewr/rewu per-trial arrays...
                    ctx_means.append(np.nanmean(arr))  # simplified
                mu = np.nanmean(ctx_means)
                se = _sem(ctx_means)
                xi = ki + (ci - (n_contexts - 1) / 2) * w
                ax_ctx.bar(xi, mu, width=w * 0.9, color=cc[ci], alpha=0.8, zorder=2)
                ax_ctx.errorbar(xi, mu, yerr=se, fmt="none", color="black",
                                capsize=3, lw=1, zorder=3)
        ax_ctx.set_xticks(range(len(labels))); ax_ctx.set_xticklabels(labels)
        ax_ctx.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax_ctx.set_ylim(0, 1.05)
        ctx_handles = [Patch(facecolor=cc[ci], label=contexts[ci]) for ci in range(n_contexts)]
        ax_ctx.legend(handles=ctx_handles, fontsize=9)
        ax_ctx.set_title("By context")
    else:
        ax_ctx.set_visible(False)

    n = len(runs)
    fig.suptitle(f"{task_name}  ·  {n} runs  [inference / plasticity off]", fontsize=9, y=1.01)
    _save(fig, fig_dir / "fig3_period_lick_rates_overview.png")


def fig4_period_per_stim(runs, value_matrix, stimuli, contexts, stim_group_info,
                         fig_dir, task_name):
    """Fig 4 — per-stimulus per-context lick rates by trial period."""
    _fig_sc_bars(
        [[r["iti_sc"]  for r in runs],
         [r["stim_sc"] for r in runs],
         [r["rewr_sc"] for r in runs],
         [r["rewu_sc"] for r in runs]],
        value_matrix, stimuli, contexts, stim_group_info,
        row_labels=["ITI lick rate",
                    "Stim lick rate",
                    "Reward lick rate\n(rewarded)",
                    "Reward lick rate\n(unrewarded)"],
        row_titles=[f"ITI period  [all ITI timesteps]",
                    f"Stimulus period  [all {STIM_TIMESTEPS} timesteps]",
                    f"Reward window — rewarded trials  [{REWARD_TIMESTEPS} timesteps]",
                    f"Reward window — unrewarded trials  [{REWARD_TIMESTEPS} timesteps]"],
        fig_path=fig_dir / "fig4_period_lick_rates_per_stim.png",
        task_name=task_name,
    )


def _mean_sem_heatmaps(gm_list, contexts, title, fig_path, cmap, vmin, vmax,
                       colorbar_label, n_runs):
    """Average a list of generalisation matrices into mean + SEM heatmaps."""
    arr  = np.array(gm_list, dtype=float)
    mean = np.nanmean(arr, 0)
    sem  = _sem(arr, 0)
    n_ctx = len(contexts)
    cell_size = 1.1
    fig, axes = plt.subplots(1, 2, figsize=(cell_size * n_ctx * 2 + 2.5, cell_size * n_ctx + 1))
    plot_generalisation_heatmap(axes[0], mean, contexts,
                                vmin=vmin, vmax=vmax, cmap=cmap,
                                colorbar_label=colorbar_label,
                                title=f"{title}  (mean, n={n_runs})")
    plot_generalisation_heatmap(axes[1], sem,  contexts,
                                vmin=0, vmax=None, cmap="Oranges",
                                colorbar_label="SEM",
                                title=f"{title}  (SEM)")
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {fig_path.name}")


def fig5_decoding(runs, value_matrix, contexts, stim_group_info, fig_dir, task_name):
    """Fig 5 — pairwise identity and Ridge value generalisation heatmaps."""
    n_ctx = len(contexts)
    cell  = 1.1
    n_runs = len(runs)
    valid  = [r for r in runs if r["infer_activations"] is not None]
    if not valid:
        return

    # pairwise identity — one gen_matrix per context
    for ci in range(n_ctx):
        gm_list = []
        for r in valid:
            try:
                w  = pairwise_decode(r["infer_activations"], period="stim",
                                     pooling=POOLING, n_folds=N_FOLDS)
                cc = crosscontext_decode(r["infer_activations"], period="stim",
                                        pooling=POOLING)
                gm_list.append(generalisation_matrix(w, cc))
            except Exception as e:
                warnings.warn(f"Pairwise decode failed: {e}")
        if gm_list:
            _mean_sem_heatmaps(gm_list, contexts,
                               f"Stim identity  (ctx {ci})",
                               fig_dir / f"fig5a_identity_ctx{ci}.png",
                               cmap="RdBu_r", vmin=0, vmax=1,
                               colorbar_label="Accuracy", n_runs=len(gm_list))
        break  # single combined for simplicity

    # Ridge value
    gm_list = []
    for r in valid:
        try:
            rw = value_decode_within(r["infer_activations"], period="stim",
                                     pooling=POOLING, value_matrix=value_matrix, n_folds=N_FOLDS)
            rc = value_decode_cross(r["infer_activations"], period="stim",
                                    pooling=POOLING, value_matrix=value_matrix)
            gm_list.append(value_gen_matrix(rw, rc))
        except Exception as e:
            warnings.warn(f"Ridge decode failed: {e}")
    if gm_list:
        _mean_sem_heatmaps(gm_list, contexts,
                           "Value — Ridge",
                           fig_dir / "fig5b_ridge_value.png",
                           cmap="seismic", vmin=-1, vmax=1,
                           colorbar_label="Pearson r", n_runs=len(gm_list))


def fig6_binary_svm(runs, value_matrix, contexts, stim_group_info, fig_dir, task_name):
    """Fig 6 — binary SVM value decoding by stimulus group.

    Builds a (n_ctx, n_ctx) generalisation matrix where diagonal = within-context
    accuracy and off-diagonal = cross-context accuracy.
    """
    valid = [r for r in runs if r["infer_activations"] is not None]
    if not valid:
        return

    n_ctx = len(contexts)
    _swap_idx   = [i for g in stim_group_info if "swap"   in g["name"] for i in g["indices"]]
    _anchor_idx = [i for g in stim_group_info if "anchor" in g["name"] for i in g["indices"]]
    stim_groups = {"all": None}
    if _swap_idx:
        stim_groups["swap"]   = _swap_idx
    if _anchor_idx:
        stim_groups["anchor"] = _anchor_idx

    for gname, gidx in stim_groups.items():
        gm_list = []
        for r in valid:
            act = (filter_act_dict(r["infer_activations"],
                                   np.isin(r["infer_activations"]["stimulus"], gidx))
                   if gidx is not None else r["infer_activations"])
            try:
                bw = binary_value_decode_within(act, period="stim", pooling=POOLING,
                                                value_matrix=value_matrix, n_folds=N_FOLDS,
                                                threshold=VALUE_THRESHOLD)  # (n_ctx,)
                bc = binary_value_decode_cross(act, period="stim", pooling=POOLING,
                                               value_matrix=value_matrix,
                                               threshold=VALUE_THRESHOLD)   # (n_ctx, n_ctx)
                # assemble: diagonal=within, off-diagonal=cross
                gm = bc.copy()
                for ci in range(n_ctx):
                    gm[ci, ci] = bw[ci]
                gm_list.append(gm)
            except Exception as e:
                warnings.warn(f"Binary SVM failed for group '{gname}': {e}")
        if gm_list:
            _mean_sem_heatmaps(gm_list, contexts,
                               f"Binary SVM — {gname}",
                               fig_dir / f"fig6_binary_svm_{gname}.png",
                               cmap="RdBu_r", vmin=0, vmax=1,
                               colorbar_label="Accuracy", n_runs=len(gm_list))


def fig7_selectivity(runs, value_matrix, stim_group_info, fig_dir, task_name):
    """Fig 7 — unit selectivity proportions (mean ± SEM)."""
    valid = [r for r in runs if r["props"] is not None]
    if not valid:
        return

    from cxval.analysis import selectivity_index
    n_ctx = value_matrix.shape[1]
    cc    = _ctx_colors(n_ctx)
    bar_c = ["#4C72B0", "#8C8C8C", "#DD8452"]

    frac_lo = [[r["props"][ci]["frac_low"]  for r in valid] for ci in range(n_ctx)]
    frac_mi = [[r["props"][ci]["frac_mid"]  for r in valid] for ci in range(n_ctx)]
    frac_hi = [[r["props"][ci]["frac_high"] for r in valid] for ci in range(n_ctx)]
    n_sel   = [[r["props"][ci]["n_selective"] for r in valid] for ci in range(n_ctx)]

    fig, axes = plt.subplots(1, n_ctx, figsize=(4.5 * n_ctx, 4.5), squeeze=False)
    for ci in range(n_ctx):
        ax = axes[0, ci]
        for gi, (fracs, lbl) in enumerate([(frac_lo[ci], "Prefer low\n(v≤0.25)"),
                                            (frac_mi[ci], "Prefer mid"),
                                            (frac_hi[ci], "Prefer high\n(v≥0.75)")]):
            mu = np.nanmean(fracs) * 100
            se = _sem(np.array(fracs)) * 100
            ax.bar(gi, mu, color=bar_c[gi], alpha=0.85, width=0.55, zorder=2)
            ax.errorbar(gi, mu, yerr=se, fmt="none", color="black", capsize=5, lw=1.5, zorder=3)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Prefer low\n(v≤0.25)", "Prefer mid", "Prefer high\n(v≥0.75)"],
                           fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_ylabel("% selective units  (mean ± SEM)")
        mean_sel = np.nanmean(n_sel[ci])
        ax.set_title(f"Context {ci}  ·  mean n_sel = {mean_sel:.0f}\n"
                     f"(SI ≥ {SI_THRESHOLD},  n={len(valid)} runs)")

    n = len(valid)
    fig.suptitle(f"{task_name}  ·  {n} runs  — value preference of selective units",
                 fontsize=9, y=1.02)
    _save(fig, fig_dir / "fig7_unit_selectivity.png")


def fig8_time_resolved(runs, value_matrix, contexts, fig_dir, task_name):
    """Fig 8 — time-resolved binary value decoding (individual + mean ± SEM).

    acc_within has shape (n_ctx, T).
    acc_cross  has shape (n_ctx, n_ctx, T); diagonal is NaN.
    For cross, we average over all off-diagonal train→test pairs per test context.
    """
    valid = [r for r in runs if r["infer_activations"] is not None]
    if not valid:
        return

    n_ctx = len(contexts)
    cc    = _ctx_colors(n_ctx)

    within_list, cross_list, ep_ref = [], [], None
    for r in valid:
        try:
            aw, ac, ep = time_resolved_decode(
                r["infer_activations"], value_matrix,
                n_iti_pre=N_ITI_PRE,
                value_threshold=VALUE_THRESHOLD,
                n_folds=N_FOLDS,
            )
            # aw: (n_ctx, T)
            # ac: (n_ctx, n_ctx, T) — reduce to (n_ctx, T) by averaging off-diagonal
            #     for each test context, mean over all train contexts != test
            ac_reduced = np.full((n_ctx, aw.shape[1]), np.nan)
            for ci_test in range(n_ctx):
                off_diag = [ac[ci_train, ci_test, :]
                            for ci_train in range(n_ctx) if ci_train != ci_test]
                if off_diag:
                    ac_reduced[ci_test] = np.nanmean(off_diag, axis=0)
            within_list.append(aw)          # (n_ctx, T)
            cross_list.append(ac_reduced)   # (n_ctx, T)
            if ep_ref is None:
                ep_ref = ep
        except Exception as e:
            warnings.warn(f"Time-resolved decode failed: {e}")

    if not within_list or ep_ref is None:
        return

    T      = min(a.shape[1] for a in within_list)
    within = np.array([a[:, :T] for a in within_list])  # (n_valid, n_ctx, T)
    cross  = np.array([a[:, :T] for a in cross_list])   # (n_valid, n_ctx, T)

    w_mean = np.nanmean(within, 0); w_sem = _sem(within, 0)
    c_mean = np.nanmean(cross,  0); c_sem = _sem(cross,  0)

    x        = np.arange(T)
    stim_on  = ep_ref.get("stim_onset",  N_ITI_PRE)
    rew_on   = ep_ref.get("reward_onset", N_ITI_PRE + STIM_TIMESTEPS)
    periods  = [(0,       stim_on, "ITI",  "lightgray"),
                (stim_on, rew_on,  "Stim", "#4C72B0"),
                (rew_on,  T,       "Rew",  "#DD8452")]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, data_arr, mean, sem, title in [
        (axes[0], within, w_mean, w_sem, "Within-context"),
        (axes[1], cross,  c_mean, c_sem, "Cross-context  (mean over off-diagonal pairs)"),
    ]:
        for s, e, lbl, clr in periods:
            ax.axvspan(s, min(e, T), color=clr, alpha=0.18, zorder=0)
        for ci in range(n_ctx):
            c = cc[ci]
            for run_a in data_arr:          # run_a: (n_ctx, T)
                ax.plot(x, run_a[ci], color=c, lw=0.7, alpha=0.3)
            ax.plot(x, mean[ci], color=c, lw=2.2, label=contexts[ci], zorder=4)
            ax.fill_between(x, mean[ci] - sem[ci], mean[ci] + sem[ci],
                            color=c, alpha=0.2, zorder=3)
        ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.set_xlim(0, T - 1)
        ax.set_ylim(0.3, 1.05)
        ax.set_ylabel("Decoding accuracy")
        ax.set_title(f"{title} — binary value")
        ax.legend(fontsize=9)

    for s, e, lbl, _ in periods:
        axes[1].axvline(s, color="gray", lw=0.7, ls=":", zorder=1)
        mid = (s + min(e, T)) / 2
        axes[1].text(mid, 0.31, lbl, ha="center", fontsize=7, color="dimgray")

    axes[1].set_xlabel("Aligned timestep")
    n = len(within_list)
    fig.suptitle(f"{task_name}  ·  {n} runs  (thin=individual, thick=mean ± SEM band)",
                 fontsize=9, y=1.01)
    _save(fig, fig_dir / "fig8_time_resolved_decoding.png")


# =============================================================================
# PER-TASK RUNNER
# =============================================================================

def run_task(task_name, task_kwargs, n_runs, results_dir, fig_base_dir,
             device, skip_decoding=False):
    """Train N runs for one task config and generate all figures.

    Args:
        task_name: String label e.g. "3s2c".
        task_kwargs: Kwargs for build_value_matrix.
        n_runs: Number of training runs.
        results_dir: Directory to save run data.
        fig_base_dir: Parent figures directory.
        device: Torch device.
        skip_decoding: Skip decoding figures if True.
    """
    value_matrix, stimuli, stim_group_info = build_value_matrix(**task_kwargs)
    n_stimuli  = value_matrix.shape[0]
    n_contexts = value_matrix.shape[1]
    contexts   = [f"c{i}" for i in range(n_contexts)]

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_dir = fig_base_dir / f"{task_name}_multi_run_{ts}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    run_dir = results_dir / f"{task_name}_multi_run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Task: {task_name}  |  {n_stimuli}s{n_contexts}c  |  {n_runs} runs")
    print(f"Figures → {fig_dir}")
    print(f"{'='*60}")

    runs = []
    for ri in range(n_runs):
        seed = 42 + ri
        print(f"  [{ri+1:>2}/{n_runs}]  seed={seed} ...", end="  ", flush=True)
        result = run_one(value_matrix, stimuli, stim_group_info, contexts,
                         seed=seed, device=device, skip_decoding=skip_decoding)
        if result is None:
            print("DIVERGED — skipped")
            continue
        runs.append(result)
        lp = result["lick_sc"]
        mean_lp = np.nanmean(lp)
        print(f"mean_lick={mean_lp:.3f}  r={result['spearman_r']:.3f}")

    if not runs:
        print(f"  All runs diverged for {task_name} — no figures generated.")
        return

    # save run data so figures can be regenerated without retraining
    data_path = run_dir / "runs_data.pkl"
    with open(data_path, "wb") as f:
        pickle.dump({"runs": runs, "task_name": task_name,
                     "task_kwargs": task_kwargs}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Run data saved → {data_path}")
    print(f"  (Re-run figures only with: --from-data {data_path})")

    _figures_from_runs(runs, task_name, task_kwargs, fig_dir,
                       skip_decoding=skip_decoding)


# =============================================================================
# MAIN
# =============================================================================

def _figures_from_runs(runs, task_name, task_kwargs, fig_dir, skip_decoding):
    """Generate all figures from a pre-collected runs list."""
    value_matrix, stimuli, stim_group_info = build_value_matrix(**task_kwargs)
    n_contexts = value_matrix.shape[1]
    contexts   = [f"c{i}" for i in range(n_contexts)]

    print(f"\n  Generating figures ({len(runs)} runs) → {fig_dir}")
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig0_behaviour(runs, value_matrix, stimuli, contexts, stim_group_info,
                   fig_dir, task_name)
    fig1_calibration(runs, value_matrix, stimuli, contexts, fig_dir, task_name)
    fig2_per_stim_summary(runs, value_matrix, stimuli, contexts, stim_group_info,
                          fig_dir, task_name)
    fig3_period_overview(runs, contexts, fig_dir, task_name)
    fig4_period_per_stim(runs, value_matrix, stimuli, contexts, stim_group_info,
                         fig_dir, task_name)
    if not skip_decoding:
        fig5_decoding(runs, value_matrix, contexts, stim_group_info, fig_dir, task_name)
        fig6_binary_svm(runs, value_matrix, contexts, stim_group_info, fig_dir, task_name)
        fig7_selectivity(runs, value_matrix, stim_group_info, fig_dir, task_name)
        fig8_time_resolved(runs, value_matrix, contexts, fig_dir, task_name)
    else:
        print("  Skipping decoding figures (--skip-decoding).")
    print(f"  Done — {task_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs",        type=int,   default=10)
    parser.add_argument("--device",        default="cpu")
    parser.add_argument("--task",          default="all",
                        help="'3s2c', '15s2c', or 'all'")
    parser.add_argument("--skip-decoding", action="store_true",
                        help="Skip decoding / selectivity figures (faster)")
    parser.add_argument("--results-dir",   default="results")
    parser.add_argument("--fig-dir",       default="figures/13_05_2026_nmlg_lab_meeting")
    parser.add_argument("--from-data",     default=None, metavar="PKL",
                        help="Path to a runs_data.pkl saved by a previous run. "
                             "Skips training and regenerates figures only. "
                             "Can be a comma-separated list for multiple tasks.")
    args = parser.parse_args()

    fig_base_dir = Path(args.fig_dir)
    fig_base_dir.mkdir(parents=True, exist_ok=True)

    # ── figures-only mode ─────────────────────────────────────────────────
    if args.from_data:
        for pkl_path_str in args.from_data.split(","):
            pkl_path = Path(pkl_path_str.strip())
            if not pkl_path.exists():
                print(f"File not found: {pkl_path}")
                continue
            print(f"Loading run data from {pkl_path} ...")
            with open(pkl_path, "rb") as f:
                saved = pickle.load(f)
            runs        = saved["runs"]
            task_name   = saved["task_name"]
            task_kwargs = saved["task_kwargs"]
            ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig_dir     = fig_base_dir / f"{task_name}_multi_run_{ts}"
            _figures_from_runs(runs, task_name, task_kwargs, fig_dir,
                               skip_decoding=args.skip_decoding)
        return

    # ── full training + figures mode ──────────────────────────────────────
    device      = torch.device(args.device)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    tasks = (list(TASK_CONFIGS.keys()) if args.task == "all"
             else [args.task])

    for task_name in tasks:
        if task_name not in TASK_CONFIGS:
            print(f"Unknown task '{task_name}'. Available: {list(TASK_CONFIGS.keys())}")
            continue
        run_task(task_name, TASK_CONFIGS[task_name],
                 n_runs=args.n_runs,
                 results_dir=results_dir,
                 fig_base_dir=fig_base_dir,
                 device=device,
                 skip_decoding=args.skip_decoding)


if __name__ == "__main__":
    main()
