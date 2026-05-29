#!/usr/bin/env python
"""
Sweep reward_lick_miss × lick_cost for the 3-stimulus / 1-context task.

Fixed:  hidden_size=64, policy_clip=0.1, entropy_coef=0.0
Varied: reward_lick_miss  — penalty for licking on an unrewarded trial
        lick_cost         — penalty per lick action regardless of reward

Outputs per run:  PSA metrics, Spearman r, lick rates, neuron selectivity
Figures:          2D heatmaps (rows=lick_cost, cols=reward_lick_miss)

Usage
-----
    python scripts/21_05_26_sweep_reward_params.py
    python scripts/21_05_26_sweep_reward_params.py --seeds 5 --device mps
    python scripts/21_05_26_sweep_reward_params.py --hidden-size 32
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
from cxval.analysis import (
    compute_unit_tuning, preferred_value_proportions, policy_similarity_analysis,
)

# =============================================================================
# SWEEP GRID  ← edit here
# =============================================================================

LICK_MISS_VALS = [0.0, -0.25, -0.5]      # reward_lick_miss
LICK_COST_VALS = [0.0,  0.05,  0.1]     # lick_cost

# =============================================================================
# TASK
# =============================================================================

VALUE_MATRIX = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)
N_STIMULI    = 3
N_CONTEXTS   = 1
STIMULI      = ["s0 (0%)", "s1 (50%)", "s2 (100%)"]
CONTEXTS     = ["c0"]

STIM_GROUP_INFO = [
    {"name": "low",  "short": "low",  "indices": [0]},
    {"name": "mid",  "short": "mid",  "indices": [1]},
    {"name": "high", "short": "high", "indices": [2]},
]

# =============================================================================
# FIXED HYPERPARAMETERS
# =============================================================================

RECURRENT_GAIN   = 0.9
POLICY_CLIP      = 0.1
ENTROPY_COEF     = 0.0    # disabled for this sweep
REWARD_LICK      = 1.0
REWARD_NO_LICK   = 0.0
BPTT_LEN         = 10
UPDATE_EVERY     = 10
GAMMA            = 0.9
LR               = 9e-4
VALUE_COEF       = 0.5
GRAD_CLIP        = 1.0
TRIALS_PER_PHASE = 300
PHASES_PER_CTX   = 1
CONTEXT_REPS     = 30
STIM_TIMESTEPS   = 5
REWARD_TIMESTEPS = 3
ITI_TIMESTEPS    = (3, 8)

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

def run_one(hidden_size: int, reward_lick_miss: float, lick_cost: float,
            seed: int, device: torch.device, out_dir: Path) -> dict:
    lm_tag = str(reward_lick_miss).replace("-", "n").replace(".", "p")
    lc_tag = str(lick_cost).replace(".", "p")
    tag    = f"miss={reward_lick_miss}  cost={lick_cost}  h={hidden_size}  seed={seed}"

    nan_row = dict(
        reward_lick_miss=reward_lick_miss, lick_cost=lick_cost,
        hidden_size=hidden_size, seed=seed,
        spearman_r=np.nan,
        psa_score=np.nan, psa_delta=np.nan,
        lick_high=np.nan, lick_mid=np.nan, lick_low=np.nan,
        frac_selective=np.nan,
        frac_low=np.nan, frac_mid=np.nan, frac_high=np.nan,
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
    obs_dim  = states.shape[1] + 2
    n_trials = len(stim_seq.trial_contexts)

    env = TaskEnv(
        states=states, reward_availability=reward_availability,
        reward_lick=REWARD_LICK, reward_no_lick=REWARD_NO_LICK,
        reward_lick_miss=reward_lick_miss, lick_cost=lick_cost,
    )

    # ── model ─────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    backbone     = RNN(input_size=obs_dim, hidden_size=hidden_size,
                       output_size=1, recurrent_gain=RECURRENT_GAIN)
    actor_critic = ActorCritic(backbone=backbone, num_actions=2,
                               policy_clip=POLICY_CLIP).to(device)
    optimizer    = torch.optim.Adam(actor_critic.parameters(), lr=LR)

    # ── training ──────────────────────────────────────────────────────────
    actor_critic.train()
    optimizer.zero_grad()
    obs, _     = env.reset()
    hidden     = None
    action_list, value_ts = [], []
    lp_buf, val_buf, rew_buf, ent_buf = [], [], [], []
    t_win = 0

    try:
        done = False
        while not done:
            obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value, hidden = actor_critic.step(obs_t, hidden)
            dist   = actor_critic.make_dist(logits)
            action = dist.sample()

            lp_buf.append(dist.log_prob(action))
            val_buf.append(value)
            ent_buf.append(dist.entropy())
            action_list.append(action.item())
            value_ts.append(value.detach().item())

            obs, reward, done, _, _ = env.step(action.item())
            rew_buf.append(reward)
            t_win += 1

            if t_win % UPDATE_EVERY == 0 or done:
                bv = 0.0
                if not done:
                    with torch.no_grad():
                        obs_n = torch.tensor(obs, dtype=torch.float32,
                                             device=device).unsqueeze(0)
                        _, bv_t, _ = actor_critic.step(obs_n, hidden)
                        bv = bv_t.item()

                ret_t  = torch.tensor(
                    compute_returns(rew_buf, bv, GAMMA),
                    dtype=torch.float32, device=device,
                )
                lp_t  = torch.stack(lp_buf).squeeze(-1)
                val_t = torch.stack(val_buf).squeeze(-1)
                ent_m = torch.stack(ent_buf).mean()
                adv   = ret_t - val_t.detach()
                std   = adv.std()
                adv   = (adv - adv.mean()) / (std + 1e-8) if std > 1e-4 else adv - adv.mean()

                loss  = (-(lp_t * adv).mean()
                         + VALUE_COEF   * F.mse_loss(val_t, ret_t)
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
                    if any(torch.isnan(p).any() for p in actor_critic.parameters()):
                        warnings.warn(f"NaN divergence ({tag})")
                        return nan_row

    except Exception as exc:
        warnings.warn(f"Training crashed ({tag}): {exc}")
        return nan_row

    all_trial_data = []
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
        reward_lick=REWARD_LICK, reward_no_lick=REWARD_NO_LICK,
        reward_lick_miss=reward_lick_miss, lick_cost=lick_cost,
    )

    actor_critic.eval()
    agent = Agent(actor_critic, device=device)
    agent.reset()
    obs, _ = infer_env.reset()
    infer_struct    = infer_state_seq.trial_structure
    infer_action_seq, infer_value_ts, infer_hidden_list = [], [], []

    done = False
    while not done:
        action, _, value = agent.act(obs)
        infer_action_seq.append(action)
        infer_value_ts.append(value.item())
        infer_hidden_list.append(agent.hidden.detach().squeeze(0).cpu())
        obs, _, done, _, _ = infer_env.step(action)

    infer_hidden_np = np.array(torch.stack(infer_hidden_list).tolist(), dtype=np.float32)

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

    # ── PSA metrics ───────────────────────────────────────────────────────
    inf_lick = np.array([d["licked"] for d in infer_trial_data])
    lick_sc  = np.full((N_STIMULI, N_CONTEXTS), np.nan)
    for si in range(N_STIMULI):
        for ci in range(N_CONTEXTS):
            m = (stim_arr == si) & (ctx_arr == ci)
            if m.sum():
                lick_sc[si, ci] = inf_lick[m].mean()

    psa     = policy_similarity_analysis(lick_sc, VALUE_MATRIX)
    lp_flat = [lick_sc[si, 0] for si in range(N_STIMULI) if not np.isnan(lick_sc[si, 0])]
    rp_flat = [float(VALUE_MATRIX[si, 0]) for si in range(N_STIMULI)
               if not np.isnan(lick_sc[si, 0])]
    spear_r = float(spearmanr(rp_flat, lp_flat)[0]) if len(lp_flat) > 2 else np.nan

    # ── selectivity ───────────────────────────────────────────────────────
    tuning        = compute_unit_tuning(infer_activations, period="stim")
    props, _      = preferred_value_proportions(
        tuning, VALUE_MATRIX,
        si_threshold=SI_THRESHOLD, value_threshold=VALUE_THRESHOLD,
    )
    n_sel         = props[0]["n_selective"]
    frac_sel      = n_sel / hidden_size
    frac_low_sel  = props[0].get("frac_low",  np.nan)
    frac_mid_sel  = props[0].get("frac_mid",  np.nan)
    frac_high_sel = props[0].get("frac_high", np.nan)

    # ── save vis_data.pkl + model.pt ──────────────────────────────────────
    run_id  = f"RNN_rp_lm{lm_tag}_lc{lc_tag}_h{hidden_size}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    vis_data = {
        # task
        "value_matrix":              VALUE_MATRIX,
        "n_stimuli":                 N_STIMULI,
        "n_contexts":                N_CONTEXTS,
        "stimuli":                   STIMULI,
        "contexts":                  CONTEXTS,
        "stim_group_info":           STIM_GROUP_INFO,
        "stim_timesteps":            STIM_TIMESTEPS,
        "reward_timesteps":          REWARD_TIMESTEPS,
        "reward_lick":               REWARD_LICK,
        "reward_lick_miss":          reward_lick_miss,
        "reward_no_lick":            REWARD_NO_LICK,
        "lick_cost":                 lick_cost,
        # training
        "n_trials":                  n_trials,
        "trials_per_phase":          TRIALS_PER_PHASE,
        "context_reps":              CONTEXT_REPS,
        "all_trial_data":            all_trial_data,
        # inference
        "infer_trial_data":          infer_trial_data,
        "infer_action_seq":          infer_action_seq,
        "infer_trial_structure":     infer_struct,
        "infer_activations":         infer_activations,
        # for injection experiments
        "infer_states":              infer_states,
        "infer_reward_availability": infer_ravail,
        # decoding helpers
        "stim_groups":               {"all": None},
        "group_colors":              {"all": "black"},
        "pooling":                   "average",
        "n_folds":                   5,
        # per-run identity
        "hidden_size":               hidden_size,
        "seed":                      seed,
        "run_id":                    run_id,
        "lick_sc":                   lick_sc,
        "psa_results":               psa,
        "spearman_r":                spear_r,
        "env_kwargs": {
            "reward_lick": REWARD_LICK, "reward_no_lick": REWARD_NO_LICK,
            "reward_lick_miss": reward_lick_miss, "lick_cost": lick_cost,
        },
    }

    with open(run_dir / "vis_data.pkl", "wb") as f:
        pickle.dump(vis_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(actor_critic.state_dict(), run_dir / "model.pt")

    return dict(
        reward_lick_miss=reward_lick_miss, lick_cost=lick_cost,
        hidden_size=hidden_size, seed=seed,
        spearman_r=spear_r,
        psa_score=float(psa[0]["psa_score"]),
        psa_delta=float(psa[0]["psa_delta"]),
        lick_high=float(psa[0]["high_lick"]),
        lick_mid=float(psa[0]["mid_lick"]),
        lick_low=float(psa[0]["low_lick"]),
        frac_selective=frac_sel,
        frac_low=frac_low_sel,
        frac_mid=frac_mid_sel,
        frac_high=frac_high_sel,
        diverged=False,
    )


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(df: pd.DataFrame, lick_miss_vals, lick_cost_vals,
                 out_dir: Path, ts: str) -> None:
    miss_lbls = [str(v) for v in lick_miss_vals]
    cost_lbls = [str(v) for v in lick_cost_vals]
    n_miss    = len(lick_miss_vals)
    n_cost    = len(lick_cost_vals)

    def _grid(col):
        """Mean over seeds → (n_cost, n_miss) array."""
        g = np.full((n_cost, n_miss), np.nan)
        for ci, lc in enumerate(lick_cost_vals):
            for mi, lm in enumerate(lick_miss_vals):
                vals = df.loc[
                    (df["lick_cost"] == lc) & (df["reward_lick_miss"] == lm), col
                ].dropna()
                if len(vals):
                    g[ci, mi] = float(vals.mean())
        return g

    def _heatmap(ax, data, title, vmin, vmax, cmap, fmt=".2f"):
        im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        for ci in range(n_cost):
            for mi in range(n_miss):
                v = data[ci, mi]
                if not np.isnan(v):
                    ax.text(mi, ci, f"{v:{fmt}}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(n_miss)); ax.set_xticklabels(miss_lbls, fontsize=8)
        ax.set_yticks(range(n_cost)); ax.set_yticklabels(cost_lbls, fontsize=8)
        ax.set_xlabel("reward_lick_miss", fontsize=8)
        ax.set_ylabel("lick_cost", fontsize=8)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)

    n_seeds = df["seed"].nunique()

    # ── Figure 1: performance heatmaps ────────────────────────────────────
    PERF = [
        ("spearman_r", "Lick–value calibration\n(Spearman r, ↑→1)", -1, 1,  "RdYlGn"),
        ("psa_score",  "PSA score\n(1−MAE, ↑→1)",                    0, 1,  "RdYlGn"),
        ("psa_delta",  "PSA delta\n(high−low, ↑→1)",                  0, 1,  "RdYlGn"),
        ("lick_high",  "Lick rate — high stim\n(↑→1)",                0, 1,  "RdYlGn"),
        ("lick_mid",   "Lick rate — mid stim\n(~0.5)",                 0, 1,  "RdYlGn"),
        ("lick_low",   "Lick rate — low stim\n(↓→0)",                  0, 1,  "RdYlGn_r"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (col, title, vmin, vmax, cmap) in zip(axes.flat, PERF):
        _heatmap(ax, _grid(col), title, vmin, vmax, cmap)
    fig.suptitle(
        f"Performance — mean over {n_seeds} seeds\n"
        "rows = lick_cost  ·  cols = reward_lick_miss",
        y=1.02, fontsize=10,
    )
    plt.tight_layout()
    p = out_dir / f"reward_sweep_performance_{ts}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Performance figure → {p}")

    # ── Figure 2: selectivity heatmaps ────────────────────────────────────
    SEL = [
        ("frac_selective", "Fraction selective\n(SI ≥ threshold)",  0, 1, "Blues"),
        ("frac_low",       "Selective: prefer low\n(frac of n_sel)",  0, 1, "Blues"),
        ("frac_mid",       "Selective: prefer mid\n(frac of n_sel)",  0, 1, "Blues"),
        ("frac_high",      "Selective: prefer high\n(frac of n_sel)", 0, 1, "Blues"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, (col, title, vmin, vmax, cmap) in zip(axes, SEL):
        _heatmap(ax, _grid(col), title, vmin, vmax, cmap)
    fig.suptitle(
        f"Neuron selectivity — mean over {n_seeds} seeds\n"
        "rows = lick_cost  ·  cols = reward_lick_miss",
        y=1.03, fontsize=10,
    )
    plt.tight_layout()
    p = out_dir / f"reward_sweep_selectivity_{ts}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Selectivity figure → {p}")

    # ── Figure 3: selectivity composition grid ────────────────────────────
    # Small stacked bar per cell showing frac low/mid/high of selective neurons
    fig, axes = plt.subplots(n_cost, n_miss,
                             figsize=(2.2 * n_miss, 2.2 * n_cost),
                             sharex=True, sharey=True)
    bar_cols = {"low": "#4C72B0", "mid": "#8C8C8C", "high": "#DD8452"}
    for ci, lc in enumerate(lick_cost_vals):
        for mi, lm in enumerate(lick_miss_vals):
            ax   = axes[ci, mi]
            sub  = df[(df["lick_cost"] == lc) & (df["reward_lick_miss"] == lm)]
            fracs = {k: float(sub[f"frac_{k}"].mean()) for k in ("low", "mid", "high")}
            bottom = 0.0
            for k, c in bar_cols.items():
                v = fracs[k] if not np.isnan(fracs[k]) else 0.0
                ax.bar(0, v * 100, bottom=bottom * 100, color=c, width=0.6, label=k)
                if v > 0.05:
                    ax.text(0, (bottom + v / 2) * 100, f"{v*100:.0f}%",
                            ha="center", va="center", fontsize=7, color="white",
                            fontweight="bold")
                bottom += v
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(0, 105)
            ax.set_xticks([])
            if ci == 0:
                ax.set_title(f"miss={lm}", fontsize=7)
            if mi == 0:
                ax.set_ylabel(f"cost={lc}", fontsize=7)

    handles = [plt.Rectangle((0,0),1,1, color=c) for c in bar_cols.values()]
    fig.legend(handles, list(bar_cols.keys()), loc="lower right",
               fontsize=8, title="Prefers", frameon=True)
    fig.suptitle(
        "Selectivity composition  (% of selective neurons preferring each value)\n"
        "rows = lick_cost  ·  cols = reward_lick_miss",
        y=1.02, fontsize=9,
    )
    plt.tight_layout()
    p = out_dir / f"reward_sweep_selectivity_grid_{ts}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Selectivity grid → {p}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seeds",       type=int,   default=3)
    parser.add_argument("--base-seed",   type=int,   default=42)
    parser.add_argument("--hidden-size", type=int,   default=64)
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--out-dir",     default="results/21_05_26_sweep_reward_params")
    args = parser.parse_args()

    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        (lm, lc)
        for lm in LICK_MISS_VALS
        for lc in LICK_COST_VALS
    ]
    n_configs = len(configs)
    n_runs    = n_configs * args.seeds

    print(f"Reward-param sweep  —  {n_configs} configs × {args.seeds} seeds = {n_runs} runs")
    print(f"hidden_size={args.hidden_size}  policy_clip={POLICY_CLIP}  "
          f"entropy_coef={ENTROPY_COEF}  device={device}")
    print(f"reward_lick_miss: {LICK_MISS_VALS}")
    print(f"lick_cost:        {LICK_COST_VALS}\n")

    partial_csv = out_dir / "reward_sweep_partial.csv"

    # Resume: collect done keys from partial CSV and any existing vis_data.pkl dirs
    if partial_csv.exists():
        existing  = pd.read_csv(partial_csv)
        rows      = existing.to_dict("records")
        done_keys = {(r["reward_lick_miss"], r["lick_cost"], r["seed"]) for r in rows}
        print(f"Resuming from partial CSV — {len(rows)} runs already recorded.\n")
    else:
        rows      = []
        done_keys = set()

    # Also skip any runs that already have a vis_data.pkl on disk (handles re-runs
    # after the final CSV was written but before it was fully processed)
    for pkl in out_dir.rglob("vis_data.pkl"):
        import pickle as _pkl
        with open(pkl, "rb") as _f:
            _d = _pkl.load(_f)
        key = (_d.get("reward_lick_miss"), _d.get("lick_cost"), _d.get("seed"))
        if key not in done_keys and None not in key:
            done_keys.add(key)
            # reconstruct a summary row from the pkl so the final CSV is complete
            _psa = _d.get("psa_results", {0: {}}).get(0, {})
            rows.append(dict(
                reward_lick_miss=key[0], lick_cost=key[1], seed=key[2],
                hidden_size=_d.get("hidden_size", args.hidden_size),
                spearman_r=_d.get("spearman_r", np.nan),
                psa_score=_psa.get("psa_score",  np.nan),
                psa_delta=_psa.get("psa_delta",  np.nan),
                lick_high=_psa.get("high_lick",  np.nan),
                lick_mid=_psa.get("mid_lick",    np.nan),
                lick_low=_psa.get("low_lick",    np.nan),
                frac_selective=np.nan, frac_low=np.nan, frac_mid=np.nan,
                frac_high=np.nan, diverged=False,
            ))

    if done_keys:
        print(f"Skipping {len(done_keys)} already-completed runs.\n")

    n_done = len(rows)
    for lm, lc in configs:
        for s in range(args.seeds):
            seed = args.base_seed + s
            if (lm, lc, seed) in done_keys:
                print(f"[ skip ]  miss={lm:>6}  cost={lc:.2f}  seed={seed}  (already done)")
                continue

            n_done += 1
            print(f"[{n_done:>3}/{n_runs}]  miss={lm:>6}  cost={lc:.2f}  seed={seed}",
                  end="  ", flush=True)
            row = run_one(args.hidden_size, lm, lc, seed, device, out_dir)
            rows.append(row)

            # Save after every completed run so a crash loses at most one run
            pd.DataFrame(rows).to_csv(partial_csv, index=False)

            if row["diverged"]:
                print("DIVERGED")
            else:
                print(f"psa={row['psa_score']:.3f}  "
                      f"hi={row['lick_high']:.2f}  "
                      f"mid={row['lick_mid']:.2f}  "
                      f"lo={row['lick_low']:.2f}  "
                      f"sel={row['frac_selective']:.2f}")

    df  = pd.DataFrame(rows)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = out_dir / f"reward_sweep_{ts}.csv"
    df.to_csv(csv, index=False)
    partial_csv.unlink(missing_ok=True)   # clean up partial file now we have the final
    print(f"\nResults → {csv}")

    plot_results(df, LICK_MISS_VALS, LICK_COST_VALS, out_dir, ts)


if __name__ == "__main__":
    main()
