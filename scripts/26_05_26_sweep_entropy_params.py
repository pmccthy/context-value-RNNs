#!/usr/bin/env python
"""
Sweep entropy_coef × reward_lick_miss × lick_cost for the 3-stimulus / 1-context task.

Sweep order (outermost → innermost):
    entropy_coef  →  reward_lick_miss  →  lick_cost  →  seed

This means entropy_coef=0.0 runs complete first, allowing vis-notebook
analysis before entropy_coef=0.01 runs finish.

Fixed:  hidden_size=64, policy_clip=0.1
Varied: entropy_coef     — [0.0, 0.01]
        reward_lick_miss — [0.0, -0.25, -0.5]
        lick_cost        — [0.0, 0.05, 0.1]

Outputs per run:  vis_data.pkl + model.pt in dated subdirectory
Summary:          reward_sweep_partial.csv (live) → reward_sweep_<ts>.csv (final)

Usage
-----
    python scripts/26_05_26_sweep_entropy_params.py
    python scripts/26_05_26_sweep_entropy_params.py --seeds 3 --device mps
    python scripts/26_05_26_sweep_entropy_params.py --out-dir results/26_05_26_sweep_entropy_params
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

ENTROPY_COEF_VALS = [0.0]           # outermost — 0.0 runs first
LICK_MISS_VALS    = [0.0, -0.25, -0.5]    # reward_lick_miss
LICK_COST_VALS    = [0.0,  0.05,  0.1]    # lick_cost

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
CHECKPOINT_EVERY = 500    # save model weights every N trials during training
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

def run_one(hidden_size: int, entropy_coef: float,
            reward_lick_miss: float, lick_cost: float,
            seed: int, device: torch.device, out_dir: Path) -> dict:
    ec_tag = str(entropy_coef).replace(".", "p")
    lm_tag = str(reward_lick_miss).replace("-", "n").replace(".", "p")
    lc_tag = str(lick_cost).replace(".", "p")
    tag    = f"ec={entropy_coef}  miss={reward_lick_miss}  cost={lick_cost}  h={hidden_size}  seed={seed}"

    nan_row = dict(
        entropy_coef=entropy_coef,
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
    initial_state_dict = {k: v.cpu().clone() for k, v in actor_critic.state_dict().items()}
    optimizer    = torch.optim.Adam(actor_critic.parameters(), lr=LR)

    # ── run directory (created early so checkpoints can be saved during training) ─
    run_id  = (f"RNN_ec{ec_tag}_lm{lm_tag}_lc{lc_tag}"
               f"_h{hidden_size}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── training ──────────────────────────────────────────────────────────
    actor_critic.train()
    optimizer.zero_grad()
    obs, _     = env.reset()
    hidden     = None
    action_list, value_ts = [], []
    lp_buf, val_buf, rew_buf, ent_buf = [], [], [], []
    t_win    = 0
    t_global = 0

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    _ckpt_ts = {
        trial["reward_window"][1]: (ti + 1)
        for ti, trial in enumerate(state_seq.trial_structure)
        if (ti + 1) % CHECKPOINT_EVERY == 0 or ti + 1 == n_trials
    }

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
            t_win    += 1
            t_global += 1

            if t_global in _ckpt_ts:
                torch.save(
                    {k: v.cpu().clone() for k, v in actor_critic.state_dict().items()},
                    ckpt_dir / f"checkpoint_{_ckpt_ts[t_global]:05d}.pt",
                )

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
                         + VALUE_COEF  * F.mse_loss(val_t, ret_t)
                         - entropy_coef * ent_m)

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
        "entropy_coef":              entropy_coef,
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
    torch.save(initial_state_dict, run_dir / "model_init.pt")

    return dict(
        entropy_coef=entropy_coef,
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
# PLOTTING  (called once per entropy_coef slice)
# =============================================================================

def plot_results(df: pd.DataFrame, lick_miss_vals, lick_cost_vals,
                 out_dir: Path, ts: str, ec_label: str) -> None:
    miss_lbls = [str(v) for v in lick_miss_vals]
    cost_lbls = [str(v) for v in lick_cost_vals]
    n_miss    = len(lick_miss_vals)
    n_cost    = len(lick_cost_vals)

    def _grid(col):
        g = np.full((n_cost, n_miss), np.nan)
        for ci, lc in enumerate(lick_cost_vals):
            for mi, lm in enumerate(lick_miss_vals):
                vals = df.loc[
                    (df["lick_cost"] == lc) & (df["reward_lick_miss"] == lm), col
                ].dropna()
                if len(vals):
                    g[ci, mi] = float(vals.mean())
        return g

    def _heatmap(ax, data, title, vmin, vmax, cmap):
        im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        for ci in range(n_cost):
            for mi in range(n_miss):
                v = data[ci, mi]
                if not np.isnan(v):
                    ax.text(mi, ci, f"{v:.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(n_miss)); ax.set_xticklabels(miss_lbls, fontsize=8)
        ax.set_yticks(range(n_cost)); ax.set_yticklabels(cost_lbls, fontsize=8)
        ax.set_xlabel("reward_lick_miss", fontsize=8)
        ax.set_ylabel("lick_cost", fontsize=8)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)

    n_seeds  = df["seed"].nunique()
    slug     = ec_label.replace("=", "").replace(".", "p")

    PERF = [
        ("spearman_r", "Spearman r  (lick–value)", -1, 1, "RdYlGn"),
        ("psa_score",  "PSA score  (↑→1)",          0, 1, "RdYlGn"),
        ("psa_delta",  "PSA delta  (high−low)",      0, 1, "RdYlGn"),
        ("lick_high",  "Lick — high stim  (↑→1)",   0, 1, "RdYlGn"),
        ("lick_mid",   "Lick — mid stim  (~0.5)",    0, 1, "RdYlGn"),
        ("lick_low",   "Lick — low stim  (↓→0)",     0, 1, "RdYlGn_r"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (col, title, vmin, vmax, cmap) in zip(axes.flat, PERF):
        _heatmap(ax, _grid(col), title, vmin, vmax, cmap)
    fig.suptitle(
        f"Performance  [{ec_label}]  — mean over {n_seeds} seeds\n"
        "rows = lick_cost  ·  cols = reward_lick_miss",
        y=1.02, fontsize=10,
    )
    plt.tight_layout()
    p = out_dir / f"perf_{slug}_{ts}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Performance → {p}")

    SEL = [
        ("frac_selective", "Fraction selective",   0, 1, "Blues"),
        ("frac_low",       "Prefer low  (of sel)", 0, 1, "Blues"),
        ("frac_mid",       "Prefer mid  (of sel)", 0, 1, "Blues"),
        ("frac_high",      "Prefer high (of sel)", 0, 1, "Blues"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, (col, title, vmin, vmax, cmap) in zip(axes, SEL):
        _heatmap(ax, _grid(col), title, vmin, vmax, cmap)
    fig.suptitle(
        f"Selectivity  [{ec_label}]  — mean over {n_seeds} seeds\n"
        "rows = lick_cost  ·  cols = reward_lick_miss",
        y=1.03, fontsize=10,
    )
    plt.tight_layout()
    p = out_dir / f"selectivity_{slug}_{ts}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Selectivity → {p}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seeds",       type=int, default=3)
    parser.add_argument("--base-seed",   type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--out-dir",     default="results/26_05_26_sweep_entropy_params")
    args = parser.parse_args()

    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # entropy outermost so ec=0 jobs finish first
    configs = [
        (ec, lm, lc)
        for ec in ENTROPY_COEF_VALS
        for lm in LICK_MISS_VALS
        for lc in LICK_COST_VALS
    ]
    n_configs = len(configs)
    n_runs    = n_configs * args.seeds

    print(f"Entropy × reward-param sweep  —  {n_configs} configs × {args.seeds} seeds = {n_runs} runs")
    print(f"hidden_size={args.hidden_size}  policy_clip={POLICY_CLIP}  device={device}")
    print(f"entropy_coef:     {ENTROPY_COEF_VALS}")
    print(f"reward_lick_miss: {LICK_MISS_VALS}")
    print(f"lick_cost:        {LICK_COST_VALS}\n")

    partial_csv = out_dir / "reward_sweep_partial.csv"

    # Resume: collect done keys from partial CSV
    if partial_csv.exists():
        existing  = pd.read_csv(partial_csv)
        rows      = existing.to_dict("records")
        done_keys = {
            (r["entropy_coef"], r["reward_lick_miss"], r["lick_cost"], r["seed"])
            for r in rows
        }
        print(f"Resuming from partial CSV — {len(rows)} runs already recorded.\n")
    else:
        rows      = []
        done_keys = set()

    # Also pick up any pkl dirs written before the CSV was flushed
    for pkl in out_dir.rglob("vis_data.pkl"):
        with open(pkl, "rb") as _f:
            _d = pickle.load(_f)
        key = (
            _d.get("entropy_coef"),
            _d.get("reward_lick_miss"),
            _d.get("lick_cost"),
            _d.get("seed"),
        )
        if None in key or key in done_keys:
            continue
        done_keys.add(key)
        _psa = _d.get("psa_results", {0: {}}).get(0, {})
        rows.append(dict(
            entropy_coef=key[0], reward_lick_miss=key[1], lick_cost=key[2], seed=key[3],
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
    for ec, lm, lc in configs:
        for s in range(args.seeds):
            seed = args.base_seed + s
            if (ec, lm, lc, seed) in done_keys:
                print(f"[ skip ]  ec={ec}  miss={lm:>6}  cost={lc:.2f}  seed={seed}")
                continue

            n_done += 1
            print(f"[{n_done:>3}/{n_runs}]  ec={ec}  miss={lm:>6}  cost={lc:.2f}  seed={seed}",
                  end="  ", flush=True)
            row = run_one(args.hidden_size, ec, lm, lc, seed, device, out_dir)
            rows.append(row)

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
    partial_csv.unlink(missing_ok=True)
    print(f"\nResults → {csv}")

    print("\nGenerating summary figures...")
    for ec in ENTROPY_COEF_VALS:
        sub = df[df["entropy_coef"] == ec]
        if len(sub):
            print(f"entropy_coef={ec}  ({len(sub)} rows)")
            plot_results(sub, LICK_MISS_VALS, LICK_COST_VALS, out_dir, ts, f"ec={ec}")


if __name__ == "__main__":
    main()
