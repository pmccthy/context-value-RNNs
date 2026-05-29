#!/usr/bin/env python
"""
Reproduce the cogNN  22_04_26_norev_ace_nopunish_model_sweep  parameters
within the cxval repo.

Source:  cogNN/experiment_scripts/run_22_04_26_norev_ace_model_sweep_both.sh
         cogNN/experiment_scripts/run_ace_plasticity_multirun_v2.py

Parameters matched exactly
--------------------------
  lick_no_reward  = 0.0    (nopunish variant)
  lick_cost       = 0.0
  gamma           = 0.0    ← key: no temporal discounting
  learning_rate   = 5e-4
  policy_clip     = 0.25
  n_runs          = 10     (seeds: 42, 49, 56, … step 7)
  trials_per_phase= 1000   × 2 training phases = 2000 training trials
  hidden_sizes    = [2, 4, 8, 16, 32, 64, 128]
  update_every    = 1      (per-timestep gradient update, no BPTT)
  no context signal (1 context, no context input)

Architectural differences
-------------------------
  cogNN uses PARTIAL READOUT: readout_fraction=0.5 — only the first
  hidden_size//2 neurons connect to the actor/critic heads.  The remaining
  half participate in recurrent dynamics but have no output projection.
  cxval now supports the same via ActorCritic(readout_fraction=0.5),
  which is used here to match cogNN exactly.

Task differences
----------------
  cogNN:  A=100%, C=50%, E=0% stimuli, 1-step stim/reward windows,
          no ITI, reversal framework (but no actual reversal for norev).
  cxval:  s0=0%, s1=50%, s2=100%, stim_timesteps=5, reward_timesteps=3,
          ITI uniform(3,8).  Stimuli are flipped in ordering convention
          (cogNN high stim = A, here high stim = s2) but value structure
          is equivalent.

Usage
-----
    python scripts/27_05_26_sweep_cognn_equivalent.py
    python scripts/27_05_26_sweep_cognn_equivalent.py --device mps
    python scripts/27_05_26_sweep_cognn_equivalent.py --out-dir results/27_05_26_sweep_cognn_equivalent
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
# SWEEP GRID
# =============================================================================

HIDDEN_SIZES = [2, 4, 8, 16, 32, 64, 128]   # full cogNN sweep; override with --hidden-sizes
N_RUNS       = 10                              # matches cogNN n_runs
BASE_SEED    = 42                              # cogNN seed formula: run_idx*7+42
SEED_STEP    = 7                               # cogNN: seed = run_idx * 7 + 42

# =============================================================================
# TASK  (cxval equivalent of ACE norev nopunish)
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
# HYPERPARAMETERS  — matched to cogNN where possible
# =============================================================================

# ── matched to cogNN ──────────────────────────────────────────────────────────
REWARD_LICK      = 1.0
REWARD_NO_LICK   = 0.0
REWARD_LICK_MISS = 0.0   # cogNN lick_no_reward = 0.0 (nopunish)
LICK_COST        = 0.0
GAMMA            = 0.0   # cogNN gamma = 0.0  ← critical difference vs prior sweeps
LR               = 5e-4  # cogNN learning_rate = 5e-4
POLICY_CLIP      = 0.25  # cogNN policy_clip = 0.25  (clips action probs in cxval)
READOUT_FRACTION = 0.5   # cogNN readout_fraction = 0.5 (partial readout)
UPDATE_EVERY     = 1     # cogNN updates every timestep
BPTT_LEN         = 1     # no backprop-through-time (consistent with per-step update)
TRIALS_PER_PHASE = 1000  # cogNN trials_per_phase = 1000
CONTEXT_REPS     = 2     # 2 phases × 1000 trials = 2000 training trials (cogNN: 2 train phases)

# ── cxval-specific (no direct cogNN equivalent) ───────────────────────────────
RECURRENT_GAIN   = 0.9
ENTROPY_COEF     = 0.0   # off (same as cogNN)
VALUE_COEF       = 0.5
GRAD_CLIP        = 1.0
STIM_TIMESTEPS   = 5
REWARD_TIMESTEPS = 3
ITI_TIMESTEPS    = (3, 8)
PHASES_PER_CTX   = 1

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

def run_one(hidden_size: int, seed: int, device: torch.device, out_dir: Path) -> dict:
    tag = f"h={hidden_size}  seed={seed}"

    nan_row = dict(
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
        reward_lick_miss=REWARD_LICK_MISS, lick_cost=LICK_COST,
    )

    # ── model ─────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    backbone     = RNN(input_size=obs_dim, hidden_size=hidden_size,
                       output_size=1, recurrent_gain=RECURRENT_GAIN)
    actor_critic = ActorCritic(backbone=backbone, num_actions=2,
                               policy_clip=POLICY_CLIP,
                               readout_fraction=READOUT_FRACTION).to(device)
    optimizer    = torch.optim.Adam(actor_critic.parameters(), lr=LR)

    # ── training  (per-timestep update, no BPTT — matches cogNN) ─────────
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
        reward_lick_miss=REWARD_LICK_MISS, lick_cost=LICK_COST,
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
    tuning   = compute_unit_tuning(infer_activations, period="stim")
    props, _ = preferred_value_proportions(
        tuning, VALUE_MATRIX,
        si_threshold=SI_THRESHOLD, value_threshold=VALUE_THRESHOLD,
    )
    n_sel         = props[0]["n_selective"]
    frac_sel      = n_sel / hidden_size
    frac_low_sel  = props[0].get("frac_low",  np.nan)
    frac_mid_sel  = props[0].get("frac_mid",  np.nan)
    frac_high_sel = props[0].get("frac_high", np.nan)

    # ── save vis_data.pkl + model.pt ──────────────────────────────────────
    run_id  = f"RNN_cognn_h{hidden_size}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        "reward_lick_miss":          REWARD_LICK_MISS,
        "reward_no_lick":            REWARD_NO_LICK,
        "lick_cost":                 LICK_COST,
        "entropy_coef":              ENTROPY_COEF,
        "readout_fraction":          READOUT_FRACTION,
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
            "reward_lick_miss": REWARD_LICK_MISS, "lick_cost": LICK_COST,
        },
    }

    with open(run_dir / "vis_data.pkl", "wb") as f:
        pickle.dump(vis_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(actor_critic.state_dict(), run_dir / "model.pt")

    return dict(
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
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--out-dir", default="results/27_05_26_sweep_cognn_equivalent")
    parser.add_argument("--hidden-sizes", default="64",
                        help="Comma-separated hidden sizes to run (default: 64)")
    parser.add_argument("--n-runs", type=int, default=N_RUNS,
                        help=f"Number of seeds per hidden size (default: {N_RUNS})")
    args = parser.parse_args()

    device       = torch.device(args.device)
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hidden_sizes = [int(h.strip()) for h in args.hidden_sizes.split(",")]
    n_runs       = args.n_runs

    n_sizes  = len(hidden_sizes)
    n_total  = n_sizes * n_runs

    print("cogNN-equivalent sweep  (nopunish, gamma=0, lr=5e-4, per-step update)")
    print(f"  hidden_sizes : {hidden_sizes}")
    print(f"  n_runs       : {n_runs}  (seeds: {BASE_SEED} + run_idx*{SEED_STEP})")
    print(f"  train trials : {TRIALS_PER_PHASE} × {CONTEXT_REPS} = {TRIALS_PER_PHASE*CONTEXT_REPS}")
    print(f"  gamma        : {GAMMA}  (cogNN-matched)")
    print(f"  lr           : {LR}  (cogNN-matched)")
    print(f"  policy_clip  : {POLICY_CLIP}  (cogNN-matched)")
    print(f"  readout_frac : {READOUT_FRACTION}  (cogNN-matched: first {READOUT_FRACTION*100:.0f}% of neurons → heads)")
    print(f"  total runs   : {n_total}")
    print(f"  out_dir      : {out_dir}\n")

    partial_csv = out_dir / "sweep_partial.csv"

    if partial_csv.exists():
        existing  = pd.read_csv(partial_csv)
        rows      = existing.to_dict("records")
        done_keys = {(r["hidden_size"], r["seed"]) for r in rows}
        print(f"Resuming — {len(rows)} runs already done.\n")
    else:
        rows      = []
        done_keys = set()

    for pkl in out_dir.rglob("vis_data.pkl"):
        with open(pkl, "rb") as _f:
            _d = pickle.load(_f)
        key = (_d.get("hidden_size"), _d.get("seed"))
        if None in key or key in done_keys:
            continue
        done_keys.add(key)
        _psa = _d.get("psa_results", {0: {}}).get(0, {})
        rows.append(dict(
            hidden_size=key[0], seed=key[1],
            spearman_r=_d.get("spearman_r", np.nan),
            psa_score=_psa.get("psa_score",  np.nan),
            psa_delta=_psa.get("psa_delta",  np.nan),
            lick_high=_psa.get("high_lick",  np.nan),
            lick_mid=_psa.get("mid_lick",    np.nan),
            lick_low=_psa.get("low_lick",    np.nan),
            frac_selective=np.nan, frac_low=np.nan, frac_mid=np.nan,
            frac_high=np.nan, diverged=False,
        ))

    n_done = len(rows)
    for hidden_size in hidden_sizes:
        for run_idx in range(n_runs):
            seed = BASE_SEED + run_idx * SEED_STEP   # matches cogNN: seed = run_idx*7+42
            if (hidden_size, seed) in done_keys:
                print(f"[ skip ]  h={hidden_size:<4}  seed={seed}")
                continue

            n_done += 1
            print(f"[{n_done:>3}/{n_total}]  h={hidden_size:<4}  seed={seed}",
                  end="  ", flush=True)
            row = run_one(hidden_size, seed, device, out_dir)
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
    csv = out_dir / f"sweep_{ts}.csv"
    df.to_csv(csv, index=False)
    partial_csv.unlink(missing_ok=True)
    print(f"\nResults → {csv}")


if __name__ == "__main__":
    main()
