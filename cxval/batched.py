"""
Batched (parallel-episodes) training for the lick/no-lick value task.

This module is the batched counterpart to the online, single-stream training
loop used in the reward-param sweep scripts.  Instead of walking one agent
through a single very long trial sequence, it runs ``batch_size`` independent
episodes in lockstep:

    * Each episode is its own trial sequence (own stimulus order, own reward
      draws), generated with cxval.tasks exactly as before.
    * At every timestep the RNN processes all B episodes as the batch
      dimension; each episode keeps its own hidden state running *across trials*
      within the episode — identical dynamics to the online loop, just stacked.
    * Each A2C update averages the gradient over the B episodes.  That averaging
      is what tames the spectral-radius explosions that diverge ~70% of seeds in
      the single-stream loop.

Episodes can differ slightly in total length because the ITI is jittered, so
all episodes are right-padded to a common length T and an ``active`` mask
(B, T) marks the real timesteps.  Padded timesteps contribute zero reward and
are excluded from the loss.

Public API
----------
generate_batch          build (states, reward_availability, active mask, trial structs)
BatchedTaskEnv          vectorised env stepping B episodes in lockstep
train_batched           parallel-episodes A2C training loop with learning-curve logging
batched_inference       stochastic eval pass returning the analysis activations dict
build_trial_data        per-trial lick / value records from recorded action streams
compute_desiderata      the three target metrics (graded lick, activity scaling, selectivity scaling)

Author: patrick.mccarthy@dpag.ox.ac.uk
"""
from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from cxval.tasks import StimulusSequence, StateSequence
from cxval.models import RNN, ActorCritic
from cxval.analysis import (
    compute_unit_tuning,
    selectivity_index_range,
    preferred_stim_proportions,
    responsive_proportions_ttest,
    policy_similarity_analysis,
    stimulus_mean_activations,
)

LICK = 0
NO_LICK = 1


# =============================================================================
# TASK GENERATION  (B parallel episodes, padded + active mask)
# =============================================================================

def generate_batch(
    value_matrix,
    n_trials_per_episode,
    batch_size,
    base_seed,
    stim_timesteps=5,
    reward_timesteps=3,
    iti_timesteps=(3, 8),
    trials_per_phase=None,
    seed_stride=1,
):
    """Generate ``batch_size`` independent episodes and pad to a common length.

    Each episode uses a distinct seed (``base_seed + b * seed_stride``) so the
    stimulus orders and reward draws differ across the batch.  Within an episode
    the trial structure is exactly what StateSequence produces, so downstream
    analysis code that consumes ``trial_structure`` keeps working per episode.

    Args:
        value_matrix: (n_stim, n_ctx) reward-probability matrix.
        n_trials_per_episode: Trials in each episode (one phase, one context rep).
        batch_size: Number of parallel episodes B.
        base_seed: Seed for episode 0; episode b uses base_seed + b*seed_stride.
        stim_timesteps, reward_timesteps, iti_timesteps: Trial timing.
        trials_per_phase: Defaults to n_trials_per_episode (single phase / context rep).
        seed_stride: Gap between consecutive episode seeds.

    Returns:
        states_b: (B, T, D) float32 padded state arrays (D = n_ctx+n_stim+1).
        ravail_b: (B, T) float32 padded reward-availability.
        active_b: (B, T) float32 mask, 1 on real timesteps of each episode.
        trial_structs: list (len B) of per-episode trial_structure lists.
    """
    value_matrix = np.asarray(value_matrix, dtype=np.float32)
    if trials_per_phase is None:
        trials_per_phase = n_trials_per_episode

    states_list, ravail_list, structs = [], [], []
    for b in range(batch_size):
        seed = base_seed + b * seed_stride
        stim_seq = StimulusSequence(
            value_matrix=value_matrix,
            trials_per_phase=trials_per_phase,
            phases_per_context=1,
            context_order="sequential",
            context_reps=1,
        )
        stim_seq.generate(seed=seed)

        state_seq = StateSequence(
            stimulus_sequence=stim_seq,
            value_matrix=value_matrix,
            stim_timesteps=stim_timesteps,
            reward_timesteps=reward_timesteps,
            iti_timesteps=iti_timesteps,
        )
        states, _, ravail = state_seq.generate(seed=seed)
        states_list.append(states.astype(np.float32))
        ravail_list.append(ravail.astype(np.float32))
        structs.append(state_seq.trial_structure)

    T = max(s.shape[0] for s in states_list)
    D = states_list[0].shape[1]
    B = batch_size

    states_b = np.zeros((B, T, D), dtype=np.float32)
    ravail_b = np.zeros((B, T), dtype=np.float32)
    active_b = np.zeros((B, T), dtype=np.float32)
    for b in range(B):
        t_b = states_list[b].shape[0]
        states_b[b, :t_b] = states_list[b]
        ravail_b[b, :t_b] = ravail_list[b]
        active_b[b, :t_b] = 1.0

    return states_b, ravail_b, active_b, structs


# =============================================================================
# VECTORISED ENVIRONMENT
# =============================================================================

class BatchedTaskEnv:
    """Vectorised lick/no-lick env stepping B episodes in lockstep.

    Behaviour is identical, per episode, to cxval.envs.TaskEnv (see that class
    for the reward semantics).  All B episodes share a single timestep counter;
    padded timesteps (active mask == 0) yield zero reward and a zeroed obs and
    should be excluded from the loss via the returned ``active`` mask.
    """

    def __init__(
        self,
        states_b,
        ravail_b,
        active_b,
        reward_lick=1.0,
        reward_no_lick=0.0,
        reward_lick_fa=-1.0,
        lick_cost=0.0,
    ):
        self.states_b = np.asarray(states_b, dtype=np.float32)
        self.ravail_b = np.asarray(ravail_b, dtype=np.float32)
        self.active_b = np.asarray(active_b, dtype=np.float32)
        self.B, self.T, self.D = self.states_b.shape
        self.obs_dim = self.D + 2  # + [rewarded, unrewarded] feedback columns
        self.reward_lick = float(reward_lick)
        self.reward_no_lick = float(reward_no_lick)
        self.reward_lick_fa = float(reward_lick_fa)
        self.lick_cost = float(lick_cost)
        self._t = 0
        self._licked_window = np.zeros(self.B, dtype=bool)
        self._lick_rewarded = np.zeros(self.B, dtype=bool)

    def reset(self):
        """Reset to t=0. Returns obs (B, obs_dim)."""
        self._t = 0
        self._licked_window[:] = False
        self._lick_rewarded[:] = False
        return self._obs()

    def _obs(self):
        t = self._t
        base = self.states_b[:, t, :].copy()              # (B, D)
        feedback = np.zeros((self.B, 2), dtype=np.float32)
        in_window = base[:, -1] > 0
        show = in_window & self._licked_window
        base[show, -1] = 0.0                               # replace cue with outcome
        feedback[show & self._lick_rewarded, 0] = 1.0      # rewarded
        feedback[show & (~self._lick_rewarded), 1] = 1.0   # unrewarded
        return np.concatenate([base, feedback], axis=1)    # (B, obs_dim)

    def step(self, actions):
        """Advance one timestep for all episodes.

        Args:
            actions: (B,) int array, 0 = lick, 1 = no-lick.

        Returns:
            obs: (B, obs_dim) float32 (zeros if the whole batch is done).
            reward: (B,) float32, already zeroed on inactive/padded episodes.
            done: bool, True once the shared counter passes T.
            info: dict with 'active' (B,) mask and diagnostics.
        """
        actions = np.asarray(actions)
        t = self._t
        in_window = self.states_b[:, t, -1] > 0
        not_in = ~in_window
        # Clear lick tracking outside reward windows so each window gets one shot.
        self._licked_window[not_in] = False
        self._lick_rewarded[not_in] = False

        licked = actions == LICK
        reward_available = self.ravail_b[:, t] > 0
        first_lick = in_window & licked & (~self._licked_window)

        reward = np.full(self.B, self.reward_no_lick, dtype=np.float32)
        reward[first_lick & reward_available] = self.reward_lick
        reward[first_lick & (~reward_available)] = self.reward_lick_fa
        if self.lick_cost != 0.0:
            reward[licked] -= self.lick_cost

        self._licked_window |= first_lick
        self._lick_rewarded[first_lick] = reward_available[first_lick]

        active_t = self.active_b[:, t]
        reward = reward * active_t

        self._t += 1
        done = self._t >= self.T
        obs = (np.zeros((self.B, self.obs_dim), dtype=np.float32)
               if done else self._obs())

        info = {
            "active": active_t,
            "in_window": in_window,
            "licked": licked,
            "reward_available": reward_available,
        }
        return obs, reward, done, info


# =============================================================================
# A2C TRAINING  (parallel episodes)
# =============================================================================

def _compute_returns_batched(rew_buf, bootstrap, gamma):
    """Discounted returns over a rollout. rew_buf: list of (B,) tensors -> (K, B)."""
    returns = []
    R = bootstrap.clone()
    for r in reversed(rew_buf):
        R = r + gamma * R
        returns.append(R)
    returns.reverse()
    return torch.stack(returns)


def train_batched(
    value_matrix,
    *,
    batch_size=32,
    n_trials_per_episode=250,
    hidden_size=64,
    reward_lick_fa=0.0,
    lick_cost=0.0,
    base_seed=42,
    model_seed=None,
    device="cpu",
    lr=5e-4,
    gamma=0.9,
    value_coef=0.5,
    entropy_coef=0.0,
    policy_clip=0.1,
    readout_fraction=0.5,
    init_scale=0.1,
    recurrent_gain=0.9,
    h2h_init="orthogonal",
    input_plastic=True,
    hidden_plastic=True,
    output_plastic=True,
    grad_clip=1.0,
    bptt_len=40,
    reward_lick=1.0,
    reward_no_lick=0.0,
    stim_timesteps=5,
    reward_timesteps=3,
    iti_timesteps=(3, 8),
    checkpoint_dir=None,
    checkpoint_every=25,
    verbose=False,
):
    """Train an ActorCritic RNN with parallel-episodes A2C.

    ``bptt_len`` is both the rollout length and the truncated-BPTT window: the
    graph is cut and the optimiser steps every ``bptt_len`` timesteps, with the
    hidden state detached but carried forward.  It sets the *representation*
    horizon — how far gradients flow through the recurrent dynamics, i.e. the
    longest temporal dependency the hidden state can learn to bridge.  Default
    40 ≈ 3 average trials (avg trial = avg ITI 5.5 + stim 5 + reward 3 = 13.5).

    ``gamma`` sets the *reward-credit* horizon of the return (≈ 1/(1-gamma)
    steps).  When gamma > 0 the return at each rollout boundary is bootstrapped
    from the value head, so the credit horizon is decoupled from ``bptt_len``
    and the value head learns anticipatory value (nonzero, value-ordered during
    the stimulus epoch) rather than only immediate reward.

    Returns a dict with the trained model, learning-curve history, recorded
    per-episode action/value streams, and the per-episode trial structures.
    """
    if model_seed is None:
        model_seed = base_seed
    device = torch.device(device)
    value_matrix = np.asarray(value_matrix, dtype=np.float32)

    states_b, ravail_b, active_b, trial_structs = generate_batch(
        value_matrix, n_trials_per_episode, batch_size, base_seed,
        stim_timesteps=stim_timesteps, reward_timesteps=reward_timesteps,
        iti_timesteps=iti_timesteps,
    )
    B, T, D = states_b.shape
    obs_dim = D + 2

    env = BatchedTaskEnv(
        states_b, ravail_b, active_b,
        reward_lick=reward_lick, reward_no_lick=reward_no_lick,
        reward_lick_fa=reward_lick_fa, lick_cost=lick_cost,
    )

    torch.manual_seed(model_seed)
    backbone = RNN(input_size=obs_dim, hidden_size=hidden_size, output_size=1,
                   recurrent_gain=recurrent_gain, init_scale=init_scale,
                   h2h_init=h2h_init, input_plastic=input_plastic,
                   hidden_plastic=hidden_plastic, output_plastic=output_plastic)
    ac = ActorCritic(backbone=backbone, num_actions=2,
                     policy_clip=policy_clip,
                     readout_fraction=readout_fraction).to(device)
    init_state_dict = {k: v.cpu().clone() for k, v in ac.state_dict().items()}
    if checkpoint_dir is not None:
        from pathlib import Path as _Path
        checkpoint_dir = _Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(ac.parameters(), lr=lr)
    ac.train()

    action_arr = np.full((B, T), -1, dtype=np.int8)
    value_arr = np.zeros((B, T), dtype=np.float32)
    history = defaultdict(list)
    grad_norms = []
    explosion_resets = 0
    diverged = False

    obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=device)
    hidden = None
    lp_buf, val_buf, ent_buf, rew_buf, msk_buf = [], [], [], [], []
    t_win = 0
    t = 0

    done = False
    while not done:
        logits, value, hidden = ac.step(obs, hidden)        # (B,2),(B,),(B,H)
        dist = ac.make_dist(logits)
        action = dist.sample()                               # (B,)

        lp_buf.append(dist.log_prob(action))
        val_buf.append(value)
        ent_buf.append(dist.entropy())

        a_np = action.detach().cpu().numpy()
        action_arr[:, t] = a_np.astype(np.int8)
        value_arr[:, t] = value.detach().cpu().numpy()

        obs_np, reward, done, info = env.step(a_np)
        rew_buf.append(torch.as_tensor(reward, dtype=torch.float32, device=device))
        msk_buf.append(torch.as_tensor(info["active"], dtype=torch.float32, device=device))
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        t += 1
        t_win += 1

        if t_win % bptt_len == 0 or done:
            if gamma != 0.0 and not done:
                with torch.no_grad():
                    _, bv, _ = ac.step(obs, hidden)
            else:
                bv = torch.zeros(B, device=device)

            val_stack = torch.stack(val_buf)                 # (K, B)
            # Explosion guard: if forward dynamics blew up, reset rather than
            # backprop a NaN/inf gradient that would corrupt the weights.
            if (not torch.isfinite(val_stack).all()
                    or not torch.isfinite(hidden).all()):
                lp_buf, val_buf, ent_buf, rew_buf, msk_buf = [], [], [], [], []
                opt.zero_grad()
                hidden = torch.zeros_like(hidden)
                t_win = 0
                explosion_resets += 1
                continue

            rets = _compute_returns_batched(rew_buf, bv, gamma)   # (K, B)
            lp = torch.stack(lp_buf)                              # (K, B)
            ent = torch.stack(ent_buf)                           # (K, B)
            msk = torch.stack(msk_buf)                           # (K, B)
            rew_stack = torch.stack(rew_buf)                     # (K, B)
            denom = msk.sum().clamp(min=1.0)

            adv = rets - val_stack.detach()
            # Masked mean/std over active entries WITHOUT boolean indexing.
            # (adv[mask] / masked .std() triggers an MPSNDArray slice-assertion
            # abort on Apple Silicon; this arithmetic form is identical and
            # MPS-safe.)
            adv_mean = (adv * msk).sum() / denom
            adv_var = (((adv - adv_mean) ** 2) * msk).sum() / denom
            adv_std = torch.sqrt(adv_var.clamp(min=0.0) + 1e-12)
            if float(denom) > 1 and float(adv_std) > 1e-4:
                adv = (adv - adv_mean) / (adv_std + 1e-8)
            else:
                adv = adv - adv_mean

            policy_loss = -((lp * adv * msk).sum() / denom)
            value_loss = value_coef * ((msk * (val_stack - rets) ** 2).sum() / denom)
            entropy_term = entropy_coef * ((ent * msk).sum() / denom)
            loss = policy_loss + value_loss - entropy_term

            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(ac.parameters(), grad_clip)
            grad_norms.append(float(gnorm))
            opt.step()
            opt.zero_grad()
            hidden = hidden.detach()

            history["update"].append(len(grad_norms))
            history["timestep"].append(t)
            history["mean_reward"].append(float((rew_stack * msk).sum() / denom))
            history["entropy"].append(float((ent * msk).sum() / denom))
            history["value_loss"].append(float(value_loss.detach()))
            history["policy_loss"].append(float(policy_loss.detach()))
            history["grad_norm"].append(float(gnorm))

            if checkpoint_dir is not None and len(grad_norms) % checkpoint_every == 0:
                torch.save({k: v.cpu().clone() for k, v in ac.state_dict().items()},
                           checkpoint_dir / f"checkpoint_{len(grad_norms):05d}.pt")

            lp_buf, val_buf, ent_buf, rew_buf, msk_buf = [], [], [], [], []
            t_win = 0

            if any(torch.isnan(p).any() for p in ac.parameters()):
                try:
                    sr = float(torch.linalg.matrix_norm(
                        ac.backbone.h2h.weight.data, ord=2))
                except Exception:
                    sr = float("nan")
                warnings.warn(
                    f"NaN divergence: bs={B} lr={lr} fa={reward_lick_fa} "
                    f"cost={lick_cost} seed={base_seed} step={len(grad_norms)} "
                    f"W_rec spectral radius={sr:.3f} resets={explosion_resets}"
                )
                diverged = True
                break

    if verbose:
        tail = history["mean_reward"][-5:] if history["mean_reward"] else []
        print(f"  bs={B} lr={lr} fa={reward_lick_fa} cost={lick_cost} "
              f"seed={base_seed}: updates={len(grad_norms)} "
              f"resets={explosion_resets} diverged={diverged} "
              f"mean_reward_tail={np.mean(tail) if tail else float('nan'):.3f}")

    return dict(
        model=ac,
        init_state_dict=init_state_dict,
        history=dict(history),
        grad_norms=grad_norms,
        explosion_resets=explosion_resets,
        diverged=diverged,
        action_arr=action_arr,
        value_arr=value_arr,
        trial_structs=trial_structs,
        value_matrix=value_matrix,
        batch_size=B,
        n_trials_per_episode=n_trials_per_episode,
        obs_dim=obs_dim,
        hidden_size=hidden_size,
        env_kwargs=dict(
            reward_lick=reward_lick, reward_no_lick=reward_no_lick,
            reward_lick_fa=reward_lick_fa, lick_cost=lick_cost,
        ),
    )


# =============================================================================
# PER-TRIAL RECORDS  (for learning curves / lick-rate aggregation)
# =============================================================================

def build_trial_data(action_arr, value_arr, trial_structs):
    """Per-trial lick / value records across all episodes.

    Each record carries (episode, trial_in_episode) so learning curves can be
    plotted as lick-rate vs trial_in_episode averaged across the B episodes.
    """
    data = []
    for b, struct in enumerate(trial_structs):
        for ti, trial in enumerate(struct):
            rs, re = trial["reward_window"]
            acts = action_arr[b, rs:re]
            data.append({
                "episode": b,
                "trial_in_episode": ti,
                "context": trial["context"],
                "stimulus": trial["stimulus"],
                "reward_available": trial["reward_available"],
                "licked": int(acts[0] == LICK) if len(acts) else 0,
                "value_estimate": float(value_arr[b, rs:re].mean()) if re > rs else np.nan,
                "lick_count": int((acts == LICK).sum()),
            })
    return data


# =============================================================================
# INFERENCE  (stochastic eval -> analysis activations dict)
# =============================================================================

@torch.no_grad()
def batched_inference(
    model,
    value_matrix,
    *,
    n_eval_episodes=16,
    n_trials_per_episode=250,
    base_seed=10_000,
    device="cpu",
    reward_lick=1.0,
    reward_no_lick=0.0,
    reward_lick_fa=0.0,
    lick_cost=0.0,
    stim_timesteps=5,
    reward_timesteps=3,
    iti_timesteps=(3, 8),
    n_iti_pre=3,
    include_hidden_states=False,
):
    """Run a stochastic eval pass and assemble the activations dict.

    Returns (activations, lick_sc, infer_trial_data):
        activations: dict with stim_hidden/reward_hidden/context/stimulus/
            reward_available/trial_structure — the format cxval.analysis expects,
            with trials pooled across all eval episodes.
        lick_sc: (n_stim, n_ctx) mean lick probability per stimulus x context.
        infer_trial_data: per-trial records.
    """
    device = torch.device(device)
    value_matrix = np.asarray(value_matrix, dtype=np.float32)
    n_stim, n_ctx = value_matrix.shape

    states_b, ravail_b, active_b, structs = generate_batch(
        value_matrix, n_trials_per_episode, n_eval_episodes, base_seed,
        stim_timesteps=stim_timesteps, reward_timesteps=reward_timesteps,
        iti_timesteps=iti_timesteps,
    )
    env = BatchedTaskEnv(
        states_b, ravail_b, active_b,
        reward_lick=reward_lick, reward_no_lick=reward_no_lick,
        reward_lick_fa=reward_lick_fa, lick_cost=lick_cost,
    )
    B, T, D = states_b.shape

    model.eval()
    obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=device)
    hidden = None
    action_arr = np.full((B, T), -1, dtype=np.int8)
    value_arr = np.zeros((B, T), dtype=np.float32)
    hidden_arr = np.zeros((B, T, model.backbone.hidden_size), dtype=np.float32)

    done = False
    t = 0
    while not done:
        logits, value, hidden = model.step(obs, hidden)
        dist = model.make_dist(logits)
        action = dist.sample()
        a_np = action.detach().cpu().numpy()
        action_arr[:, t] = a_np.astype(np.int8)
        value_arr[:, t] = value.detach().cpu().numpy()
        hidden_arr[:, t] = hidden.detach().cpu().numpy()
        obs_np, _, done, _ = env.step(a_np)
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        t += 1

    # Pool trials across episodes into stacked stim/reward/baseline windows.
    stim_hidden, reward_hidden, baseline_hidden = [], [], []
    ctx_list, stim_list, ravail_list = [], [], []
    infer_trial_data = []
    H = model.backbone.hidden_size
    for b, struct in enumerate(structs):
        for ti, trial in enumerate(struct):
            ss, se = trial["stim_window"]
            rs, re = trial["reward_window"]
            stim_hidden.append(hidden_arr[b, ss:se])
            reward_hidden.append(hidden_arr[b, rs:re])
            # pre-stimulus baseline = the n_iti_pre timesteps just before stim
            bs0 = ss - n_iti_pre
            if bs0 >= 0:
                baseline_hidden.append(hidden_arr[b, bs0:ss])
            else:                                   # pad if ITI shorter than n_iti_pre
                pad = np.zeros((n_iti_pre, H), dtype=np.float32)
                pad[-ss:] = hidden_arr[b, 0:ss]
                baseline_hidden.append(pad)
            ctx_list.append(trial["context"])
            stim_list.append(trial["stimulus"])
            ravail_list.append(trial["reward_available"])
            acts = action_arr[b, rs:re]
            infer_trial_data.append({
                "episode": b,
                "context": trial["context"],
                "stimulus": trial["stimulus"],
                "reward_available": trial["reward_available"],
                "licked": int(acts[0] == LICK) if len(acts) else 0,
                "value_estimate": float(value_arr[b, rs:re].mean()) if re > rs else np.nan,
                "value_reward": float(value_arr[b, rs:re].mean()) if re > rs else np.nan,
                "value_stim": float(value_arr[b, ss:se].mean()) if se > ss else np.nan,
                "lick_count": int((acts == LICK).sum()),
            })

    activations = {
        "stim_hidden": np.stack(stim_hidden),
        "reward_hidden": np.stack(reward_hidden),
        "baseline_hidden": np.stack(baseline_hidden),
        "context": np.array(ctx_list),
        "stimulus": np.array(stim_list),
        "reward_available": np.array(ravail_list, dtype=bool),
        "trial_structure": [d for d in infer_trial_data],
    }
    if include_hidden_states:
        activations["hidden_states"] = hidden_arr        # full (B, T, H) for the vis notebook

    stim_arr = activations["stimulus"]
    ctx_arr = activations["context"]
    inf_lick = np.array([d["licked"] for d in infer_trial_data])
    lick_sc = np.full((n_stim, n_ctx), np.nan)
    for si in range(n_stim):
        for ci in range(n_ctx):
            m = (stim_arr == si) & (ctx_arr == ci)
            if m.sum():
                lick_sc[si, ci] = inf_lick[m].mean()

    return activations, lick_sc, infer_trial_data


# =============================================================================
# THE THREE DESIDERATA
# =============================================================================

def compute_desiderata(
    activations,
    lick_sc,
    value_matrix,
    *,
    si_threshold=0.1,
    value_threshold=0.25,
    silent_threshold=1e-4,
    period="stim",
    margin=0.0,
    ttest_alpha=0.05,
    ttest_n_sub=1000,   # fixed trials/stim, single deterministic t-test (n_rep=1)
    ttest_n_rep=1,
):
    """Evaluate the three target properties for one trained model.

    1. Graded licking:        lick_low  < lick_mid  < lick_high
    2. Activity scaling:      act_low   < act_mid   < act_high   (mean pop. activity)
    3. Selectivity scaling:   frac_low  < frac_mid  < frac_high  (of selective units)

    Selectivity is reported two ways: the winner-take-all preferred-stimulus
    proportions (``frac_*``), and — when ``activations`` carries a
    ``baseline_hidden`` window — the experiment-matched t-test responsive
    proportions (``frac_resp_*``, fraction of units significantly driven above
    pre-stimulus baseline; mixed selectivity allowed). The latter sets
    ``selectivity_ttest_ok`` and ``all_three_hold_ttest``.

    ``margin`` requires each inequality to hold by at least that amount, so
    near-ties don't count as satisfied.

    Returns a flat dict of metrics plus boolean *_ok flags and all_three_hold.
    """
    value_matrix = np.asarray(value_matrix, dtype=np.float32)

    # --- 1. graded licking -------------------------------------------------
    psa = policy_similarity_analysis(lick_sc, value_matrix)
    lick_low = float(psa[0]["low_lick"])
    lick_mid = float(psa[0]["mid_lick"])
    lick_high = float(psa[0]["high_lick"])
    graded_ok = (lick_low + margin < lick_mid) and (lick_mid + margin < lick_high)
    # where mid sits between low and high (0 = at low, 1 = at high)
    span = lick_high - lick_low
    mid_frac = float((lick_mid - lick_low) / span) if span > 1e-9 else np.nan

    # --- 2. population-activity scaling -----------------------------------
    mean_acts = stimulus_mean_activations(activations, period=period)
    pop = {si: float(np.nanmean(v)) for si, v in mean_acts.items()}
    n_stim = value_matrix.shape[0]
    act_low = pop.get(0, np.nan)
    act_mid = pop.get(n_stim // 2, np.nan)
    act_high = pop.get(n_stim - 1, np.nan)
    activity_scaling_ok = (act_low + margin < act_mid) and (act_mid + margin < act_high)

    # --- 3. selectivity-proportion scaling (cogNN convention; matches the
    #        vis_3s1c notebook: range-SI, si_threshold, silent mask) ---------
    tuning = compute_unit_tuning(activations, period=period)
    si_range = selectivity_index_range(tuning)
    props, _ = preferred_stim_proportions(
        tuning, si_threshold=si_threshold, si_values=si_range,
        silent_threshold=silent_threshold,
    )
    fps = props[0]["frac_per_stim"]
    n_stim_t = tuning.shape[1]
    frac_low = float(fps[0])
    frac_mid = float(fps[n_stim_t // 2])
    frac_high = float(fps[-1])
    n_selective = props[0]["n_selective"]
    selectivity_scaling_ok = (
        not any(np.isnan([frac_low, frac_mid, frac_high]))
        and (frac_low + margin < frac_mid)
        and (frac_mid + margin < frac_high)
    )

    all_three_hold = bool(graded_ok and activity_scaling_ok and selectivity_scaling_ok)

    # --- 3b. experiment-matched selectivity (t-test vs baseline) -----------
    frac_resp_low = frac_resp_mid = frac_resp_high = np.nan
    selectivity_ttest_ok = False
    all_three_hold_ttest = False
    if "baseline_hidden" in activations:
        rp = responsive_proportions_ttest(
            activations, period=period, alpha=ttest_alpha,
            n_sub=ttest_n_sub, n_rep=ttest_n_rep,
        )
        rps = rp["frac_per_stim"]
        frac_resp_low = float(rps[0])
        frac_resp_mid = float(rps[n_stim // 2])
        frac_resp_high = float(rps[-1])
        selectivity_ttest_ok = bool(
            (frac_resp_low + margin < frac_resp_mid)
            and (frac_resp_mid + margin < frac_resp_high)
        )
        all_three_hold_ttest = bool(
            graded_ok and activity_scaling_ok and selectivity_ttest_ok)

    return dict(
        lick_low=lick_low, lick_mid=lick_mid, lick_high=lick_high,
        mid_frac=mid_frac, graded_ok=bool(graded_ok),
        psa_score=float(psa[0]["psa_score"]), psa_delta=float(psa[0]["psa_delta"]),
        act_low=act_low, act_mid=act_mid, act_high=act_high,
        activity_scaling_ok=bool(activity_scaling_ok),
        frac_selective=float(n_selective / tuning.shape[0]),
        frac_low=float(frac_low), frac_mid=float(frac_mid), frac_high=float(frac_high),
        selectivity_scaling_ok=bool(selectivity_scaling_ok),
        all_three_hold=all_three_hold,
        frac_resp_low=frac_resp_low, frac_resp_mid=frac_resp_mid,
        frac_resp_high=frac_resp_high,
        selectivity_ttest_ok=selectivity_ttest_ok,
        all_three_hold_ttest=all_three_hold_ttest,
        psa_results=psa,
    )


# =============================================================================
# NOTEBOOK LOADER
# =============================================================================

def load_run(run_dir, device="cpu", n_eval_episodes=16, n_trials_per_episode=500):
    """Reconstruct a saved (discrete) run for the vis_3s1c notebook.

    The sweep CSVs don't store the big activation arrays (to save disk), so this
    rebuilds the ActorCritic from ``model.pt`` and re-runs inference, returning a
    dict that mirrors the saved ``vis_data`` plus a freshly-computed
    ``infer_activations`` (including the full ``hidden_states`` (B, T, H)),
    ``lick_sc`` and ``infer_trial_data``. Feed ``out['infer_activations']`` to the
    per-trial analysis functions (compute_unit_tuning, stimulus_mean_activations,
    responsive_proportions_ttest, the decoders, …).

    Args:
        run_dir: a run directory containing model.pt and vis_data.pkl.
        device, n_eval_episodes, n_trials_per_episode: inference settings.

    Returns:
        dict: the saved vis_data metadata, with 'infer_activations', 'lick_sc',
        'infer_trial_data' regenerated, and 'model' (the loaded ActorCritic).
    """
    import pickle
    from pathlib import Path
    run_dir = Path(run_dir)
    with open(run_dir / "vis_data.pkl", "rb") as f:
        vd = pickle.load(f)
    vm = np.asarray(vd["value_matrix"], dtype=np.float32)
    n_stim, n_ctx = vm.shape
    obs_dim = n_ctx + n_stim + 1 + 2                       # + reward cue + 2 feedback
    backbone = RNN(input_size=obs_dim, hidden_size=int(vd["hidden_size"]),
                   output_size=1, recurrent_gain=vd.get("recurrent_gain", 0.9))
    ac = ActorCritic(backbone=backbone, num_actions=2, policy_clip=0.1,
                     readout_fraction=float(vd.get("readout_fraction", 0.5))).to(device)
    ac.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    ac.eval()
    acts, lick_sc, itd = batched_inference(
        ac, vm, n_eval_episodes=n_eval_episodes, n_trials_per_episode=n_trials_per_episode,
        device=device, reward_lick_fa=float(vd.get("reward_lick_fa", 0.0)),
        lick_cost=float(vd.get("lick_cost", 0.0)), include_hidden_states=True)
    out = dict(vd)
    out["infer_activations"] = acts
    out["lick_sc"] = lick_sc
    out["infer_trial_data"] = itd
    out["model"] = ac
    return out
