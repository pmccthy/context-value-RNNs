"""
Context-switch vigour task: 2-context, N-stimulus (default 4) reward-swap task,
trained with a continuous lick-vigour actor-critic (cxval.vigour.VigourActorCritic)
on top of a rank-swept backbone (cxval.models.build_backbone).

This module mirrors cxval.vigour's train_vigour / infer_vigour but for a
MULTI-CONTEXT task with two training regimes:
    "block"       — contexts alternate in long runs (StimulusSequence,
                    context_order='sequential'), trials_per_block trials per
                    block, blocks_per_context blocks of EACH context.
    "interleaved" — the same total per-context trial budget, but shuffled at
                    the single-trial level (InterleavedStimulusSequence), so
                    context switches are unpredictable trial-to-trial.

Unlike train_vigour, this module RECORDS the full (B, T) vigour / value /
reward stream during training itself (same pattern cxval.batched.train_batched
already uses for action_arr/value_arr), so post-training you get genuine
trial-by-trial learning curves per (stimulus, context) condition straight from
the training rollout -- not a proxy from periodic re-evaluation.

Author: patrick.mccarthy@dpag.ox.ac.uk
"""
from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from cxval.tasks import StimulusSequence, StateSequence, InterleavedStimulusSequence
from cxval.models import build_backbone
from cxval.vigour import VigourActorCritic, BatchedVigourEnv
from cxval.analysis import classify_outcome


# =============================================================================
# TASK GENERATION  (block or interleaved, B parallel episodes, padded + active mask)
# =============================================================================

def generate_context_batch(
    value_matrix,
    mode,
    batch_size,
    base_seed,
    trials_per_block=None,
    blocks_per_context=None,
    trials_per_context=None,
    stim_timesteps=5,
    reward_timesteps=3,
    iti_timesteps=(3, 8),
    seed_stride=1,
    cue_context=True,
    context_order="sequential",
):
    """Generate ``batch_size`` independent multi-context episodes, padded to a
    common length T. Mirrors cxval.batched.generate_batch's contract exactly
    (same return signature), so it's a drop-in for the same downstream env /
    training-loop code.

    Args:
        value_matrix: (n_stim, n_ctx) reward-probability matrix.
        mode: "block" or "interleaved".
        batch_size: Number of parallel episodes B.
        base_seed: Seed for episode 0; episode b uses base_seed + b*seed_stride.
        trials_per_block, blocks_per_context: Required for mode="block". Total
            trials per episode = trials_per_block * blocks_per_context * n_ctx.
        trials_per_context: Required for mode="interleaved" (int or per-context
            list). Total trials per episode = sum(trials_per_context). Pass
            trials_per_block * blocks_per_context (same value used for the
            block condition) to match per-context trial budgets exactly.
        stim_timesteps, reward_timesteps, iti_timesteps: Trial timing.
        seed_stride: Gap between consecutive episode seeds.
        cue_context: if False, the context one-hot is omitted from the
            returned states (see cxval.tasks.StateSequence's
            include_context) -- pair with BatchedVigourEnv's
            feedback_action_reward=True for the uncued/meta-learning task
            variant, where context must be inferred from the agent's own
            action/reward history instead of being cued directly.
        context_order: "sequential" (default, exact old behaviour) or
            "random". Forwarded to StimulusSequence for mode="block" only
            (mode="interleaved" already reshuffles trial order and is
            unaffected). Added to resolve a position/schedule confound
            discovered in Experiment 14's independent decoding-probe
            analysis: with the default "sequential", every episode visits
            contexts in the exact same fixed order, so context is
            deterministically predictable from trial_in_episode ALONE, with
            no need to read any evidence -- a held-out-by-episode decode
            accuracy check can be trivially inflated by a decoder that has
            merely learned to track elapsed time/position (verified:
            position-only, zero-hidden-state linear decode = 0.667, MLP =
            0.997). Passing context_order="random" independently shuffles
            each episode's block order, closing that loophole for
            decode-accuracy evaluations. NOTE: block_schedule() assumes a
            single shared schedule (from episode 0) applies to every
            episode, which is only true for context_order="sequential" --
            do not use block_schedule()/trials_since_switch-style analyses
            together with context_order="random"; use "random" only for
            raw context-ID decode-accuracy checks.

    Returns:
        states_b, ravail_b, active_b, trial_structs — identical format to
        cxval.batched.generate_batch.
    """
    value_matrix = np.asarray(value_matrix, dtype=np.float32)
    n_ctx = value_matrix.shape[1]

    states_list, ravail_list, structs = [], [], []
    for b in range(batch_size):
        seed = base_seed + b * seed_stride

        if mode == "block":
            if trials_per_block is None or blocks_per_context is None:
                raise ValueError("mode='block' requires trials_per_block and blocks_per_context")
            stim_seq = StimulusSequence(
                value_matrix=value_matrix,
                trials_per_phase=trials_per_block,
                phases_per_context=1,
                context_order=context_order,
                context_reps=blocks_per_context,
            )
        elif mode == "interleaved":
            tpc = trials_per_context
            if tpc is None:
                if trials_per_block is None or blocks_per_context is None:
                    raise ValueError(
                        "mode='interleaved' requires trials_per_context, or "
                        "trials_per_block+blocks_per_context to derive it"
                    )
                tpc = trials_per_block * blocks_per_context
            stim_seq = InterleavedStimulusSequence(
                value_matrix=value_matrix, trials_per_context=tpc
            )
        else:
            raise ValueError(f"unknown mode {mode!r} (expected 'block' or 'interleaved')")

        stim_seq.generate(seed=seed)

        state_seq = StateSequence(
            stimulus_sequence=stim_seq,
            value_matrix=value_matrix,
            stim_timesteps=stim_timesteps,
            reward_timesteps=reward_timesteps,
            iti_timesteps=iti_timesteps,
            include_context=cue_context,
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


def block_schedule(trial_structs, episode=0):
    """Extract the deterministic context-block schedule from a "block"-mode run.

    Only meaningful for mode="block" (context_order='sequential' makes every
    episode's context sequence identical, so episode 0 suffices). Returns a
    list of dicts {context, start_trial, end_trial} in trial_in_episode units,
    for shading context blocks on a learning-curve plot.
    """
    struct = trial_structs[episode]
    blocks = []
    cur_ctx, start = None, 0
    for i, tr in enumerate(struct):
        if tr["context"] != cur_ctx:
            if cur_ctx is not None:
                blocks.append({"context": cur_ctx, "start_trial": start, "end_trial": i})
            cur_ctx, start = tr["context"], i
    blocks.append({"context": cur_ctx, "start_trial": start, "end_trial": len(struct)})
    return blocks


# =============================================================================
# PER-TRIAL RECORDS  (from a recorded (B, T) vigour/value/reward stream)
# =============================================================================

def build_context_trial_data(vig_arr, valest_arr, reward_arr, trial_structs):
    """Per-trial (stimulus, context) vigour / value / reward records.

    Args:
        vig_arr, valest_arr, reward_arr: (B, T) arrays recorded step-by-step
            (vigour actually used to step the env, critic value estimate,
            reward received), all indexed on the same clock as trial_structs.
        trial_structs: list (len B) of per-episode trial_structure lists (as
            returned by generate_context_batch).

    Returns:
        List of dicts, one per (episode, trial): episode, trial_in_episode,
        context, stimulus, reward_available, vigour (mean over reward window),
        value_estimate (mean over reward window), reward_consumed (mean reward
        over reward window), lick_count (vigour-analog: vigour * reward_timesteps).
    """
    data = []
    for b, struct in enumerate(trial_structs):
        for ti, trial in enumerate(struct):
            rs, re = trial["reward_window"]
            if re <= rs:
                continue
            data.append({
                "episode": b,
                "trial_in_episode": ti,
                "context": trial["context"],
                "stimulus": trial["stimulus"],
                "reward_available": trial["reward_available"],
                "vigour": float(vig_arr[b, rs:re].mean()),
                "value_estimate": float(valest_arr[b, rs:re].mean()),
                "reward_consumed": float(reward_arr[b, rs:re].mean()),
                "lick_count": float(vig_arr[b, rs:re].mean()) * (re - rs),
            })
    return data


# =============================================================================
# TRAINING  (continuous A2C, records the full training-time vigour/value/reward stream)
# =============================================================================

def _returns(rew_buf, bootstrap, gamma):
    out, R = [], bootstrap.clone()
    for r in reversed(rew_buf):
        R = r + gamma * R
        out.append(R)
    out.reverse()
    return torch.stack(out)


def train_context_vigour(
    value_matrix,
    mode,
    *,
    rank="full",
    batch_size=32,
    trials_per_block=400,
    blocks_per_context=6,
    trials_per_context=None,
    hidden_size=128,
    vigour_cost=0.8,
    cost_type="quadratic",
    reward_fa=0.0,
    reward_lick=1.0,
    base_seed=42,
    model_seed=None,
    device="cpu",
    lr=5e-4,
    gamma=0.9,
    value_coef=0.5,
    action_std=0.05,
    readout_fraction=0.5,
    init_scale=0.02,
    recurrent_gain=0.9,
    lowrank_gain=0.0,
    lowrank_scale=1.0,
    grad_clip=1.0,
    bptt_len=40,
    stim_timesteps=5,
    reward_timesteps=3,
    iti_timesteps=(3, 8),
    activity_coef=0.0,
    activity_at="iti",
    activity_floor_coef=0.0,
    aux_coef=0.0,
    aux_at="reward",
    detach_readout=False,
    init_model=None,
    checkpoint_dir=None,
    n_checkpoints=5,
    verbose=False,
    dynamics="elman",
    tau=None,
    dt=1.0,
    cue_context=True,
):
    """Train a VigourActorCritic on the context-switch task, recording the full
    training-time (B, T) vigour/value/reward stream for learning curves.

    ``rank``: passed straight to cxval.models.build_backbone (1, 2, 3, ... or
    "full" for a dense RNN). ``mode``: "block" or "interleaved" (see
    generate_context_batch). ``activity_coef`` defaults to 0 (off) -- the FENS
    final config used a firing-rate ITI penalty, but per-request this task
    does not need activation-suppression regularisation.

    ``activity_floor_coef`` (added 08_08_26, belief-state search Experiment
    12): defaults to 0 (off, exact old behaviour). The MIRROR IMAGE of
    ``activity_coef``: instead of penalising mean-squared activity (encouraging
    sparsity), it REWARDS it (encourages nonzero activity), using the same
    windowing (``activity_at``) and the same underlying quantity. Motivated by
    a direct finding in this search: combining ``detach_readout=True`` with
    ``aux_coef>0`` and no other force keeping the recurrence non-trivial can
    let the hidden state collapse to its trivial h=0 fixed point over enough
    training (confirmed via saved activations going to exactly 0 in one run)
    -- at h=0, a linear ``ctx_head`` degenerates to a constant, input-
    independent prediction of the class marginal, which is a very reachable,
    low (though not optimal) cross-entropy solution. A small positive
    ``activity_floor_coef`` directly discourages that trivial solution without
    touching ``activity_coef``'s existing (opposite-signed) sparsity role.

    ``cue_context``: if False, trains the UNCUED / meta-learning variant of
    this task -- the context one-hot is removed from the observation (see
    cxval.tasks.StateSequence.include_context) and the agent's own previous
    action and reward are fed back into its next observation instead (see
    cxval.vigour.BatchedVigourEnv.feedback_action_reward), so the model must
    infer the latent context purely from its own action/reward history
    (RL^2 / "learning to reinforcement learn" style, Duan et al. 2016,
    Wang et al. 2018). Everything else about the task/training loop is
    unchanged; only the observation construction differs. The ground-truth
    context is still recorded in trial_structs either way, for downstream
    decoding/analysis -- only the MODEL's input is affected by this flag.

    ``dynamics``/``tau``/``dt``: passed straight to build_backbone. IMPORTANT
    -- dynamics="mastrogiuseppe" only differs from "elman" if tau is given
    (and > dt); with tau=None (the default, memoryless map) the two are
    mathematically identical, see build_backbone's docstring.

    ``aux_coef``/``aux_at``: (added 08_08_26, belief-state search) if
    aux_coef > 0, adds an auxiliary CONTEXT-classification head
    (VigourActorCritic.ctx_head, a linear readout of the full hidden state)
    trained with cross-entropy against the ground-truth latent context,
    weighted by aux_coef and added to the actor-critic loss every update
    (same pattern as cxval.vigour.train_vigour's existing aux_coef/stim_head,
    which classifies stimulus identity instead). aux_at selects which part
    of each trial is supervised: "reward" (default, the reward window) or
    "stim" (the stimulus window) or "all" (the whole trial including ITI) or
    "iti" (added for Experiment 16 -- ONLY the pre-stimulus window, isolating
    the incentive to genuine cross-trial persistence rather than letting the
    aux loss piggyback on same-trial reactive information available at
    "stim"/"reward"/"all").
    Default aux_coef=0.0 (off) reproduces the exact old behaviour
    (ctx_head=None, no extra loss term, no extra head parameters).

    ``detach_readout``: (added 08_08_26, belief-state search) if True, RL
    (policy/value) gradients stop at the vigour/value readout and never
    reach the backbone -- see the inline comment near where it's applied,
    and cxval.vigour.VigourActorCritic.detach_readout's own docstring
    (pre-existing attribute; this kwarg just exposes it here). Useful with
    aux_coef > 0 to isolate whether the RL objective is competing with the
    auxiliary loss for the same hidden units. Default False = old behaviour.

    ``init_model``: (added 08_08_26, belief-state search) optional
    VigourActorCritic state_dict to warm-start ALL weights from (loaded with
    strict=False, so it tolerates e.g. warm-starting into a run with a new
    aux_coef>0 ctx_head that the source checkpoint doesn't have). None
    (default) = fresh random init, exact old behaviour. Same intent as
    cxval.vigour.train_vigour's existing init_model= kwarg.

    ``checkpoint_dir``: if given, saves the model state_dict at ``n_checkpoints``
    evenly-spaced points across training (as ``checkpoint_{update:05d}.pt``,
    matching the FENS-2026 checkpoint naming convention), IN ADDITION to
    ``init_state_dict`` (returned; the caller is responsible for saving it as
    model_init.pt) and the final trained state_dict (model.pt) -- so a run's
    full trajectory (init -> regular checkpoints -> final) is recoverable,
    e.g. to pick an earlier-training checkpoint as the "final" model for
    downstream evaluation instead of the fully-trained one.

    Returns a dict with the trained model, its initial state_dict, the update-
    level history (mean_reward, policy_loss, value_loss, grad_norm), the full
    per-trial training record (see build_context_trial_data), the trial
    structures, the block schedule (if mode="block"), a training-outcome
    classification (see cxval.analysis.classify_outcome -- distinguishes
    "converged" from "collapsed"/zero-vigour-stuck from "context_only", using
    the LAST 20% of training trials), and run metadata.
    """
    if model_seed is None:
        model_seed = base_seed
    device = torch.device(device)
    value_matrix = np.asarray(value_matrix, dtype=np.float32)
    n_stim, n_ctx = value_matrix.shape

    states_b, ravail_b, active_b, trial_structs = generate_context_batch(
        value_matrix, mode, batch_size, base_seed,
        trials_per_block=trials_per_block, blocks_per_context=blocks_per_context,
        trials_per_context=trials_per_context,
        stim_timesteps=stim_timesteps, reward_timesteps=reward_timesteps,
        iti_timesteps=iti_timesteps, cue_context=cue_context,
    )
    B, T, _ = states_b.shape
    env = BatchedVigourEnv(states_b, ravail_b, active_b, reward_lick=reward_lick,
                           reward_fa=reward_fa, vigour_cost=vigour_cost,
                           cost_type=cost_type, feedback_action_reward=not cue_context)
    D = env.obs_dim  # base state dim, +2 if cue_context=False (prev action, prev reward)

    need_iti_mask = (activity_coef > 0 or activity_floor_coef > 0) and activity_at == "iti"
    iti_mask_bt = np.zeros((B, T), dtype=np.float32)
    if need_iti_mask:
        for b, struct in enumerate(trial_structs):
            for tr in struct:
                lo, hi = tr["iti_window"]
                iti_mask_bt[b, lo:hi] = 1.0
    iti_mask_bt = torch.as_tensor(iti_mask_bt, device=device)

    # Per-timestep ground-truth context labels + supervision mask for the
    # optional auxiliary belief-prediction head (aux_coef > 0). Ground truth
    # is available here even though the model's OWN observation may not
    # include it (cue_context=False) -- see aux_coef's docstring above.
    ctx_id_bt = np.zeros((B, T), dtype=np.int64)
    aux_mask_bt = np.zeros((B, T), dtype=np.float32)
    if aux_coef > 0:
        for b, struct in enumerate(trial_structs):
            for tr in struct:
                if aux_at == "all":
                    lo, hi = tr["iti_window"][0], tr["reward_window"][1]
                elif aux_at == "reward":
                    lo, hi = tr["reward_window"]
                elif aux_at == "iti":
                    # (added 08_08_26, belief-state search Experiment 16)
                    # supervise ONLY the pre-stimulus ITI window -- unlike
                    # "all" (which also includes stim/reward and lets the
                    # aux loss piggyback on same-trial reactive information),
                    # this isolates the incentive to exactly the window
                    # where above-chance decodability would mean genuine
                    # cross-trial persistence (see Experiment 14's
                    # independent-probe finding that hidden_iti carries a
                    # small but real, schedule-robust signal even with NO
                    # aux loss at all -- this tests whether directly
                    # incentivizing that specific window can grow it).
                    lo, hi = tr["iti_window"]
                else:
                    lo, hi = tr["stim_window"]
                ctx_id_bt[b, lo:hi] = tr["context"]
                aux_mask_bt[b, lo:hi] = 1.0
    ctx_id_bt = torch.as_tensor(ctx_id_bt, device=device)
    aux_mask_bt_ctx = torch.as_tensor(aux_mask_bt, device=device)

    torch.manual_seed(model_seed)
    backbone = build_backbone(
        rank, input_size=D, hidden_size=hidden_size, output_size=1,
        recurrent_gain=recurrent_gain, init_scale=init_scale,
        lowrank_gain=lowrank_gain, lowrank_scale=lowrank_scale,
        dynamics=dynamics, tau=tau, dt=dt,
    )
    ac = VigourActorCritic(backbone, action_std=action_std,
                           readout_fraction=readout_fraction,
                           aux_n_ctx=(n_ctx if aux_coef > 0 else 0)).to(device)
    ac.detach_readout = detach_readout
    # detach_readout (added 08_08_26, belief-state search Experiment 8): if
    # True, stops policy/value (RL) gradients at the vigour/value readout so
    # they do NOT flow into the recurrent backbone (see
    # VigourActorCritic.detach_readout's docstring in cxval/vigour.py --
    # pre-existing attribute, just never wired to a train_context_vigour
    # kwarg before now). With aux_coef > 0, this isolates the backbone to
    # being shaped ONLY by the auxiliary context-classification loss (+ any
    # activity penalty), so the RL objective can't compete with/override it
    # for control of the same hidden units. Default False reproduces the
    # exact old behaviour (RL gradients always reach the backbone).
    if init_model is not None:
        # Warm-start the FULL actor-critic from a previous run's state_dict
        # (added 08_08_26, belief-state search, curriculum experiment -- see
        # results/08_08_26_belief_state_search/REPORT.md Experiment 7).
        # strict=False so warm-starting an aux_coef=0 model into an
        # aux_coef>0 continuation (new ctx_head, absent from the source
        # state_dict) doesn't error -- matches cxval.vigour.train_vigour's
        # existing init_model= pattern/intent, generalised for this case.
        missing, unexpected = ac.load_state_dict(init_model, strict=False)
        if verbose:
            print(f"  warm-started from init_model (missing={missing}, unexpected={unexpected})")
    ac.policy_mode = "score"
    init_state_dict = {k: v.cpu().clone() for k, v in ac.state_dict().items()}
    opt = torch.optim.Adam(ac.parameters(), lr=lr)
    ac.train()

    if checkpoint_dir is not None:
        from pathlib import Path as _Path
        checkpoint_dir = _Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # T (timesteps) is already known at this point, so estimate the total
        # number of updates up front and space checkpoints evenly across it.
        expected_updates = max(1, T // bptt_len)
        checkpoint_every = max(1, expected_updates // max(1, n_checkpoints))

    # Full training-time stream, recorded step-by-step (same pattern as
    # cxval.batched.train_batched's action_arr/value_arr).
    vig_arr = np.zeros((B, T), dtype=np.float32)
    valest_arr = np.zeros((B, T), dtype=np.float32)
    reward_arr = np.zeros((B, T), dtype=np.float32)

    history = defaultdict(list)
    grad_norms = []
    diverged = False
    obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=device)
    hidden = None
    lp_buf, val_buf, rew_buf, msk_buf = [], [], [], []
    act_buf, actmask_buf = [], []
    aux_buf = []  # (ctx_logits, label_t, mask_t) per timestep, only if ac.ctx_head is not None
    gt = 0
    t_win = 0
    done = False
    while not done:
        mean, value, hidden = ac.step(obs, hidden)
        if activity_coef > 0 or activity_floor_coef > 0:
            # penalise (activity_coef) or reward (activity_floor_coef) the
            # RATE (activity), not the raw pre-nonlinearity current --
            # identical to `hidden` for "elman" dynamics.
            act_buf.append(ac.backbone.activity(hidden))
            actmask_buf.append(iti_mask_bt[:, gt] if gt < T else torch.zeros(B, device=device))
        if ac.ctx_head is not None and gt < T:
            aux_buf.append((ac.ctx_head(hidden), ctx_id_bt[:, gt], aux_mask_bt_ctx[:, gt]))
        dist = ac.make_dist(mean)
        a = dist.sample()
        v = a.clamp(0.0, 1.0)
        lp_buf.append(dist.log_prob(a))
        gt_step = gt
        gt += 1
        val_buf.append(value)
        obs_np, reward, done, info = env.step(v.detach().cpu().numpy())

        if gt_step < T:
            vig_arr[:, gt_step] = v.detach().cpu().numpy()
            valest_arr[:, gt_step] = value.detach().cpu().numpy()
            reward_arr[:, gt_step] = reward

        rew_buf.append(torch.as_tensor(reward, dtype=torch.float32, device=device))
        msk_buf.append(torch.as_tensor(info["active"], dtype=torch.float32, device=device))
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        t_win += 1

        if t_win % bptt_len == 0 or done:
            if gamma != 0.0 and not done:
                with torch.no_grad():
                    _, bv, _ = ac.step(obs, hidden)
            else:
                bv = torch.zeros(B, device=device)
            val_stack = torch.stack(val_buf)
            if not torch.isfinite(val_stack).all() or not torch.isfinite(hidden).all():
                lp_buf, val_buf, rew_buf, msk_buf, act_buf, actmask_buf, aux_buf = \
                    [], [], [], [], [], [], []
                opt.zero_grad(); hidden = torch.zeros_like(hidden); t_win = 0
                continue
            rets = _returns(rew_buf, bv, gamma)
            lp = torch.stack(lp_buf); msk = torch.stack(msk_buf)
            denom = msk.sum().clamp(min=1.0)
            adv = rets - val_stack.detach()
            adv_mean = (adv * msk).sum() / denom
            adv_std = torch.sqrt((((adv - adv_mean) ** 2) * msk).sum() / denom + 1e-12)
            if float(denom) > 1 and float(adv_std) > 1e-4:
                adv = (adv - adv_mean) / (adv_std + 1e-8)
            else:
                adv = adv - adv_mean
            policy_loss = -((lp * adv * msk).sum() / denom)
            value_loss = value_coef * ((msk * (val_stack - rets) ** 2).sum() / denom)
            loss = policy_loss + value_loss
            if act_buf:
                act_stack = torch.stack(act_buf)
                amask = torch.stack(actmask_buf)
                act_denom = amask.sum().clamp(min=1.0)
                mean_sq_act = ((act_stack ** 2).mean(-1) * amask).sum() / act_denom
                if activity_coef > 0:
                    act_loss = activity_coef * mean_sq_act
                    loss = loss + act_loss
                    history["activity_loss"].append(float(act_loss.detach()))
                if activity_floor_coef > 0:
                    # negative sign: REWARD (not penalise) mean-squared
                    # activity, to counteract collapse to the trivial h=0
                    # fixed point -- see activity_floor_coef's docstring.
                    floor_loss = -activity_floor_coef * mean_sq_act
                    loss = loss + floor_loss
                    history["activity_floor_loss"].append(float(floor_loss.detach()))
            if ac.ctx_head is not None and aux_buf:
                al = torch.stack([x[0] for x in aux_buf])      # (Tw, B, n_ctx)
                lab = torch.stack([x[1] for x in aux_buf])     # (Tw, B)
                am = torch.stack([x[2] for x in aux_buf])      # (Tw, B)
                ce = nn.functional.cross_entropy(al.reshape(-1, n_ctx), lab.reshape(-1),
                                                 reduction="none")
                aux_loss = aux_coef * ((ce * am.reshape(-1)).sum() / am.sum().clamp(min=1.0))
                loss = loss + aux_loss
                history["aux_loss"].append(float(aux_loss.detach()))
            loss.backward()
            gn = nn.utils.clip_grad_norm_(ac.parameters(), grad_clip)
            grad_norms.append(float(gn))
            opt.step(); opt.zero_grad(); hidden = hidden.detach()

            history["update"].append(len(grad_norms))
            history["trial_progress"].append(gt)     # global timestep at this update
            history["mean_reward"].append(float((torch.stack(rew_buf) * msk).sum() / denom))
            history["policy_loss"].append(float(policy_loss.detach()))
            history["value_loss"].append(float(value_loss.detach()))
            history["grad_norm"].append(float(gn))

            if checkpoint_dir is not None and len(grad_norms) % checkpoint_every == 0:
                torch.save({k: v.cpu().clone() for k, v in ac.state_dict().items()},
                           checkpoint_dir / f"checkpoint_{len(grad_norms):05d}.pt")

            lp_buf, val_buf, rew_buf, msk_buf, act_buf, actmask_buf, aux_buf = \
                [], [], [], [], [], [], []
            t_win = 0
            if any(torch.isnan(p).any() for p in ac.parameters()):
                warnings.warn(f"context_vigour training diverged (NaN): mode={mode} rank={rank} seed={base_seed}")
                diverged = True
                break

    train_trial_data = build_context_trial_data(vig_arr, valest_arr, reward_arr, trial_structs)
    blocks = block_schedule(trial_structs, episode=0) if mode == "block" else None
    train_outcome = classify_outcome(
        {k: np.array([d[k] for d in train_trial_data]) for k in
         ("trial_in_episode", "stimulus", "context", "vigour")} if train_trial_data
        else dict(trial_in_episode=np.array([]), stimulus=np.array([]),
                  context=np.array([]), vigour=np.array([])),
        value_matrix, frac=0.2,
    )

    if verbose:
        tail = history["mean_reward"][-5:]
        print(f"  context_vigour mode={mode} rank={rank} seed={base_seed}: "
              f"updates={len(grad_norms)} diverged={diverged} "
              f"reward_tail={np.mean(tail) if tail else float('nan'):.3f} "
              f"train_outcome={train_outcome['outcome']}")

    return dict(
        model=ac, init_state_dict=init_state_dict, history=dict(history),
        grad_norms=grad_norms, diverged=diverged,
        train_trial_data=train_trial_data, trial_structs=trial_structs,
        block_schedule=blocks, value_matrix=value_matrix,
        hidden_size=hidden_size, obs_dim=D, rank=rank, mode=mode,
        n_stim=n_stim, n_ctx=n_ctx, train_outcome=train_outcome,
    )


# =============================================================================
# INFERENCE  (deterministic eval on a held-out dataset, same mode as training)
# =============================================================================

@torch.no_grad()
def infer_context_vigour(
    model,
    value_matrix,
    mode,
    *,
    n_eval_episodes=16,
    trials_per_block=400,
    blocks_per_context=2,
    trials_per_context=None,
    base_seed=100_000,
    device="cpu",
    vigour_cost=0.8,
    cost_type="quadratic",
    reward_fa=0.0,
    reward_lick=1.0,
    stim_timesteps=5,
    reward_timesteps=3,
    iti_timesteps=(3, 8),
    n_iti_pre=3,
    cue_context=True,
):
    """Deterministic eval pass on a HELD-OUT dataset (base_seed defaults far
    from any training seed range). Returns (activations, vmean, infer_trial_data,
    blocks, outcome) analogous to cxval.vigour.infer_vigour / load_vigour_run,
    but with n_ctx contexts, vmean keyed by (stimulus, context), and outcome a
    training-outcome classification (see cxval.analysis.classify_outcome) computed
    over the WHOLE inference set (frac=None -- this is already a deterministic,
    held-out pass, so there's no "still learning" transient to exclude the way
    there is for a training rollout).

    ``cue_context``: MUST match whatever the model was trained with (see
    train_context_vigour's cue_context) -- controls both whether the context
    one-hot is generated and whether the env feeds back
    action/reward, so the observation dimensionality here matches what the
    model actually expects.
    """
    device = torch.device(device)
    value_matrix = np.asarray(value_matrix, dtype=np.float32)
    n_stim, n_ctx = value_matrix.shape

    states_b, ravail_b, active_b, structs = generate_context_batch(
        value_matrix, mode, n_eval_episodes, base_seed,
        trials_per_block=trials_per_block, blocks_per_context=blocks_per_context,
        trials_per_context=trials_per_context,
        stim_timesteps=stim_timesteps, reward_timesteps=reward_timesteps,
        iti_timesteps=iti_timesteps, cue_context=cue_context,
    )
    env = BatchedVigourEnv(states_b, ravail_b, active_b, reward_lick=reward_lick,
                           reward_fa=reward_fa, vigour_cost=vigour_cost,
                           cost_type=cost_type, feedback_action_reward=not cue_context)
    B, T, _ = states_b.shape
    H = model.backbone.hidden_size
    model.eval()
    obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=device)
    hidden = None
    vig = np.zeros((B, T), np.float32)
    val = np.zeros((B, T), np.float32)
    rew = np.zeros((B, T), np.float32)
    hid = np.zeros((B, T, H), np.float32)
    done = False; t = 0
    while not done:
        mean, value, hidden = model.step(obs, hidden)
        v = model.squash(mean)
        vig[:, t] = v.cpu().numpy()
        val[:, t] = value.cpu().numpy()
        # record ACTIVITY (rate), not the raw state -- identical to `hidden`
        # for the default "elman" dynamics, but for "mastrogiuseppe" dynamics
        # `hidden` is the raw pre-nonlinearity current and downstream
        # decoding/geometry analysis should see the rate instead.
        hid[:, t] = model.backbone.activity(hidden).cpu().numpy()
        obs_np, reward, done, _ = env.step(v.cpu().numpy())
        rew[:, t] = reward
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        t += 1

    infer_trial_data = build_context_trial_data(vig, val, rew, structs)

    stim_hidden, baseline_hidden, ctx_list, stim_list, ravail_list = [], [], [], [], []
    vmean_sum = np.zeros((n_stim, n_ctx), np.float64)
    vmean_cnt = np.zeros((n_stim, n_ctx), np.int64)
    for b, struct in enumerate(structs):
        for tr in struct:
            ss, se = tr["stim_window"]; rs, re = tr["reward_window"]
            s, c = tr["stimulus"], tr["context"]
            stim_hidden.append(hid[b, ss:se])
            bs0 = ss - n_iti_pre
            baseline_hidden.append(hid[b, bs0:ss] if bs0 >= 0
                                   else np.zeros((n_iti_pre, H), np.float32))
            ctx_list.append(c); stim_list.append(s); ravail_list.append(tr["reward_available"])
            if re > rs:
                vmean_sum[s, c] += float(vig[b, rs:re].mean())
                vmean_cnt[s, c] += 1

    vmean = np.divide(vmean_sum, np.maximum(vmean_cnt, 1))
    vmean[vmean_cnt == 0] = np.nan

    activations = dict(
        stim_hidden=np.stack(stim_hidden), baseline_hidden=np.stack(baseline_hidden),
        context=np.array(ctx_list), stimulus=np.array(stim_list),
        reward_available=np.array(ravail_list, dtype=bool),
    )
    blocks = block_schedule(structs, episode=0) if mode == "block" else None
    outcome = classify_outcome(
        {k: np.array([d[k] for d in infer_trial_data]) for k in
         ("trial_in_episode", "stimulus", "context", "vigour")} if infer_trial_data
        else dict(trial_in_episode=np.array([]), stimulus=np.array([]),
                  context=np.array([]), vigour=np.array([])),
        value_matrix, frac=None,
    )
    return activations, vmean, infer_trial_data, blocks, outcome


# =============================================================================
# ACTIVATION EXTRACTION  (rich per-trial, per-epoch hidden state for decoding /
# population-geometry analysis -- see scripts/csha_ccnss_2026/16_07_26_extract_activations.py)
# =============================================================================

@torch.no_grad()
def extract_activations(
    model,
    value_matrix,
    mode,
    *,
    n_eval_episodes=16,
    trials_per_block=400,
    blocks_per_context=2,
    trials_per_context=None,
    base_seed=200_000,
    device="cpu",
    vigour_cost=0.8,
    cost_type="quadratic",
    reward_fa=0.0,
    reward_lick=1.0,
    stim_timesteps=5,
    reward_timesteps=3,
    iti_timesteps=(3, 8),
    n_iti_pre=3,
    cue_context=True,
    context_order="sequential",
):
    """Deterministic held-out inference pass that captures, per trial, the
    full hidden-state trajectory sliced into three FIXED-LENGTH epochs so
    trials stack cleanly into (n_trials, n_timesteps, hidden_size) tensors
    ready for decoding / condition-averaged population-geometry analysis
    (Bernardi et al. 2020-style, or cross-condition generalisation decoders):

        hidden_iti     (n_trials, n_iti_pre, H)        -- pre-stimulus baseline
        hidden_stim    (n_trials, stim_timesteps, H)   -- stimulus period
        hidden_reward  (n_trials, reward_timesteps, H) -- outcome/reward period

    This is deliberately separate from infer_context_vigour (which only
    keeps a coarse stim_hidden/baseline_hidden slice sized for the vmean
    summary table saved alongside every training run) -- this function is
    for a dedicated, possibly much larger, held-out dataset built purely for
    downstream decoding, using base_seed=200_000 by default so it doesn't
    overlap the training seed range (base_seed) or infer_context_vigour's
    own held-out range (100_000).

    ``cue_context``: MUST match whatever the model was trained with (see
    train_context_vigour's cue_context) -- see infer_context_vigour's
    matching docstring note.

    ``context_order``: "sequential" (default, exact old behaviour) or
    "random" -- forwarded to generate_context_batch (see its docstring for
    the full rationale: closes a position/schedule-leakage confound in
    held-out decode-accuracy checks). IMPORTANT: this function's returned
    `block_schedule` is only meaningful for context_order="sequential" (it
    is derived from episode 0 alone and assumed to apply to every episode,
    per block_schedule()'s own docstring) -- do not use the returned
    `block_schedule` / any trials_since_switch-style analysis on activations
    extracted with context_order="random"; that setting is for raw
    context-ID decode-accuracy checks only.

    Returns a dict of columnar per-trial arrays (episode, trial_in_episode,
    context, stimulus, reward_available, true_value, vigour, value_estimate,
    reward_consumed, hidden_iti, hidden_stim, hidden_reward), plus `outcome`
    (classify_outcome over the whole set), `block_schedule`, and shape info
    (n_stim, n_ctx, hidden_size).
    """
    device = torch.device(device)
    value_matrix = np.asarray(value_matrix, dtype=np.float32)
    n_stim, n_ctx = value_matrix.shape

    states_b, ravail_b, active_b, structs = generate_context_batch(
        value_matrix, mode, n_eval_episodes, base_seed,
        trials_per_block=trials_per_block, blocks_per_context=blocks_per_context,
        trials_per_context=trials_per_context,
        stim_timesteps=stim_timesteps, reward_timesteps=reward_timesteps,
        iti_timesteps=iti_timesteps, cue_context=cue_context,
        context_order=context_order,
    )
    env = BatchedVigourEnv(states_b, ravail_b, active_b, reward_lick=reward_lick,
                           reward_fa=reward_fa, vigour_cost=vigour_cost,
                           cost_type=cost_type, feedback_action_reward=not cue_context)
    B, T, _ = states_b.shape
    H = model.backbone.hidden_size
    model.eval()
    obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=device)
    hidden = None
    vig = np.zeros((B, T), np.float32)
    val = np.zeros((B, T), np.float32)
    rew = np.zeros((B, T), np.float32)
    hid = np.zeros((B, T, H), np.float32)
    done = False; t = 0
    while not done:
        mean, value, hidden = model.step(obs, hidden)
        v = model.squash(mean)
        vig[:, t] = v.cpu().numpy()
        val[:, t] = value.cpu().numpy()
        # see infer_context_vigour above -- record activity/rate, not raw state
        hid[:, t] = model.backbone.activity(hidden).cpu().numpy()
        obs_np, reward, done, _ = env.step(v.cpu().numpy())
        rew[:, t] = reward
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        t += 1

    trial_data = build_context_trial_data(vig, val, rew, structs)
    trial_lookup = {(d["episode"], d["trial_in_episode"]): d for d in trial_data}

    episode_l, trial_l, ctx_l, stim_l, ravail_l, trueval_l = [], [], [], [], [], []
    vig_l, valest_l, rewc_l = [], [], []
    iti_h, stim_h, rew_h = [], [], []
    # per-TIMESTEP vigour/reward (not just the trial-averaged scalar above) --
    # lets downstream analysis see WITHIN-trial adaptation (e.g. a switch's
    # surprise arriving mid-reward-window and being corrected for within the
    # same trial), which the trial-averaged "vigour" field alone can mask.
    vig_iti_l, vig_stim_l, vig_rew_l = [], [], []
    rew_iti_l, rew_stim_l, rew_rew_l = [], [], []
    for b, struct in enumerate(structs):
        for ti, tr in enumerate(struct):
            ss, se = tr["stim_window"]; rs, re = tr["reward_window"]
            if re <= rs:
                continue  # no reward window recorded (matches build_context_trial_data)
            s, c = tr["stimulus"], tr["context"]
            bs0 = ss - n_iti_pre
            in_bounds = bs0 >= 0
            iti_h.append(hid[b, bs0:ss] if in_bounds
                         else np.full((n_iti_pre, H), np.nan, np.float32))
            stim_h.append(hid[b, ss:se])
            rew_h.append(hid[b, rs:re])
            vig_iti_l.append(vig[b, bs0:ss] if in_bounds
                             else np.full(n_iti_pre, np.nan, np.float32))
            vig_stim_l.append(vig[b, ss:se])
            vig_rew_l.append(vig[b, rs:re])
            rew_iti_l.append(rew[b, bs0:ss] if in_bounds
                             else np.full(n_iti_pre, np.nan, np.float32))
            rew_stim_l.append(rew[b, ss:se])
            rew_rew_l.append(rew[b, rs:re])
            episode_l.append(b); trial_l.append(ti); ctx_l.append(c); stim_l.append(s)
            ravail_l.append(tr["reward_available"]); trueval_l.append(float(value_matrix[s, c]))
            d = trial_lookup.get((b, ti), {})
            vig_l.append(d.get("vigour", np.nan)); valest_l.append(d.get("value_estimate", np.nan))
            rewc_l.append(d.get("reward_consumed", np.nan))

    blocks = block_schedule(structs, episode=0) if mode == "block" else None
    outcome = classify_outcome(
        dict(trial_in_episode=np.array(trial_l), stimulus=np.array(stim_l),
             context=np.array(ctx_l), vigour=np.array(vig_l)),
        value_matrix, frac=None,
    )
    return dict(
        episode=np.array(episode_l), trial_in_episode=np.array(trial_l),
        context=np.array(ctx_l), stimulus=np.array(stim_l),
        reward_available=np.array(ravail_l, dtype=bool),
        true_value=np.array(trueval_l, dtype=np.float32),
        vigour=np.array(vig_l, dtype=np.float32),
        value_estimate=np.array(valest_l, dtype=np.float32),
        reward_consumed=np.array(rewc_l, dtype=np.float32),
        hidden_iti=np.stack(iti_h).astype(np.float32),
        hidden_stim=np.stack(stim_h).astype(np.float32),
        hidden_reward=np.stack(rew_h).astype(np.float32),
        vigour_iti=np.stack(vig_iti_l).astype(np.float32),
        vigour_stim=np.stack(vig_stim_l).astype(np.float32),
        vigour_reward=np.stack(vig_rew_l).astype(np.float32),
        reward_iti=np.stack(rew_iti_l).astype(np.float32),
        reward_stim=np.stack(rew_stim_l).astype(np.float32),
        reward_reward=np.stack(rew_rew_l).astype(np.float32),
        outcome=outcome, block_schedule=blocks,
        n_stim=n_stim, n_ctx=n_ctx, hidden_size=H,
    )
