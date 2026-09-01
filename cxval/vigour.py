"""
Continuous lick-VIGOUR actor-critic — an alternative to the discrete lick/no-lick
policy.

Motivation
----------
With a binary action and a linearly-rewarded decision the reward-maximising
policy is bang-bang (lick fully whenever expected value > 0), so the 50%
stimulus saturates near the 100% one. If instead the agent emits a continuous
vigour ``v in [0, 1]`` each timestep and licking carries a CONVEX (quadratic)
effort cost, the optimum becomes interior:

    reward_t = v * base_t - 0.5 * cost * v**2     ->     v* = E[base] / cost

i.e. optimal vigour is LINEAR in expected value (the response-vigour account of
Niv et al. 2007; Shadmehr et al.). Since ``base`` in the reward window is
+reward_lick when reward is available (prob = stimulus value) and reward_fa
otherwise, E[base] = value, so v* = value / cost: the lick rate becomes a linear
readout of the value-scaled population, the 50% stimulus lands in the middle, and
NO false-alarm penalty is needed (so the 0%-detector recruitment that broke
selectivity does not occur). Vigour does not change the observation, so v* is
independent of gamma.

Public API mirrors cxval.batched: VigourActorCritic, BatchedVigourEnv,
train_vigour, infer_vigour, vigour_metrics.

Author: patrick.mccarthy@dpag.ox.ac.uk
"""
from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from cxval.models import RNN
from cxval.batched import generate_batch
from cxval.analysis import (
    compute_unit_tuning, stimulus_mean_activations, responsive_proportions_ttest,
)


# =============================================================================
# MODEL
# =============================================================================

class VigourActorCritic(nn.Module):
    """Actor-critic with a continuous vigour head (Gaussian policy on [0,1])."""

    def __init__(self, backbone: RNN, action_std=0.1, readout_fraction=0.5,
                 demean_readout=False, readout_mode=None, aux_n_stim=0, aux_n_ctx=0):
        super().__init__()
        self.backbone = backbone
        self.n_readout = max(1, int(backbone.hidden_size * readout_fraction))
        self.vigour_head = nn.Linear(self.n_readout, 1)
        self.value_head = nn.Linear(self.n_readout, 1)
        self.action_std = action_std
        # Optional auxiliary "which stimulus did I just see?" classifier off the
        # FULL hidden state. Training it (at a delay, after the stimulus has left
        # the input) pressures the population to maintain stimulus IDENTITY, which
        # creates identity-exclusive cells rather than a pure value-magnitude code.
        self.aux_n_stim = aux_n_stim
        self.stim_head = nn.Linear(backbone.hidden_size, aux_n_stim) if aux_n_stim else None
        # Optional auxiliary "which LATENT CONTEXT am I in right now?" classifier
        # off the full hidden state (added 08_08_26, belief-state search --
        # see results/08_08_26_belief_state_search/REPORT.md Experiment 5).
        # Unlike stim_head (stimulus is directly observed every trial), context
        # is NOT observed in the uncued task -- the only way to get a low loss
        # here is to integrate evidence (the fed-back prev-action/prev-reward
        # history) over multiple trials, so training this head is a direct,
        # explicit incentive for exactly the belief-state representation this
        # search is trying to elicit. aux_n_ctx=0 (default) leaves stim_head
        # untouched and reproduces the exact old behaviour (ctx_head=None).
        self.aux_n_ctx = aux_n_ctx
        self.ctx_head = nn.Linear(backbone.hidden_size, aux_n_ctx) if aux_n_ctx else None
        # How vigour reads the population — three coding hypotheses to compare:
        #   "linear" : raw readout (default gain/rate code; vigour = w·h)
        #   "demean" : readout with the mean (overall-level) direction removed, so
        #              vigour is invariant to uniform changes in population activity
        #   "total"  : vigour reads ONLY the mean activity (overall level literally
        #              controls vigour) — the opposite pole
        self.readout_mode = readout_mode or ("demean" if demean_readout else "linear")
        self.demean_readout = (self.readout_mode == "demean")   # back-compat attr
        self.policy_mode = "score"          # "pathwise" squashes vigour via sigmoid
        # When True, stop the policy/value (RL) gradients at the readout so they do
        # NOT flow into the recurrent backbone. The hidden units are then shaped only
        # by the auxiliary classification loss (+ any activity penalty); the vigour
        # and value heads become linear readouts trained OFF (but not INTO) that
        # representation. This is the "classification + RL, readout-only RL" model.
        self.detach_readout = False
        # Lower bound on executed vigour (score/Gaussian policy only). Default 0.0
        # reproduces the original behaviour. With action_std as small as it is here
        # (0.05), a hard clamp at exactly 0 is a real local-minimum trap for the
        # score-function (REINFORCE) gradient: once the readout mean drifts negative
        # enough that ~all samples clip to 0, the executed action stops varying with
        # the mean, so d(action)/d(mean) effectively vanishes and there is no signal
        # to climb back out — consistent with seeds that get permanently stuck at
        # vigour=0 after the reversal. Raising min_vigour above 0 keeps the executed
        # action (and its reward/cost feedback) non-degenerate even when the mean has
        # collapsed, giving the policy a path back out.
        self.min_vigour = 0.0
        # Width of a SMOOTH ("soft-clamp") squash for the score policy. None (default)
        # reproduces the original hard clamp exactly. When set, replaces the hard clamp
        # with a logistic sigmoid -- v = sigmoid(a/squash_width - 3) -- applied to the
        # SAMPLED action `a` (same place the hard clamp was applied), so the
        # score-function/REINFORCE objective (advantage, critic, TD-bootstrapping --
        # everything believed to shape value-scaled population activity) is completely
        # untouched; only how a raw sample maps to an executed vigour changes.
        #
        # Motivation: a hard clamp has EXACTLY zero gradient once |a| drifts far enough
        # past [min_vigour,1] that ~all samples clip to the same executed value --
        # reward stops covarying with which sample was drawn, so REINFORCE's TRUE
        # EXPECTED gradient (not just an unlucky estimate) is zero, and there is no
        # signal anywhere to climb back out (see min_vigour's docstring above for the
        # same failure mode this also targets, from a different angle). A logistic
        # sigmoid never goes fully flat, so reward always covaries at least weakly with
        # the sample, keeping REINFORCE's expectation nonzero no matter how far `a` has
        # drifted.
        #
        # The "-3" offset is deliberate, not cosmetic: an EARLIER version centred the
        # sigmoid on [0,1] directly (v=sigmoid((a-0.5)/width)), which gives v(a=0)~0.42
        # -- i.e. an untrained network (raw output ~0 at init) already executes ~42%
        # vigour by default. The original hard clamp gave v(0)=0 exactly, a free
        # "do nothing" default; the miscentred version instead pays the quadratic
        # vigour cost on EVERY trial from the very first update, which measurably
        # destabilised training (a smoke-test seed's reward went negative). Offsetting
        # by -3 restores v(a=0)=sigmoid(-3)=0.047 regardless of width, matching the
        # clamp's benign default while keeping the sigmoid's never-fully-flat gradient.
        #
        # squash_width still controls a genuine, unavoidable trade-off (a sigmoid's
        # gradient decays ~exponentially away from its transition zone, so you can't
        # have both "linear across [0,1]" AND "non-negligible gradient several units
        # past it"). width=1.3 keeps grad >= ~0.03-0.1 out to a=6-8 (where a
        # pre-reversal-100% stimulus's readout typically sits, and now needs to fall
        # back toward suppression after reversal) while v(0)=0.047 keeps the
        # to-be-suppressed stimulus's default cheap, same as the original clamp.
        self.squash_width = None

    def squash(self, mean):
        """Map the head output to a vigour in [min_vigour,1]: hard clamp for the score
        (Gaussian) policy (or a smooth soft-clamp if squash_width is set -- see its
        docstring above), smooth sigmoid for the pathwise (deterministic) policy
        (rescaled into [min_vigour,1] too, so the floor applies identically either way)."""
        if self.policy_mode == "pathwise":
            v = torch.sigmoid(mean)
            return self.min_vigour + (1.0 - self.min_vigour) * v if self.min_vigour else v
        if self.squash_width:
            v = torch.sigmoid(mean / self.squash_width - 3.0)
            return self.min_vigour + (1.0 - self.min_vigour) * v
        return mean.clamp(self.min_vigour, 1.0)

    def _read(self, ro):
        if self.readout_mode == "demean":
            return ro - ro.mean(dim=-1, keepdim=True)
        if self.readout_mode == "total":
            return ro.mean(dim=-1, keepdim=True).expand_as(ro)
        return ro

    def step(self, obs, hidden=None):
        if hidden is None:
            hidden = self.backbone.init_hidden(obs.shape[0], obs.device)
        hidden = self.backbone.recurrence(obs, hidden)
        # backbone.activity(hidden) is a no-op (identity) for the default
        # "elman" dynamics, but for "mastrogiuseppe" dynamics `hidden` is the
        # raw pre-nonlinearity current -- readout must use the RATE
        # phi(hidden), never the raw current. `hidden` itself (returned
        # below, unchanged) stays the raw state, since that's what the NEXT
        # call's backbone.recurrence() expects as h_prev.
        ro = self.backbone.activity(hidden)[..., :self.n_readout]
        if self.detach_readout:                 # RL grads stop here (don't enter backbone)
            ro = ro.detach()
        mean = self.vigour_head(self._read(ro)).squeeze(-1)
        value = self.value_head(ro).squeeze(-1)
        return mean, value, hidden

    def make_dist(self, mean):
        return Normal(mean, self.action_std)


# =============================================================================
# ENVIRONMENT
# =============================================================================

class BatchedVigourEnv:
    """Vectorised continuous-vigour env.

    Action ``v`` (B,) in [0,1] per timestep. Observation is the base state
    (context | stimulus | reward-window cue) with NO feedback columns by
    default. Reward at t:  v * base - 0.5 * cost * v**2, where base is
    reward_lick if (in reward window and reward available), reward_fa if (in
    window, no reward), 0 otherwise. Vigour does not alter the next
    observation UNLESS feedback_action_reward=True.

    feedback_action_reward: if True, appends 2 extra trailing columns to
        every observation -- [previous action v_{t-1}, previous reward
        r_{t-1}] (zeros at reset, since there's no previous step yet) --
        turning obs_dim into D+2. This is the standard meta-RL / RL^2
        (Duan et al. 2016, Wang et al. 2018) trick: since the agent's own
        action and the reward it received are the ONLY signal available to
        it about what just happened, feeding them back lets a recurrent
        policy learn to integrate them over time and implicitly infer any
        latent variable (e.g. context) that isn't otherwise observed --
        see cxval.tasks.StateSequence's include_context=False, which is
        the intended pairing: turn OFF the explicit context cue and turn
        ON this feedback so context becomes something the model has to
        infer from experience rather than being told directly.
    """

    def __init__(self, states_b, ravail_b, active_b,
                 reward_lick=1.0, reward_fa=0.0, vigour_cost=1.1,
                 cost_type="quadratic", feedback_action_reward=False):
        self.states_b = np.asarray(states_b, dtype=np.float32)
        self.ravail_b = np.asarray(ravail_b, dtype=np.float32)
        self.active_b = np.asarray(active_b, dtype=np.float32)
        self.B, self.T, self.D = self.states_b.shape
        self.feedback_action_reward = feedback_action_reward
        self.obs_dim = self.D + (2 if feedback_action_reward else 0)
        self.reward_lick = float(reward_lick)
        self.reward_fa = float(reward_fa)
        self.cost = float(vigour_cost)
        self.cost_type = cost_type           # "quadratic" (graded) or "linear" (bang-bang)
        self._t = 0

    def _augment(self, base_obs, prev_action, prev_reward):
        if not self.feedback_action_reward:
            return base_obs
        return np.concatenate(
            [base_obs, prev_action[:, None].astype(np.float32),
            prev_reward[:, None].astype(np.float32)], axis=-1)

    def reset(self):
        self._t = 0
        base = self.states_b[:, 0, :].copy()
        zeros = np.zeros(self.B, dtype=np.float32)
        return self._augment(base, zeros, zeros)

    def step(self, v):
        v = np.asarray(v, dtype=np.float32)
        t = self._t
        in_win = self.states_b[:, t, -1] > 0
        avail = self.ravail_b[:, t] > 0
        base = np.where(in_win, np.where(avail, self.reward_lick, self.reward_fa),
                        0.0).astype(np.float32)
        effort = (0.5 * self.cost * v * v) if self.cost_type == "quadratic" else (self.cost * v)
        reward = (v * base - effort) * self.active_b[:, t]
        reward = reward.astype(np.float32)
        self._t += 1
        done = self._t >= self.T
        base_obs = (np.zeros((self.B, self.D), dtype=np.float32)
                   if done else self.states_b[:, self._t, :].copy())
        obs = self._augment(base_obs, v, reward)
        return obs, reward, done, {"active": self.active_b[:, t]}


# =============================================================================
# TRAINING  (continuous A2C; parallels train_batched)
# =============================================================================

def _returns(rew_buf, bootstrap, gamma):
    out, R = [], bootstrap.clone()
    for r in reversed(rew_buf):
        R = r + gamma * R
        out.append(R)
    out.reverse()
    return torch.stack(out)


def pretrain_backbone(value_matrix, *, hidden_size=128, init_scale=0.05, recurrent_gain=0.9,
                      batch_size=32, n_trials_per_episode=1500, n_epochs=6, base_seed=0,
                      lr=1e-3, weight_decay=0.0, bptt_len=40, label_window="stim",
                      model_seed=0, device="cpu", rescale_spectral_radius=1.0,
                      stim_timesteps=5, reward_timesteps=3, iti_timesteps=(3, 8)):
    """Supervised pretraining: train an RNN backbone (same architecture the vigour
    model uses) to classify which of the n_stim stimuli is present, from its hidden
    state. With balanced stimuli this yields ~equally sized groups of stimulus-
    selective units. Returns (backbone_state_dict, info). Feed the state_dict to
    ``train_vigour(init_backbone=...)`` to warm-start RL from this representation.

    label_window: "stim" supervises during the stimulus (stim-selective cells) or
    "reward" during the (post-stimulus) reward window (maintained-identity cells).
    """
    device = torch.device(device)
    value_matrix = np.asarray(value_matrix, dtype=np.float32)
    n_stim = value_matrix.shape[0]
    torch.manual_seed(model_seed)

    sb, _, _, structs = generate_batch(
        value_matrix, n_trials_per_episode, batch_size, base_seed,
        stim_timesteps=stim_timesteps, reward_timesteps=reward_timesteps,
        iti_timesteps=iti_timesteps)
    B, T, D = sb.shape
    backbone = RNN(input_size=D, hidden_size=hidden_size, output_size=1,
                   recurrent_gain=recurrent_gain, init_scale=init_scale).to(device)
    clf = nn.Linear(hidden_size, n_stim).to(device)
    opt = torch.optim.Adam(list(backbone.parameters()) + list(clf.parameters()),
                           lr=lr, weight_decay=weight_decay)

    lab_bt = np.zeros((B, T), np.int64); m_bt = np.zeros((B, T), np.float32)
    for b, struct in enumerate(structs):
        for tr in struct:
            lo, hi = tr["stim_window"] if label_window == "stim" else tr["reward_window"]
            lab_bt[b, lo:hi] = tr["stimulus"]; m_bt[b, lo:hi] = 1.0
    states = torch.as_tensor(sb, dtype=torch.float32, device=device)
    lab = torch.as_tensor(lab_bt, device=device); msk = torch.as_tensor(m_bt, device=device)

    acc = np.nan
    for ep in range(n_epochs):
        hidden = backbone.init_hidden(B, device); opt.zero_grad()
        lbuf, ybuf, mbuf = [], [], []
        correct = tot = 0.0
        for t in range(T):
            hidden = backbone.recurrence(states[:, t, :], hidden)
            lbuf.append(clf(hidden)); ybuf.append(lab[:, t]); mbuf.append(msk[:, t])
            if (t + 1) % bptt_len == 0 or t == T - 1:
                al = torch.stack(lbuf); yy = torch.stack(ybuf); mm = torch.stack(mbuf)
                ce = nn.functional.cross_entropy(al.reshape(-1, n_stim), yy.reshape(-1),
                                                 reduction="none")
                loss = (ce * mm.reshape(-1)).sum() / mm.sum().clamp(min=1.0)
                loss.backward(); opt.step(); opt.zero_grad(); hidden = hidden.detach()
                pred = al.reshape(-1, n_stim).argmax(-1)
                correct += float(((pred == yy.reshape(-1)).float() * mm.reshape(-1)).sum())
                tot += float(mm.sum())
                lbuf, ybuf, mbuf = [], [], []
        acc = correct / max(tot, 1.0)

    # Pretraining can inflate the recurrent gain (→ explosive dynamics that
    # destabilise the subsequent policy-gradient RL). Rescale the recurrent matrix
    # to a target spectral radius, preserving the learned identity STRUCTURE (a
    # uniform scale doesn't change which units encode which stimulus) while taming
    # the gain.
    sr = np.nan
    if rescale_spectral_radius is not None:
        W = backbone.h2h.weight.detach().cpu().numpy()
        sr = float(np.max(np.abs(np.linalg.eigvals(W))))
        if sr > 1e-6:
            with torch.no_grad():
                backbone.h2h.weight.mul_(rescale_spectral_radius / sr)
    return {k: v.cpu().clone() for k, v in backbone.state_dict().items()}, \
        {"final_acc": acc, "spectral_radius_pre": sr}


def train_vigour(value_matrix, *, batch_size=32, n_trials_per_episode=1500,
                 hidden_size=128, vigour_cost=1.1, cost_type="quadratic",
                 reward_fa=0.0, base_seed=42,
                 model_seed=None, device="cpu", lr=5e-4, gamma=0.9, value_coef=0.5,
                 action_std=0.1, readout_fraction=0.5, init_scale=0.05,
                 recurrent_gain=0.9, grad_clip=1.0, bptt_len=40, reward_lick=1.0,
                 stim_timesteps=5, reward_timesteps=3, iti_timesteps=(3, 8),
                 demean_readout=False, readout_mode=None,
                 aux_coef=0.0, aux_at="reward", activity_coef=0.0, activity_at="all",
                 nonneg_coef=0.0, detach_readout=False, min_vigour=0.0, squash_width=None,
                 init_backbone=None, init_model=None, lr_warmup=0,
                 policy="score", probe_every=0, probe_fn=None,
                 reward_scale_by_stim=None, track_gradients=False,
                 checkpoint_dir=None, checkpoint_every=25, verbose=False):
    if model_seed is None:
        model_seed = base_seed
    device = torch.device(device)
    value_matrix = np.asarray(value_matrix, dtype=np.float32)
    n_stim = value_matrix.shape[0]

    states_b, ravail_b, active_b, trial_structs = generate_batch(
        value_matrix, n_trials_per_episode, batch_size, base_seed,
        stim_timesteps=stim_timesteps, reward_timesteps=reward_timesteps,
        iti_timesteps=iti_timesteps)
    B, T, D = states_b.shape
    env = BatchedVigourEnv(states_b, ravail_b, active_b, reward_lick=reward_lick,
                           reward_fa=reward_fa, vigour_cost=vigour_cost,
                           cost_type=cost_type)

    # Per-timestep stimulus-identity labels + supervision mask for the auxiliary
    # head. Supervise during the chosen window ("reward" = after the stimulus has
    # left the input, forcing maintained identity; "stim" = during the stimulus).
    stim_id_bt = np.zeros((B, T), dtype=np.int64)
    aux_mask_bt = np.zeros((B, T), dtype=np.float32)
    if aux_coef > 0:
        for b, struct in enumerate(trial_structs):
            for tr in struct:
                lo, hi = tr["reward_window"] if aux_at == "reward" else tr["stim_window"]
                stim_id_bt[b, lo:hi] = tr["stimulus"]
                aux_mask_bt[b, lo:hi] = 1.0
    stim_id_bt = torch.as_tensor(stim_id_bt, device=device)
    aux_mask_bt = torch.as_tensor(aux_mask_bt, device=device)

    # Per-timestep stimulus label restricted to the REWARD window specifically (decoupled
    # from the aux head's own aux_at choice, and built regardless of aux_coef) -- used by
    # (a) reward_scale_by_stim, which multiplies the realized reward r_t at exactly the
    # instants reward could be delivered for a given stimulus (leaving the vigour-cost
    # structure at every OTHER timestep, for every stimulus, completely untouched), and
    # (b) track_gradients' per-stimulus analytic policy-gradient probe below. Outside the
    # reward window this defaults to stimulus 0, which is harmless for (a) since reward is
    # exactly 0 there regardless, and is masked out by `msk` for (b).
    need_reward_stim_labels = track_gradients or (reward_scale_by_stim is not None)
    stim_id_reward_bt = None
    if need_reward_stim_labels:
        _sid = np.zeros((B, T), dtype=np.int64)
        for b, struct in enumerate(trial_structs):
            for trstruct in struct:
                rs, re = trstruct["reward_window"]
                _sid[b, rs:re] = trstruct["stimulus"]
        stim_id_reward_bt = torch.as_tensor(_sid, device=device)
    if reward_scale_by_stim is not None:
        reward_scale_arr = np.asarray(reward_scale_by_stim, dtype=np.float32)
        assert reward_scale_arr.shape == (n_stim,), \
            f"reward_scale_by_stim must have {n_stim} entries, got {reward_scale_arr.shape}"

    # Optional ITI-window mask. Used by (a) the targeted firing-rate penalty
    # (activity_at="iti"), which pushes the pre-stimulus baseline toward the ReLU
    # floor, and (b) the anti-suppression penalty (nonneg_coef), which discourages
    # evoked activity from dropping *below* each unit's own ITI baseline so that
    # non-preferred responses sit at-or-above baseline rather than dipping under it.
    need_iti_mask = (activity_coef > 0 and activity_at == "iti") or nonneg_coef > 0
    iti_mask_bt = np.zeros((B, T), dtype=np.float32)
    if need_iti_mask:
        for b, struct in enumerate(trial_structs):
            for tr in struct:
                lo, hi = tr["iti_window"]
                iti_mask_bt[b, lo:hi] = 1.0
    iti_mask_bt = torch.as_tensor(iti_mask_bt, device=device)
    need_act_buf = activity_coef > 0 or nonneg_coef > 0

    torch.manual_seed(model_seed)
    backbone = RNN(input_size=D, hidden_size=hidden_size, output_size=1,
                   recurrent_gain=recurrent_gain, init_scale=init_scale)
    if init_backbone is not None:                 # warm-start from a pretrained backbone
        backbone.load_state_dict(init_backbone)
    ac = VigourActorCritic(backbone, action_std=action_std,
                           readout_fraction=readout_fraction,
                           demean_readout=demean_readout,
                           readout_mode=readout_mode,
                           aux_n_stim=(n_stim if aux_coef > 0 else 0)).to(device)
    ac.policy_mode = policy   # "pathwise" squashes vigour with sigmoid (smooth gradient)
    ac.detach_readout = detach_readout   # readout-only RL: stop RL grads at the readout
    ac.min_vigour = min_vigour           # floor on executed vigour (0.0 = old behaviour)
    ac.squash_width = squash_width       # smooth soft-clamp width (None = old hard clamp)
    if init_model is not None:            # warm-start the FULL actor-critic (e.g. reversal)
        ac.load_state_dict(init_model)
    init_state_dict = {k: v.cpu().clone() for k, v in ac.state_dict().items()}
    opt = torch.optim.Adam(ac.parameters(), lr=lr)
    ac.train()
    if checkpoint_dir is not None:
        from pathlib import Path as _Path
        checkpoint_dir = _Path(checkpoint_dir); checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # For the pathwise (deterministic) actor we differentiate the reward
    # r = v*base - effort(v) directly, so precompute the per-timestep reward base
    # (a constant w.r.t. v) and the active mask.
    if policy == "pathwise":
        _inwin = states_b[:, :, -1] > 0
        _avail = ravail_b > 0
        base_bt = np.where(_inwin, np.where(_avail, reward_lick, reward_fa), 0.0).astype(np.float32)
        base_bt = torch.as_tensor(base_bt, device=device)
        active_bt = torch.as_tensor(active_b.astype(np.float32), device=device)

    history = defaultdict(list)
    grad_norms = []
    diverged = False
    obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=device)
    hidden = None
    lp_buf, val_buf, rew_buf, msk_buf = [], [], [], []
    pw_buf = []                                 # differentiable per-timestep reward (pathwise)
    aux_buf = []                                # (logits, label_t, mask_t) per timestep
    act_buf = []                                # hidden states (for firing-rate penalty)
    actmask_buf = []                            # per-step weight for the activity penalty
    a_buf, mean_buf, stimid_buf = [], [], []    # raw sample/mean/stim-label (track_gradients only)
    gt = 0
    t_win = 0
    done = False
    while not done:
        t_idx = gt                              # this step's index into the *_bt label arrays
                                                 # (captured before gt is incremented below, same
                                                 # convention the aux_buf line already relied on)
        mean, value, hidden = ac.step(obs, hidden)
        if need_act_buf:
            act_buf.append(hidden)
            if need_iti_mask:
                actmask_buf.append(iti_mask_bt[:, t_idx] if t_idx < T
                                   else torch.zeros(B, device=device))
        if ac.stim_head is not None and t_idx < T:
            aux_buf.append((ac.stim_head(hidden), stim_id_bt[:, t_idx], aux_mask_bt[:, t_idx]))
        if policy == "pathwise":
            v = torch.sigmoid(mean)                   # smooth squash → gradient always flows
            b_t = base_bt[:, t_idx] if t_idx < T else torch.zeros_like(mean)
            a_t = active_bt[:, t_idx] if t_idx < T else torch.zeros_like(mean)
            eff = 0.5 * vigour_cost * v * v if cost_type == "quadratic" else vigour_cost * v
            pw_buf.append((v * b_t - eff) * a_t)      # differentiable immediate reward
            lp_buf.append(torch.zeros_like(mean))     # placeholder (unused for pathwise)
        else:
            dist = ac.make_dist(mean)
            a = dist.sample()
            v = ac.squash(a)          # clamp to [min_vigour, 1] (was hardcoded [0, 1] here,
                                       # bypassing squash()/min_vigour entirely — fixed)
            lp_buf.append(dist.log_prob(a))
            if track_gradients:
                a_buf.append(a.detach()); mean_buf.append(mean.detach())
        gt += 1
        val_buf.append(value)
        obs_np, reward, done, info = env.step(v.detach().cpu().numpy())
        if reward_scale_by_stim is not None and t_idx < T:
            # Scale the REALIZED reward (payoff net of vigour cost) exactly at the
            # instants this stimulus's reward could be delivered -- leaves every other
            # timestep, for every stimulus, byte-for-byte identical to an unscaled run.
            scale_t = reward_scale_arr[stim_id_reward_bt[:, t_idx].cpu().numpy()]
            reward = reward * scale_t
        rew_buf.append(torch.as_tensor(reward, dtype=torch.float32, device=device))
        msk_buf.append(torch.as_tensor(info["active"], dtype=torch.float32, device=device))
        if track_gradients:
            stimid_buf.append(stim_id_reward_bt[:, t_idx] if t_idx < T
                              else torch.zeros(B, dtype=torch.long, device=device))
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
                lp_buf, val_buf, rew_buf, msk_buf, aux_buf, pw_buf, act_buf, actmask_buf = \
                    [], [], [], [], [], [], [], []
                a_buf, mean_buf, stimid_buf = [], [], []
                opt.zero_grad(); hidden = torch.zeros_like(hidden); t_win = 0
                continue
            rets = _returns(rew_buf, bv, gamma)
            lp = torch.stack(lp_buf); msk = torch.stack(msk_buf)
            rew_stack = torch.stack(rew_buf)
            denom = msk.sum().clamp(min=1.0)
            adv = rets - val_stack.detach()
            adv_mean = (adv * msk).sum() / denom
            adv_std = torch.sqrt((((adv - adv_mean) ** 2) * msk).sum() / denom + 1e-12)
            if float(denom) > 1 and float(adv_std) > 1e-4:
                adv = (adv - adv_mean) / (adv_std + 1e-8)
            else:
                adv = adv - adv_mean
            if track_gradients and policy != "pathwise" and a_buf:
                # Analytic per-timestep REINFORCE score at the output layer, BEFORE it is
                # backprop'd through the network: d/d(mean_t) [-log N(a_t; mean_t, sigma) *
                # A_t] = -(a_t - mean_t)/sigma^2 * A_t (the log-derivative-trick gradient,
                # exact -- not a proxy). Binned by which stimulus's reward window each
                # timestep falls in, this is the cheapest possible answer to "how much raw
                # policy-gradient signal is stimulus S actually contributing this update",
                # with NO extra backward() pass (everything here is already-buffered
                # forward-pass output plus the advantage computed above).
                a_stack = torch.stack(a_buf); mean_stack = torch.stack(mean_buf)
                stim_stack = torch.stack(stimid_buf)
                score = (a_stack - mean_stack) / (action_std ** 2) * adv     # (Tw, B)
                per_stim = []
                for s in range(n_stim):
                    sel = (stim_stack == s) & (msk > 0)
                    per_stim.append(float(score[sel].abs().mean()) if sel.any() else float("nan"))
                history["policy_grad_by_stim"].append(per_stim)
            if policy == "pathwise":
                # maximise the (differentiable) immediate reward directly
                policy_loss = -((torch.stack(pw_buf) * msk).sum() / denom)
            else:
                policy_loss = -((lp * adv * msk).sum() / denom)
            value_loss = value_coef * ((msk * (val_stack - rets) ** 2).sum() / denom)
            loss = policy_loss + value_loss
            if ac.stim_head is not None and aux_buf:
                al = torch.stack([x[0] for x in aux_buf])      # (Tw, B, n_stim)
                lab = torch.stack([x[1] for x in aux_buf])     # (Tw, B)
                am = torch.stack([x[2] for x in aux_buf])      # (Tw, B)
                ce = nn.functional.cross_entropy(al.reshape(-1, n_stim), lab.reshape(-1),
                                                 reduction="none")
                aux_loss = aux_coef * ((ce * am.reshape(-1)).sum() / am.sum().clamp(min=1.0))
                loss = loss + aux_loss
                history["aux_loss"].append(float(aux_loss.detach()))
            if act_buf:
                act_stack = torch.stack(act_buf)               # (Tw, B, H)
                iti_w = torch.stack(actmask_buf) if need_iti_mask else None
                if activity_coef > 0:
                    # L2 firing-rate (metabolic) penalty: pull mean squared
                    # activity down so the baseline drops and responses stay
                    # upward (ReLU floor) rather than dipping below a high tonic.
                    # activity_at="iti" targets only the pre-stimulus window.
                    amask = iti_w if activity_at == "iti" else msk
                    act_denom = amask.sum().clamp(min=1.0)
                    act_loss = activity_coef * (((act_stack ** 2).mean(-1) * amask).sum()
                                                / act_denom)
                    loss = loss + act_loss
                    history["activity_loss"].append(float(act_loss.detach()))
                if nonneg_coef > 0:
                    # Anti-suppression: hinge penalty on evoked (stim/outcome)
                    # activity falling below each unit's own (detached) ITI
                    # baseline. NOTE: the target is self-referential (the baseline
                    # rises with overall activity), so large coefficients can cause
                    # runaway activity inflation; pushing the baseline to the ReLU
                    # floor via activity_at="iti" is the more robust route.
                    denom_iti = iti_w.sum(0).clamp(min=1.0)            # (B,)
                    base_bh = ((act_stack * iti_w.unsqueeze(-1)).sum(0)
                               / denom_iti.unsqueeze(-1)).detach()     # (B, H)
                    evoked = msk * (1.0 - iti_w)                       # (Tw, B)
                    below = torch.relu(base_bh.unsqueeze(0) - act_stack)
                    nn_denom = evoked.sum().clamp(min=1.0)
                    nn_loss = nonneg_coef * ((below.mean(-1) * evoked).sum() / nn_denom)
                    loss = loss + nn_loss
                    history["nonneg_loss"].append(float(nn_loss.detach()))
            loss.backward()
            if track_gradients:
                # Parameter-GROUP gradient norms, taken PRE-clip (i.e. before grad_clip
                # rescales everything down to the same global norm, which would erase
                # exactly the differences we're trying to see) -- how much of this
                # update's gradient is actually reaching each part of the network.
                def _gnorm(params):
                    gs = [p.grad.detach() for p in params if p.grad is not None]
                    return float(torch.sqrt(sum((g ** 2).sum() for g in gs))) if gs else float("nan")
                history["grad_norm_backbone"].append(_gnorm(ac.backbone.parameters()))
                history["grad_norm_vigour_head"].append(_gnorm(ac.vigour_head.parameters()))
                history["grad_norm_value_head"].append(_gnorm(ac.value_head.parameters()))
                if ac.stim_head is not None:
                    history["grad_norm_stim_head"].append(_gnorm(ac.stim_head.parameters()))
            gn = nn.utils.clip_grad_norm_(ac.parameters(), grad_clip)
            grad_norms.append(float(gn))
            if lr_warmup > 0:                          # linear LR warmup over first updates
                f = min(1.0, len(grad_norms) / float(lr_warmup))
                for g in opt.param_groups:
                    g["lr"] = lr * f
            opt.step(); opt.zero_grad(); hidden = hidden.detach()
            history["update"].append(len(grad_norms))
            history["mean_reward"].append(float((rew_stack * msk).sum() / denom))
            history["grad_norm"].append(float(gn))
            if track_gradients:                  # scalar loss VALUES, for comparing against
                history["policy_loss"].append(float(policy_loss.detach()))   # the grad norms
                history["value_loss"].append(float(value_loss.detach()))     # above
            # periodic deterministic probe: log a caller-supplied metrics dict vs update
            # (used to trace per-stimulus vigour/activity/selectivity over training).
            if probe_every and probe_fn is not None and len(grad_norms) % probe_every == 0:
                ac.eval()
                with torch.no_grad():
                    pv = probe_fn(ac)
                ac.train()
                history["probe_update"].append(len(grad_norms))
                for _k, _v in pv.items():
                    history["probe_" + _k].append(_v)
            if checkpoint_dir is not None and len(grad_norms) % checkpoint_every == 0:
                torch.save({k: v.cpu().clone() for k, v in ac.state_dict().items()},
                           checkpoint_dir / f"checkpoint_{len(grad_norms):05d}.pt")
            lp_buf, val_buf, rew_buf, msk_buf, aux_buf, pw_buf, act_buf, actmask_buf = \
                [], [], [], [], [], [], [], []
            a_buf, mean_buf, stimid_buf = [], [], []
            t_win = 0
            if any(torch.isnan(p).any() for p in ac.parameters()):
                warnings.warn("vigour training diverged (NaN)")
                diverged = True
                break

    if verbose:
        tail = history["mean_reward"][-5:]
        print(f"  vigour seed={base_seed}: updates={len(grad_norms)} "
              f"diverged={diverged} reward_tail={np.mean(tail) if tail else float('nan'):.3f}")
    return dict(model=ac, init_state_dict=init_state_dict, history=dict(history),
                grad_norms=grad_norms, diverged=diverged, trial_structs=trial_structs,
                value_matrix=value_matrix, hidden_size=hidden_size, obs_dim=D)


# =============================================================================
# INFERENCE + METRICS
# =============================================================================

@torch.no_grad()
def infer_vigour(model, value_matrix, *, n_eval_episodes=12, n_trials_per_episode=400,
                 base_seed=10_000, device="cpu", vigour_cost=1.1, cost_type="quadratic",
                 reward_fa=0.0, stim_timesteps=5, reward_timesteps=3, iti_timesteps=(3, 8),
                 n_iti_pre=3):
    """Deterministic eval: mean vigour per stimulus + activations (with baseline)."""
    device = torch.device(device)
    value_matrix = np.asarray(value_matrix, dtype=np.float32)
    states_b, ravail_b, active_b, structs = generate_batch(
        value_matrix, n_trials_per_episode, n_eval_episodes, base_seed,
        stim_timesteps=stim_timesteps, reward_timesteps=reward_timesteps,
        iti_timesteps=iti_timesteps)
    env = BatchedVigourEnv(states_b, ravail_b, active_b, reward_fa=reward_fa,
                           vigour_cost=vigour_cost, cost_type=cost_type)
    B, T, D = states_b.shape
    H = model.backbone.hidden_size
    model.eval()
    obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=device)
    hidden = None
    vig = np.zeros((B, T), np.float32)
    hid = np.zeros((B, T, H), np.float32)
    done = False; t = 0
    while not done:
        mean, _, hidden = model.step(obs, hidden)
        v = model.squash(mean)                         # deterministic policy = squashed mean
        vig[:, t] = v.cpu().numpy()
        hid[:, t] = hidden.cpu().numpy()
        obs_np, _, done, _ = env.step(v.cpu().numpy())
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        t += 1

    stim_hidden, baseline_hidden, stim_list = [], [], []
    vig_by_stim = {0: [], 1: [], 2: []}
    for b, struct in enumerate(structs):
        for tr in struct:
            ss, se = tr["stim_window"]; rs, re = tr["reward_window"]; s = tr["stimulus"]
            stim_hidden.append(hid[b, ss:se]); stim_list.append(s)
            bs0 = ss - n_iti_pre
            baseline_hidden.append(hid[b, bs0:ss] if bs0 >= 0
                                   else np.zeros((n_iti_pre, H), np.float32))
            vig_by_stim[s].append(float(vig[b, rs:re].mean()))   # mean vigour in window
    acts = {"stim_hidden": np.stack(stim_hidden),
            "baseline_hidden": np.stack(baseline_hidden),
            "stimulus": np.array(stim_list),
            "context": np.zeros(len(stim_list), dtype=int)}   # single context
    vmean = {s: float(np.mean(vig_by_stim[s])) for s in (0, 1, 2)}
    return acts, vmean


@torch.no_grad()
def infer_rpe(model, value_matrix, *, n_eval_episodes=12, n_trials_per_episode=400,
              base_seed=10_000, device="cpu", vigour_cost=1.1, cost_type="quadratic",
              reward_fa=0.0, stim_timesteps=5, reward_timesteps=3, iti_timesteps=(3, 8),
              gamma=0.9):
    """Deterministic eval: mean reward-prediction error (TD error) per stimulus.

    delta_t = r_t + gamma*V(s_{t+1}) - V(s_t), the actor-critic analog of
    dopaminergic RPE, averaged over each trial's reward window (the same window
    infer_vigour averages vigour over). V(s_t) comes for free from the same
    forward pass used to pick the action, so this costs one rollout, same as
    infer_vigour -- no extra model calls. The very last timestep bootstraps
    V(s_T)=0 (episode truncation), matching train_vigour's own end-of-episode
    treatment.
    """
    device = torch.device(device)
    value_matrix = np.asarray(value_matrix, dtype=np.float32)
    states_b, ravail_b, active_b, structs = generate_batch(
        value_matrix, n_trials_per_episode, n_eval_episodes, base_seed,
        stim_timesteps=stim_timesteps, reward_timesteps=reward_timesteps,
        iti_timesteps=iti_timesteps)
    env = BatchedVigourEnv(states_b, ravail_b, active_b, reward_fa=reward_fa,
                           vigour_cost=vigour_cost, cost_type=cost_type)
    B, T, D = states_b.shape
    model.eval()
    obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=device)
    hidden = None
    val = np.zeros((B, T + 1), np.float32)   # V(s_t); val[:,T] stays 0 (terminal bootstrap)
    rew = np.zeros((B, T), np.float32)
    done = False; t = 0
    while not done:
        mean, value, hidden = model.step(obs, hidden)
        v = model.squash(mean)
        val[:, t] = value.cpu().numpy()
        obs_np, r, done, _ = env.step(v.cpu().numpy())
        rew[:, t] = r
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        t += 1
    delta = rew + gamma * val[:, 1:] - val[:, :-1]        # (B, T) TD error

    rpe_by_stim = {0: [], 1: [], 2: []}
    for b, struct in enumerate(structs):
        for tr in struct:
            rs, re = tr["reward_window"]; s = tr["stimulus"]
            rpe_by_stim[s].append(float(delta[b, rs:re].mean()))
    return {s: float(np.mean(rpe_by_stim[s])) if rpe_by_stim[s] else float("nan")
            for s in (0, 1, 2)}


@torch.no_grad()
def infer_value(model, value_matrix, *, n_eval_episodes=12, n_trials_per_episode=400,
                base_seed=10_000, device="cpu", vigour_cost=1.1, cost_type="quadratic",
                reward_fa=0.0, stim_timesteps=5, reward_timesteps=3, iti_timesteps=(3, 8)):
    """Deterministic eval: mean CRITIC value estimate V(s) per stimulus (the raw
    prediction, not the error) -- the direct complement to infer_rpe, for checking
    whether a seed's critic has genuinely collapsed to (correctly, self-consistently)
    predicting low reward for a stimulus, vs. still expecting more than it's getting.
    Same rollout structure/cost as infer_vigour/infer_rpe."""
    device = torch.device(device)
    value_matrix = np.asarray(value_matrix, dtype=np.float32)
    states_b, ravail_b, active_b, structs = generate_batch(
        value_matrix, n_trials_per_episode, n_eval_episodes, base_seed,
        stim_timesteps=stim_timesteps, reward_timesteps=reward_timesteps,
        iti_timesteps=iti_timesteps)
    env = BatchedVigourEnv(states_b, ravail_b, active_b, reward_fa=reward_fa,
                           vigour_cost=vigour_cost, cost_type=cost_type)
    B, T, D = states_b.shape
    model.eval()
    obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=device)
    hidden = None
    val = np.zeros((B, T), np.float32)
    done = False; t = 0
    while not done:
        mean, value, hidden = model.step(obs, hidden)
        v = model.squash(mean)
        val[:, t] = value.cpu().numpy()
        obs_np, _, done, _ = env.step(v.cpu().numpy())
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        t += 1

    val_by_stim = {0: [], 1: [], 2: []}
    for b, struct in enumerate(structs):
        for tr in struct:
            rs, re = tr["reward_window"]; s = tr["stimulus"]
            val_by_stim[s].append(float(val[b, rs:re].mean()))
    return {s: float(np.mean(val_by_stim[s])) if val_by_stim[s] else float("nan")
            for s in (0, 1, 2)}


def infer_vigour_stream(model, value_matrix, *, device="cpu", n_trials_per_episode=500,
                        base_seed=10_000, vigour_cost=1.1, cost_type="quadratic",
                        reward_fa=0.0, stim_timesteps=5, reward_timesteps=3,
                        iti_timesteps=(3, 8), **_ignored):
    """Single-episode (T,H) hidden stream + that episode's trial_structure.

    Runs one deterministic episode (batch_size=1, fixed ``base_seed``) so the
    returned (hidden_2d, trial_structure, vigour_2d) line up with the same
    episode used by ``load_vigour_run``. Each trial_structure dict carries
    ``stim_window``/``reward_window``/``stimulus``/``context``/``reward_available``
    — the windowed format the vis-notebook trial-aligned / decoding cells expect.
    Reusable for the trained model, the untrained ``model_init``, or any
    checkpoint, all giving comparable trial-aligned snippets.
    """
    device = torch.device(device)
    value_matrix = np.asarray(value_matrix, dtype=np.float32)
    sb, rbv, abv, structs = generate_batch(
        value_matrix, n_trials_per_episode, 1, base_seed,
        stim_timesteps=stim_timesteps, reward_timesteps=reward_timesteps,
        iti_timesteps=iti_timesteps)
    env = BatchedVigourEnv(sb, rbv, abv, reward_fa=reward_fa,
                           vigour_cost=vigour_cost, cost_type=cost_type)
    T = sb.shape[1]; H = model.backbone.hidden_size
    model.eval()
    obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=device); hidden = None
    hid = np.zeros((T, H), np.float32); vig = np.zeros(T, np.float32); done = False; t = 0
    with torch.no_grad():
        while not done:
            mean, _, hidden = model.step(obs, hidden)
            v = model.squash(mean)
            hid[t] = hidden[0].cpu().numpy(); vig[t] = float(v[0].cpu())
            o, _, done, _ = env.step(v.cpu().numpy())
            obs = torch.as_tensor(o, dtype=torch.float32, device=device); t += 1
    return hid, structs[0], vig


def load_vigour_run(run_dir, device="cpu", n_eval_episodes=16, n_trials_per_episode=500,
                    value_matrix=None, stim_timesteps=5, reward_timesteps=3,
                    iti_timesteps=(3, 8), n_iti_pre=3):
    """Reconstruct a saved vigour run into a vis_3s1c-notebook-compatible vd dict.

    Rebuilds VigourActorCritic from model.pt (architecture inferred from the
    weight shapes — no vis_data needed), re-runs inference, and returns a dict
    mirroring the discrete vis_data contract: value_matrix, n_stimuli/n_contexts,
    stimuli/contexts, stim/reward timesteps, infer_activations (incl. the full
    hidden_states (B,T,H)), infer_trial_data (with 'licked' = mean vigour and
    'value_estimate'), lick_sc (= vigour-by-stim), a psa_results stub, and the
    loaded model. Cells that read infer_activations (tuning, selectivity, mean
    activation, decoding, RDM) work; lick/PSA-specific panels are not meaningful
    for a vigour model.
    """
    from pathlib import Path
    run_dir = Path(run_dir)
    sd = torch.load(run_dir / "model.pt", map_location=device)
    H = sd["backbone.h2h.weight"].shape[0]
    obs_dim = sd["backbone.input2h.weight"].shape[1]
    rf = sd["vigour_head.weight"].shape[1] / H
    if value_matrix is None:
        value_matrix = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)
    n_stim, n_ctx = value_matrix.shape
    bb = RNN(input_size=obs_dim, hidden_size=H, output_size=1, recurrent_gain=0.9)
    ac = VigourActorCritic(bb, action_std=0.1, readout_fraction=rf).to(device)
    ac.load_state_dict(sd); ac.eval()

    # untrained model: load model_init.pt if saved, else reconstruct from seed/init_scale
    import re
    m_is = re.search(r"is([0-9p]+)_", run_dir.name)
    m_sd = re.search(r"seed(\d+)", run_dir.name)
    init_scale = float(m_is.group(1).replace("p", ".")) if m_is else 0.05
    seed = int(m_sd.group(1)) if m_sd else 42
    bb_i = RNN(input_size=obs_dim, hidden_size=H, output_size=1, recurrent_gain=0.9)
    ac_init = VigourActorCritic(bb_i, action_std=0.1, readout_fraction=rf).to(device)
    if (run_dir / "model_init.pt").exists():
        ac_init.load_state_dict(torch.load(run_dir / "model_init.pt", map_location=device))
    else:
        torch.manual_seed(seed)
        bb_i = RNN(input_size=obs_dim, hidden_size=H, output_size=1,
                   recurrent_gain=0.9, init_scale=init_scale)
        ac_init = VigourActorCritic(bb_i, action_std=0.1, readout_fraction=rf).to(device)
    ac_init.eval()

    sb, rbv, abv, structs = generate_batch(
        value_matrix, n_trials_per_episode, n_eval_episodes, 10_000,
        stim_timesteps=stim_timesteps, reward_timesteps=reward_timesteps,
        iti_timesteps=iti_timesteps)
    env = BatchedVigourEnv(sb, rbv, abv)
    B, T, D = sb.shape
    obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=device); hidden = None
    vig = np.zeros((B, T), np.float32); val = np.zeros((B, T), np.float32)
    hid = np.zeros((B, T, H), np.float32); done = False; t = 0
    with torch.no_grad():
        while not done:
            mean, value, hidden = ac.step(obs, hidden)
            v = ac.squash(mean)
            vig[:, t] = v.cpu().numpy(); val[:, t] = value.cpu().numpy()
            hid[:, t] = hidden.cpu().numpy()
            o, _, done, _ = env.step(v.cpu().numpy())
            obs = torch.as_tensor(o, dtype=torch.float32, device=device); t += 1

    stim_h, rew_h, base_h, ctx_l, stim_l, rav_l, itd = [], [], [], [], [], [], []
    vig_by = {s: [] for s in range(n_stim)}
    gt = 0
    for b, struct in enumerate(structs):
        for tr in struct:
            ss, se = tr["stim_window"]; rs, re = tr["reward_window"]; s = tr["stimulus"]
            stim_h.append(hid[b, ss:se]); rew_h.append(hid[b, rs:re])
            bs0 = ss - n_iti_pre
            base_h.append(hid[b, bs0:ss] if bs0 >= 0 else np.zeros((n_iti_pre, H), np.float32))
            ctx_l.append(tr["context"]); stim_l.append(s); rav_l.append(tr["reward_available"])
            vw = float(vig[b, rs:re].mean()); vig_by[s].append(vw)
            itd.append({"global_trial": gt, "stimulus": s, "context": tr["context"],
                        "reward_available": tr["reward_available"],
                        "licked": vw, "value_estimate": float(val[b, rs:re].mean()),
                        "lick_count": vw * reward_timesteps})   # vigour analog of lick count
            gt += 1
    # Single-episode (T,H) stream + that episode's windowed trial_structure, for the
    # trial-aligned / decoding / untrained / checkpoint notebook cells.
    hidden_states_2d = hid[0]                                   # (T, H)
    trial_structure_win = structs[0]                            # has stim/reward windows
    acts = {"stim_hidden": np.stack(stim_h), "reward_hidden": np.stack(rew_h),
            "baseline_hidden": np.stack(base_h), "hidden_states": hidden_states_2d,
            "context": np.array(ctx_l), "stimulus": np.array(stim_l),
            "reward_available": np.array(rav_l, dtype=bool),
            "trial_structure": trial_structure_win}
    lick_sc = np.full((n_stim, n_ctx), np.nan)
    for s in range(n_stim):
        lick_sc[s, 0] = np.mean(vig_by[s]) if vig_by[s] else np.nan

    def _build_model(state_dict):
        bb2 = RNN(input_size=obs_dim, hidden_size=H, output_size=1, recurrent_gain=0.9)
        m = VigourActorCritic(bb2, action_std=0.1, readout_fraction=rf).to(device)
        m.load_state_dict(state_dict); m.eval()
        return m

    stream_kwargs = dict(value_matrix=value_matrix, device=device,
                         n_trials_per_episode=n_trials_per_episode, base_seed=10_000,
                         stim_timesteps=stim_timesteps, reward_timesteps=reward_timesteps,
                         iti_timesteps=iti_timesteps)
    return dict(
        value_matrix=value_matrix, n_stimuli=n_stim, n_contexts=n_ctx,
        stimuli=["s0 (0%)", "s1 (50%)", "s2 (100%)"][:n_stim], contexts=["c0"],
        stim_group_info=[{"name": "low", "short": "low", "indices": [0]},
                         {"name": "mid", "short": "mid", "indices": [1]},
                         {"name": "high", "short": "high", "indices": [2]}],
        reward_timesteps=reward_timesteps, stim_timesteps=stim_timesteps,
        all_trial_data=itd, infer_trial_data=itd, infer_activations=acts,
        hidden_size=H, lick_sc=lick_sc, readout_fraction=rf, obs_dim=obs_dim,
        psa_results={0: {"psa_score": np.nan, "psa_delta": np.nan,
                         "high_lick": float(lick_sc[-1, 0]),
                         "mid_lick": float(lick_sc[n_stim // 2, 0]),
                         "low_lick": float(lick_sc[0, 0]),
                         "high_stim": np.array([n_stim - 1]), "low_stim": np.array([0]),
                         "mid_stim": np.array([n_stim // 2])}},
        model=ac, model_init=ac_init, init_scale=init_scale, is_vigour=True,
        build_model=_build_model, stream_kwargs=stream_kwargs,
        infer_vigour_stream=infer_vigour_stream,
        # episode-0 observation stream (lines up with hidden_states / trial_structure)
        infer_states=sb[0], infer_reward_availability=rbv[0],
        # compatibility stubs (vigour has no discrete env / checkpoints):
        run_id=run_dir.name, seed=seed, reward_lick_fa=0.0, lick_cost=0.0,
        entropy_coef=0.0, recurrent_gain=0.9, gamma=0.9, env_kwargs={},
        grad_norms=[], explosion_resets=0)


def vigour_metrics(acts, vmean, value_matrix, *, period="stim", ttest_n_sub=1000,
                   ttest_n_rep=1):
    """Vigour-by-stim + activity-scaling + t-test selectivity, all vs value."""
    n_stim = np.asarray(value_matrix).shape[0]
    mean_acts = stimulus_mean_activations(acts, period=period)
    pop = {s: float(np.nanmean(v)) for s, v in mean_acts.items()}
    act = [pop.get(0, np.nan), pop.get(n_stim // 2, np.nan), pop.get(n_stim - 1, np.nan)]
    rp = responsive_proportions_ttest(acts, period=period, n_sub=ttest_n_sub,
                                      n_rep=ttest_n_rep)["frac_per_stim"]
    sel = [float(rp[0]), float(rp[n_stim // 2]), float(rp[-1])]
    vig = [vmean[0], vmean[1], vmean[2]]
    span = vig[2] - vig[0]
    return dict(
        vig_low=vig[0], vig_mid=vig[1], vig_high=vig[2],
        vig_mid_frac=(vig[1] - vig[0]) / span if span > 1e-6 else np.nan,
        vig_ordered=bool(vig[0] < vig[1] < vig[2]),
        act_low=act[0], act_mid=act[1], act_high=act[2],
        activity_ok=bool(act[0] < act[1] < act[2]),
        sel_low=sel[0], sel_mid=sel[1], sel_high=sel[2],
        sel_ok=bool(sel[0] < sel[1] < sel[2]),
    )


# ── activation injection (vigour analog of cxval.injection.run_injection) ──────

def reference_mean_acts(model, value_matrix, *, device="cpu", period="stim",
                        n_eval_episodes=12, n_trials_per_episode=400, base_seed=10_000,
                        vigour_cost=1.1, cost_type="quadratic"):
    """Mean hidden-activation vector per stimulus (the 'population code' for each
    stim), used to build injection directions. Returns {stim_idx: (H,) array}."""
    acts, _ = infer_vigour(model, value_matrix, n_eval_episodes=n_eval_episodes,
                           n_trials_per_episode=n_trials_per_episode, base_seed=base_seed,
                           device=device, vigour_cost=vigour_cost, cost_type=cost_type)
    key = "stim_hidden" if period == "stim" else "baseline_hidden"
    mean_per_step = acts[key].mean(axis=1)                      # (n_trials, H)
    stim = acts["stimulus"]
    return {int(s): mean_per_step[stim == s].mean(axis=0) for s in np.unique(stim)}


def run_vigour_injection(model, value_matrix, *, stim_idx_target, injection_vector, alpha,
                         device="cpu", n_eval_episodes=12, n_trials_per_episode=400,
                         base_seed=20_000, vigour_cost=1.1, cost_type="quadratic",
                         reward_fa=0.0, stim_timesteps=5, reward_timesteps=3,
                         iti_timesteps=(3, 8), inject_window="stim"):
    """Inject alpha*injection_vector into the hidden state (post-activation, clamped
    >=0) during the chosen window of TARGET-stimulus trials, then read mean vigour in
    the reward window of those trials. The perturbation propagates through the
    recurrence into the reward window, so a stim-window injection changes the
    downstream vigour. Returns dict with target reward-window vigour (mean + per-trial)
    and, for context, the mean vigour of every stimulus under the same run.
    """
    device = torch.device(device)
    value_matrix = np.asarray(value_matrix, dtype=np.float32)
    sb, rbv, abv, structs = generate_batch(
        value_matrix, n_trials_per_episode, n_eval_episodes, base_seed,
        stim_timesteps=stim_timesteps, reward_timesteps=reward_timesteps,
        iti_timesteps=iti_timesteps)
    env = BatchedVigourEnv(sb, rbv, abv, reward_fa=reward_fa,
                           vigour_cost=vigour_cost, cost_type=cost_type)
    B, T, D = sb.shape
    H = model.backbone.hidden_size
    model.eval()
    inj = torch.as_tensor(np.asarray(injection_vector, np.float32) * float(alpha),
                          dtype=torch.float32, device=device)
    wkey = "stim_window" if inject_window == "stim" else inject_window
    inject = np.zeros((B, T), bool)
    for b, struct in enumerate(structs):
        for tr in struct:
            if tr["stimulus"] == stim_idx_target:
                s0, s1 = tr[wkey]; inject[b, s0:s1] = True

    obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=device); hidden = None
    vig = np.zeros((B, T), np.float32); done = False; t = 0
    with torch.no_grad():
        while not done:
            mean, _, hidden = model.step(obs, hidden)
            v = model.squash(mean)
            vig[:, t] = v.cpu().numpy()
            m = inject[:, t]
            if m.any():
                add = torch.zeros((B, H), dtype=torch.float32, device=device)
                add[torch.as_tensor(m, device=device)] = inj
                hidden = (hidden + add).clamp(min=0.0)
            o, _, done, _ = env.step(v.cpu().numpy()); t += 1
            obs = torch.as_tensor(o, dtype=torch.float32, device=device)

    n_stim = value_matrix.shape[0]
    rew_vig = {s: [] for s in range(n_stim)}
    target_trials = []
    for b, struct in enumerate(structs):
        for tr in struct:
            rs, re = tr["reward_window"]; s = tr["stimulus"]
            vw = float(vig[b, rs:re].mean()); rew_vig[s].append(vw)
            if s == stim_idx_target:
                target_trials.append(vw)
    target_trials = np.array(target_trials, np.float32)
    return dict(
        alpha=float(alpha), stim_idx_target=stim_idx_target,
        target_vigour=float(target_trials.mean()) if len(target_trials) else np.nan,
        target_vigour_sem=float(target_trials.std() / np.sqrt(max(1, len(target_trials)))),
        target_trials=target_trials,
        vmean={s: float(np.mean(rew_vig[s])) for s in range(n_stim)})


def vigour_injection_sweep(model, value_matrix, *, stim_idx_target, injection_vector,
                           alphas, **kw):
    """Sweep injection scale alpha for one target stimulus. Returns
    (per_alpha list, summary dict with 'alphas','target_vigour','target_vigour_sem')."""
    per = [run_vigour_injection(model, value_matrix, stim_idx_target=stim_idx_target,
                                injection_vector=injection_vector, alpha=a, **kw)
           for a in alphas]
    summary = dict(
        alphas=np.asarray([float(a) for a in alphas], np.float32),
        target_vigour=np.asarray([p["target_vigour"] for p in per], np.float32),
        target_vigour_sem=np.asarray([p["target_vigour_sem"] for p in per], np.float32))
    return per, summary
