"""Activation injection utilities for perturbation experiments.

Two injection methods are supported:

    "sparse"    — add a fixed scalar current to each neuron that belongs to a
                  stimulus-selective group (defined by SI threshold).  Every
                  neuron in the group receives the same amount; the direction
                  is flat / uniform within the group.

    "subspace"  — add a vector proportional to the mean activation of a
                  reference stimulus.  Every neuron is moved in the direction
                  of the reference population vector.  This steers the
                  representation through activation space without assuming any
                  neuron-level grouping.

In both cases the injection is applied *post-ReLU*: the hidden state is shifted
by alpha × vector at every timestep that falls inside a stim window for the
target stimulus.  After the shift the hidden state is clamped to ≥ 0 to keep it
in the ReLU-feasible region.

Lick probability is read directly from the policy head (no action sampling)
at the first reward-window timestep of each injected trial.
"""
from __future__ import annotations

import numpy as np
import torch


# ── helpers ───────────────────────────────────────────────────────────────────

def _lick_prob_from_hidden(actor_critic, hidden_vec: torch.Tensor) -> float:
    """Return P(lick) from a single hidden-state vector (1, H) tensor."""
    with torch.no_grad():
        readout = hidden_vec[..., :actor_critic.n_readout]
        logits = actor_critic.policy_head(readout)
        probs  = actor_critic.make_dist(logits).probs  # (1, n_actions)
    return float(probs[0, 0].cpu())                     # action 0 = LICK


def build_injection_vector(
    method: str,
    ref_mean_act: np.ndarray,
    neuron_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build a unit-length injection direction vector.

    Args:
        method: "sparse" or "subspace".
        ref_mean_act: (hidden_size,) mean activation of the reference stimulus.
        neuron_mask: (hidden_size,) boolean mask identifying selective neurons.
            Required and used only for method="sparse".

    Returns:
        (hidden_size,) float32 array, unit L2 norm.
    """
    if method == "sparse":
        if neuron_mask is None:
            raise ValueError("neuron_mask required for method='sparse'")
        v = neuron_mask.astype(np.float32)
    elif method == "subspace":
        v = ref_mean_act.astype(np.float32)
    else:
        raise ValueError(f"Unknown method '{method}'; choose 'sparse' or 'subspace'")

    norm = np.linalg.norm(v)
    return v / (norm + 1e-12)


# ── core injection runner ─────────────────────────────────────────────────────

def run_injection(
    actor_critic,
    infer_states: np.ndarray,
    infer_reward_availability: np.ndarray,
    trial_structure: list[dict],
    stim_idx_target: int,
    injection_vector: np.ndarray,
    alpha: float,
    env_cls,
    env_kwargs: dict,
    device: torch.device,
) -> dict:
    """Run one full inference pass with additive injection during target stim periods.

    Rebuilds the environment from scratch so feedback observations remain
    self-consistent with the sampled actions on each run.

    Args:
        actor_critic: Trained ActorCritic (will be run in eval mode, no grad).
        infer_states: (T, obs_dim_base) pre-generated state array.
        infer_reward_availability: (T,) availability array.
        trial_structure: list of trial dicts (from StateSequence.trial_structure).
        stim_idx_target: stimulus index to inject on (e.g. mid = 1 for 3s1c).
        injection_vector: (hidden_size,) unit vector defining the injection direction.
        alpha: injection scale (positive → towards ref, negative → away).
        env_cls: TaskEnv class.
        env_kwargs: keyword arguments forwarded to env_cls (reward scalings etc.).
        device: torch device.

    Returns:
        dict with keys:
            lick_probs    — (n_target_trials,) lick probability at 1st reward ts
            stim_hidden   — (n_target_trials, stim_ts, hidden_size) post-injection
                            hidden states during stim window
            mean_stim_h   — (n_target_trials, hidden_size) mean over stim window
            trial_indices — (n_target_trials,) indices into trial_structure
    """
    actor_critic.eval()

    inj_t = torch.tensor(injection_vector * alpha, dtype=torch.float32, device=device)

    # Build a boolean mask of all timesteps that are in target stim periods
    T = infer_states.shape[0]
    inject_ts = np.zeros(T, dtype=bool)
    for trial in trial_structure:
        if trial["stimulus"] == stim_idx_target:
            s, e = trial["stim_window"]
            inject_ts[s:e] = True

    # Reconstruct env
    env = env_cls(
        states=infer_states,
        reward_availability=infer_reward_availability,
        **env_kwargs,
    )

    obs, _ = env.reset()
    hidden = None
    hidden_list = []

    with torch.no_grad():
        done = False
        t = 0
        while not done:
            obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value, hidden = actor_critic.step(obs_t, hidden)

            # Apply injection post-ReLU during target stim periods
            if inject_ts[t]:
                hidden = (hidden + inj_t.unsqueeze(0)).clamp(min=0.0)

            # Sample action to keep env feedback consistent
            action = actor_critic.make_dist(logits).sample().item()
            obs, _, done, _, _ = env.step(action)

            hidden_list.append(hidden.detach().squeeze(0).cpu())
            t += 1

    hidden_np = np.array(torch.stack(hidden_list).tolist(), dtype=np.float32)  # (T, hidden_size)

    # Collect results for target stimulus trials
    lick_probs    = []
    stim_hiddens  = []
    trial_indices = []

    for ti, trial in enumerate(trial_structure):
        if trial["stimulus"] != stim_idx_target:
            continue

        ss, se = trial["stim_window"]
        rs, _  = trial["reward_window"]

        # Lick prob from policy head at first reward timestep
        h_r = torch.tensor(hidden_np[rs], dtype=torch.float32, device=device).unsqueeze(0)
        lick_probs.append(_lick_prob_from_hidden(actor_critic, h_r))

        stim_hiddens.append(hidden_np[ss:se])
        trial_indices.append(ti)

    stim_h = np.stack(stim_hiddens) if stim_hiddens else np.zeros(
        (0, trial_structure[0]["stim_window"][1] - trial_structure[0]["stim_window"][0],
         hidden_np.shape[1])
    )

    return {
        "lick_probs":    np.array(lick_probs),
        "stim_hidden":   stim_h,
        "mean_stim_h":   stim_h.mean(axis=1) if len(stim_h) else np.zeros((0, hidden_np.shape[1])),
        "trial_indices": np.array(trial_indices),
    }


# ── sweep over injection scales ───────────────────────────────────────────────

def injection_sweep(
    actor_critic,
    infer_states: np.ndarray,
    infer_reward_availability: np.ndarray,
    trial_structure: list[dict],
    stim_idx_target: int,
    injection_vector: np.ndarray,
    alphas,
    env_cls,
    env_kwargs: dict,
    device: torch.device,
    mean_ref_acts: dict | None = None,
) -> tuple[dict, dict]:
    """Sweep over injection scales (alphas) and record lick probs + geometry.

    Args:
        (all as for run_injection, plus:)
        alphas: array-like of scalar injection strengths (include 0 for baseline).
        mean_ref_acts: optional dict {stim_idx: (hidden_size,) array}.
            When supplied, cosine similarity of the injected mean stim hidden
            state to each reference vector is computed for each trial.

    Returns:
        per_alpha: dict keyed by float(alpha) → run_injection output dict,
                   with optional "cos_sim" key (dict {ref_si: (n_trials,) array}).
        summary: dict with arrays aligned to alphas:
                 "alphas", "mean_lick_prob", "std_lick_prob", and optionally
                 "mean_cos_sim_{ref_si}" for each reference stimulus.
    """
    alphas = [float(a) for a in alphas]
    per_alpha: dict[float, dict] = {}

    for alpha in alphas:
        result = run_injection(
            actor_critic, infer_states, infer_reward_availability,
            trial_structure, stim_idx_target, injection_vector, alpha,
            env_cls, env_kwargs, device,
        )

        if mean_ref_acts is not None and len(result["mean_stim_h"]) > 0:
            cos_sim = {}
            for ref_si, ref_h in mean_ref_acts.items():
                ref_n = ref_h / (np.linalg.norm(ref_h) + 1e-12)
                h     = result["mean_stim_h"]                              # (n_trials, H)
                norms = np.linalg.norm(h, axis=1, keepdims=True)
                h_n   = h / (norms + 1e-12)
                cos_sim[ref_si] = h_n @ ref_n                             # (n_trials,)
            result["cos_sim"] = cos_sim

        per_alpha[alpha] = result

    summary: dict = {
        "alphas":          np.array(alphas),
        "mean_lick_prob":  np.array([per_alpha[a]["lick_probs"].mean() for a in alphas]),
        "std_lick_prob":   np.array([per_alpha[a]["lick_probs"].std()  for a in alphas]),
    }
    if mean_ref_acts is not None:
        for ref_si in mean_ref_acts:
            key = f"mean_cos_sim_{ref_si}"
            summary[key] = np.array([
                per_alpha[a]["cos_sim"][ref_si].mean()
                if "cos_sim" in per_alpha[a] and len(per_alpha[a]["cos_sim"][ref_si]) else np.nan
                for a in alphas
            ])

    return per_alpha, summary
