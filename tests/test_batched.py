"""
Regression tests for cxval.batched.

1. test_env_equivalence: the vectorised BatchedTaskEnv must reproduce the
   single-episode cxval.envs.TaskEnv exactly (obs + rewards), across reward
   settings, with zeroed padded tails.
2. test_train_smoke: train_batched runs end-to-end, stays finite, and the
   metric pipeline (batched_inference -> compute_desiderata) returns finite
   numbers.

Run:  python tests/test_batched.py    (or: pytest tests/test_batched.py)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxval.tasks import StimulusSequence, StateSequence  # noqa: E402
from cxval.envs import TaskEnv  # noqa: E402
from cxval.batched import (  # noqa: E402
    generate_batch, BatchedTaskEnv,
    train_batched, batched_inference, compute_desiderata, build_trial_data,
)

VALUE_MATRIX = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)
SETTINGS = [
    dict(reward_lick_fa=0.0,  lick_cost=0.0),
    dict(reward_lick_fa=-0.5, lick_cost=0.0),
    dict(reward_lick_fa=0.0,  lick_cost=0.1),
    dict(reward_lick_fa=-1.0, lick_cost=0.05),
]


def _single_rollout(states, ravail, actions, **rw):
    env = TaskEnv(states=states, reward_availability=ravail, **rw)
    obs, _ = env.reset()
    obs_seq, rew_seq = [obs.copy()], []
    done, t = False, 0
    while not done and t < len(actions):
        obs, r, done, _, _ = env.step(int(actions[t]))
        rew_seq.append(float(r))
        if not done:
            obs_seq.append(obs.copy())
        t += 1
    return np.array(obs_seq), np.array(rew_seq)


def test_env_equivalence():
    rng = np.random.default_rng(0)
    B, n_trials, base_seed = 4, 30, 42
    for st in SETTINGS:
        states_b, ravail_b, active_b, _ = generate_batch(
            VALUE_MATRIX, n_trials_per_episode=n_trials, batch_size=B, base_seed=base_seed)
        _, T, _ = states_b.shape
        lens = active_b.sum(axis=1).astype(int)
        actions = rng.integers(0, 2, size=(B, T)).astype(np.int64)

        benv = BatchedTaskEnv(states_b, ravail_b, active_b, **st)
        bobs, brew = [benv.reset().copy()], []
        done, t = False, 0
        while not done:
            obs_np, reward, done, _ = benv.step(actions[:, t])
            brew.append(reward.copy())
            if not done:
                bobs.append(obs_np.copy())
            t += 1
        bobs, brew = np.stack(bobs), np.stack(brew)

        for b in range(B):
            Lb = lens[b]
            s_obs, s_rew = _single_rollout(
                states_b[b, :Lb], ravail_b[b, :Lb], actions[b, :Lb], **st)
            assert np.allclose(bobs[:Lb, b, :], s_obs[:Lb], atol=1e-6), (st, b, "obs")
            assert np.allclose(brew[:Lb, b], s_rew[:Lb], atol=1e-6), (st, b, "rew")
            assert np.allclose(brew[Lb:, b], 0.0), (st, b, "pad_rew")
    print("test_env_equivalence: PASS")


def test_train_smoke():
    out = train_batched(
        VALUE_MATRIX, batch_size=8, n_trials_per_episode=80, hidden_size=24,
        reward_lick_fa=-1.0, lick_cost=0.0, base_seed=42, lr=1e-3,
        bptt_len=8, device="cpu")
    assert not out["diverged"], "smoke run diverged"
    assert len(out["grad_norms"]) > 0
    acts, lick_sc, _ = batched_inference(
        out["model"], VALUE_MATRIX, n_eval_episodes=6, n_trials_per_episode=80,
        reward_lick_fa=-1.0, device="cpu")
    d = compute_desiderata(acts, lick_sc, VALUE_MATRIX)
    for k in ("lick_low", "lick_mid", "lick_high", "act_low", "act_high",
              "frac_low", "frac_high"):
        assert np.isfinite(d[k]), f"non-finite {k}"
    td = build_trial_data(out["action_arr"], out["value_arr"], out["trial_structs"])
    assert len(td) == out["batch_size"] * 80
    print("test_train_smoke: PASS")


if __name__ == "__main__":
    test_env_equivalence()
    test_train_smoke()
    print("ALL TESTS PASS")
