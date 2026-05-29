#!/usr/bin/env python3
"""
Patch a vis_data.pkl that is missing 'infer_activations'.

Loads the saved model checkpoint, re-runs the inference pass (no training),
and adds the missing keys to the pkl in-place.  Everything else in the pkl
(behaviour data, training curves, etc.) is left untouched.

Usage:
    python scripts/patch_infer_activations.py <run_id>
    python scripts/patch_infer_activations.py <run_id> --results-dir /path/to/results
    python scripts/patch_infer_activations.py <run_id> --iti-min 3 --iti-max 8

The inference sequence is re-generated with seed = <seed from run_id> + 1,
matching the convention in the training notebooks.  ITI jitter defaults to
(3, 8) — pass --iti-min / --iti-max if the original run used different values.
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxval.tasks import StimulusSequence, StateSequence
from cxval.envs import TaskEnv
from cxval.models import RNN, ActorCritic
from cxval.agents import Agent


# ── run-id parsing ─────────────────────────────────────────────────────────

_RUN_ID_RE = re.compile(
    r"RNN_\d+s\d+c_h(?P<hidden>\d+)_g(?P<gain>[0-9.]+)"
    r"_ep\d+_bptt\d+_upd\d+_seed(?P<seed>\d+)"
)

def parse_run_id(run_id: str) -> dict:
    m = _RUN_ID_RE.search(run_id)
    if m is None:
        raise ValueError(
            f"Cannot parse run_id '{run_id}'.\n"
            "Expected pattern: RNN_<n>s<n>c_h<hidden>_g<gain>_ep<n>_bptt<n>_upd<n>_seed<n>_<date>"
        )
    return dict(
        hidden_size    = int(m.group("hidden")),
        recurrent_gain = float(m.group("gain")),
        seed           = int(m.group("seed")),
    )


# ── inference ──────────────────────────────────────────────────────────────

def run_inference(vd: dict, hidden_size: int, recurrent_gain: float,
                  seed: int, iti_timesteps: tuple, policy_clip: float) -> dict:
    value_matrix     = vd["value_matrix"]
    trials_per_phase = vd["trials_per_phase"]
    stim_timesteps   = vd["stim_timesteps"]
    reward_timesteps = vd["reward_timesteps"]

    infer_stim_seq = StimulusSequence(
        value_matrix=value_matrix,
        trials_per_phase=trials_per_phase,
        phases_per_context=1,
        context_order="sequential",
        context_reps=1,
    )
    infer_stim_seq.generate(seed=seed + 1)

    infer_state_seq = StateSequence(
        stimulus_sequence=infer_stim_seq,
        value_matrix=value_matrix,
        stim_timesteps=stim_timesteps,
        reward_timesteps=reward_timesteps,
        iti_timesteps=iti_timesteps,
    )
    infer_states, _, infer_reward_avail = infer_state_seq.generate(seed=seed + 1)

    infer_env = TaskEnv(
        states=infer_states,
        reward_availability=infer_reward_avail,
        reward_lick=vd["reward_lick"],
        reward_no_lick=vd["reward_no_lick"],
        reward_lick_miss=vd["reward_lick_miss"],
        lick_cost=vd["lick_cost"],
    )

    device   = torch.device("cpu")
    obs_dim  = infer_states.shape[1] + 2
    backbone = RNN(input_size=obs_dim, hidden_size=hidden_size,
                   output_size=1, recurrent_gain=recurrent_gain)
    ac = ActorCritic(backbone=backbone, num_actions=2,
                     policy_clip=policy_clip).to(device)
    return ac, infer_env, infer_state_seq


def collect_hidden(ac, infer_env, device):
    agent = Agent(ac, device=device)
    agent.reset()
    obs, _ = infer_env.reset()
    hidden_list = []
    done = False
    while not done:
        action, _, _ = agent.act(obs)
        hidden_list.append(agent.hidden.detach().squeeze(0))
        obs, _, done, _, _ = infer_env.step(action)
    return np.array(torch.stack(hidden_list).tolist(), dtype=np.float32)


def build_infer_activations(hidden_np: np.ndarray,
                             infer_state_seq) -> dict:
    trial_struct = infer_state_seq.trial_structure
    return {
        "hidden_states": hidden_np,
        "stim_hidden":   np.stack([
            hidden_np[t["stim_window"][0]:t["stim_window"][1]]
            for t in trial_struct
        ]),
        "reward_hidden": np.stack([
            hidden_np[t["reward_window"][0]:t["reward_window"][1]]
            for t in trial_struct
        ]),
        "context":          np.array([t["context"]          for t in trial_struct]),
        "stimulus":         np.array([t["stimulus"]         for t in trial_struct]),
        "reward_available": np.array([t["reward_available"] for t in trial_struct]),
        "trial_structure":  trial_struct,
    }


def build_stim_groups(stim_group_info: list, stimuli: list) -> tuple[dict, dict]:
    swap_idx   = [i for g in stim_group_info if "swap"   in g["name"] for i in g["indices"]]
    anchor_idx = [i for g in stim_group_info if "anchor" in g["name"] for i in g["indices"]]
    stim_groups  = {"all": None}
    group_colors = {"all": "black"}
    if swap_idx:
        stim_groups["swap"]  = swap_idx
        group_colors["swap"] = "tomato"
    if anchor_idx:
        stim_groups["anchor"]  = anchor_idx
        group_colors["anchor"] = "steelblue"
    return stim_groups, group_colors


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id", help="Run ID (folder name under results/)")
    parser.add_argument("--results-dir", default="results",
                        help="Path to results directory (default: results/)")
    parser.add_argument("--iti-min", type=int, default=3,
                        help="Minimum ITI timesteps (default: 3)")
    parser.add_argument("--iti-max", type=int, default=8,
                        help="Maximum ITI timesteps (default: 8)")
    parser.add_argument("--policy-clip", type=float, default=0.1,
                        help="Policy clip value used during training (default: 0.1)")
    args = parser.parse_args()

    run_dir   = Path(args.results_dir) / args.run_id
    pkl_path  = run_dir / "vis_data.pkl"
    model_path = run_dir / "model.pt"

    for p in [pkl_path, model_path]:
        if not p.exists():
            sys.exit(f"Not found: {p}")

    print(f"Loading  {pkl_path}")
    with open(pkl_path, "rb") as f:
        vd = pickle.load(f)

    if "infer_activations" in vd:
        print("vis_data.pkl already contains 'infer_activations' — nothing to do.")
        return

    params = parse_run_id(args.run_id)
    print(f"Parsed   hidden_size={params['hidden_size']}  "
          f"recurrent_gain={params['recurrent_gain']}  seed={params['seed']}")
    print(f"ITI      ({args.iti_min}, {args.iti_max}) timesteps")

    ac, infer_env, infer_state_seq = run_inference(
        vd,
        hidden_size    = params["hidden_size"],
        recurrent_gain = params["recurrent_gain"],
        seed           = params["seed"],
        iti_timesteps  = (args.iti_min, args.iti_max),
        policy_clip    = args.policy_clip,
    )

    print(f"Loading  {model_path}")
    ac.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    ac.eval()

    print("Running  inference ...")
    hidden_np = collect_hidden(ac, infer_env, device=torch.device("cpu"))
    print(f"         hidden_states shape: {hidden_np.shape}")

    infer_activations = build_infer_activations(hidden_np, infer_state_seq)
    stim_groups, group_colors = build_stim_groups(
        vd["stim_group_info"], vd["stimuli"]
    )

    vd["infer_activations"] = infer_activations
    vd.setdefault("stim_groups",  stim_groups)
    vd.setdefault("group_colors", group_colors)
    vd.setdefault("pooling",      "average")
    vd.setdefault("n_folds",      5)

    with open(pkl_path, "wb") as f:
        pickle.dump(vd, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved    {pkl_path}")


if __name__ == "__main__":
    main()
