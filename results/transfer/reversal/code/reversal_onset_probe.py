#!/usr/bin/env python
"""Compute the literal trial-0-of-the-reversal probe point: per-stimulus vigour, RPE,
and critic value, evaluated on the REVERSED task using the model EXACTLY as it was at
the end of original training -- i.e. before a single reversal-phase gradient update.

Why this doesn't already exist in history.json: probe_vigour/probe_rpe/probe_value are
logged every --probe-every training UPDATES (default 20), and each update aggregates a
whole bptt_len=40-timestep window (several trials) of gradient signal -- so the first
logged post-reversal point already reflects ~20 updates' worth of learning, not the
network's naive reaction to the flipped contingency. That naive reaction IS recoverable
without retraining, though: model_init.pt in every <post_runs>/<model_type>/seed<N>/ dir
IS the exact pre-reversal model (the warm-start point), and config.json's "train" dict
has the exact min_vigour/squash_width/vigour_cost it was evaluated with -- so we can
just re-run the SAME deterministic probe (infer_vigour/infer_rpe/infer_value) train_
reversal.py itself uses, on that saved model, on the reversed value_matrix, with zero
additional training.

Writes a sidecar <post_runs>/<model_type>/seed<N>/history_onset.json (does NOT touch
the original history.json, so the raw training-time record stays untouched):
    {"vigour": [v0, v50, v100], "rpe": [...], "value": [...]}
_load_phase_seed(..., include_onset=True) in seed_groups.py picks this up automatically
and prepends it as the trials=0 point of the post-reversal curve.

Usage:
  python code/reversal_onset_probe.py --post-runs results/action_std_0p15_full/model_runs_reversal
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cxval.vigour import VigourActorCritic, infer_vigour, infer_rpe, infer_value
from cxval.models import RNN


def build_model(sd, min_vigour=0.0, squash_width=None):
    """Identical to train_reversal.py's build_model (kept independent here rather than
    imported, since that module is a __main__-style script, not a library import)."""
    H = sd["backbone.h2h.weight"].shape[0]
    obs = sd["backbone.input2h.weight"].shape[1]
    rf = sd["vigour_head.weight"].shape[1] / H
    aux = sd["stim_head.weight"].shape[0] if "stim_head.weight" in sd else 0
    ac = VigourActorCritic(RNN(input_size=obs, hidden_size=H, output_size=1,
                               recurrent_gain=0.9), action_std=0.05,
                           readout_fraction=rf, aux_n_stim=aux)
    ac.min_vigour = min_vigour
    ac.squash_width = squash_width
    ac.load_state_dict(sd); ac.eval()
    return ac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--post-runs", required=True,
                    help="model_runs_reversal dir (needs config.json + model_init.pt "
                         "per seed, both written by train_reversal.py)")
    ap.add_argument("--n-eval-episodes", type=int, default=4)
    ap.add_argument("--n-trials-per-episode", type=int, default=150)
    ap.add_argument("--force", action="store_true", help="recompute even if history_onset.json exists")
    args = ap.parse_args()

    n = 0
    for f in sorted(glob.glob(str(Path(args.post_runs) / "*" / "seed*" / "config.json"))):
        d = Path(f).parent
        onset_path = d / "history_onset.json"
        if onset_path.exists() and not args.force:
            continue
        cfg = json.loads((d / "config.json").read_text())
        tr = cfg["train"]
        sd_path = d / "model_init.pt"
        if not sd_path.exists():
            print(f"  skip {d} (no model_init.pt)"); continue
        sd = torch.load(sd_path, map_location="cpu")
        model = build_model(sd, min_vigour=tr.get("min_vigour", 0.0),
                            squash_width=tr.get("squash_width"))
        VM_rev = np.asarray(cfg["value_matrix"], np.float32)     # already the REVERSED matrix
        cost = tr["vigour_cost"]
        kw = dict(n_eval_episodes=args.n_eval_episodes,
                 n_trials_per_episode=args.n_trials_per_episode, vigour_cost=cost)
        _, vmean = infer_vigour(model, VM_rev, **kw)
        rpe = infer_rpe(model, VM_rev, **kw)
        val = infer_value(model, VM_rev, **kw)
        onset_path.write_text(json.dumps({
            "vigour": [float(vmean[s]) for s in range(3)],
            "rpe": [float(rpe[s]) for s in range(3)],
            "value": [float(val[s]) for s in range(3)],
        }, indent=2) + "\n")
        n += 1
    print(f"wrote {n} history_onset.json sidecar file(s) under {args.post_runs}")


if __name__ == "__main__":
    main()
