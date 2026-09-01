"""Build a cxval.analysis-compatible `act_dict` from the time_resolved/*.npz
files already on disk (unified_figures/transfer/figure_data*/time_resolved/).

This is the one piece of glue both the decoding figures AND a proper,
trial-count-controlled responsiveness test need: cxval.analysis's decode
functions (pairwise_decode, crosscontext_decode, ...) and its own
responsive_proportions_ttest all expect a dict with stim_hidden/reward_hidden/
baseline_hidden/stimulus[/context], not the raw `aligned`+`bounds` npz layout.
No new rollouts or retraining needed -- everything required is already saved.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np


def act_dict_from_npz(npz_path):
    """One seed's time_resolved npz -> act_dict (no `context` key -- caller
    adds it when combining pre/post for cross-context decoding)."""
    z = np.load(npz_path, allow_pickle=True)
    aligned = np.asarray(z["aligned"], dtype=np.float32)   # (trial, time, unit)
    stim = np.asarray(z["stimulus"]).astype(int)
    n_iti, stim_ts, rew_ts = (int(b) for b in z["bounds"])
    return dict(
        baseline_hidden=aligned[:, 0:n_iti, :],
        stim_hidden=aligned[:, n_iti:n_iti + stim_ts, :],
        reward_hidden=aligned[:, n_iti + stim_ts:n_iti + stim_ts + rew_ts, :],
        stimulus=stim,
    )


def act_dict_one_seed_pre_post(pre_npz, post_npz):
    """Combine one seed's pre- and post-reversal act_dicts into one, with a
    `context` array (0=pre, 1=post) for crosscontext_decode/pairwise_decode."""
    a_pre = act_dict_from_npz(pre_npz)
    a_post = act_dict_from_npz(post_npz)
    n_pre, n_post = len(a_pre["stimulus"]), len(a_post["stimulus"])
    out = {}
    for k in ("baseline_hidden", "stim_hidden", "reward_hidden"):
        out[k] = np.concatenate([a_pre[k], a_post[k]], axis=0)
    out["stimulus"] = np.concatenate([a_pre["stimulus"], a_post["stimulus"]])
    out["context"] = np.concatenate([np.zeros(n_pre, dtype=int), np.ones(n_post, dtype=int)])
    return out


def seed_files(time_resolved_dir, model_type):
    pattern = str(Path(time_resolved_dir) / f"{model_type}_seed*.npz")
    files = sorted(glob.glob(pattern))
    return {int(re.search(r"_seed(\d+)\.npz$", f).group(1)): f for f in files}
