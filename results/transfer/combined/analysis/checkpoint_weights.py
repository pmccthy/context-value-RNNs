"""checkpoint_weights.py

Utilities for loading RNN training checkpoints (results/transfer/model_runs_ckpt/...)
and quantifying how much weights change through learning, overall and split by
hidden-unit group (readout units 0:64 vs non-readout units 64:128).

Checkpoints only exist with fine-grained resolution for the PRE-reversal phase
(results/transfer/model_runs_ckpt/{model_type}/seed{N}/checkpoints/checkpoint_NNNNN.pt,
~61 snapshots per seed). The post-reversal runs (model_runs_reversal / model_runs_reversal_5k)
currently only have model_init.pt (= start, the pre-reversal final state) and model.pt
(= end of reversal training) -- i.e. a single before/after delta, not a trajectory.
Getting a fine reversal-phase trajectory needs a rerun with --checkpoint-every set on the
reversal training script.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

MODEL_TYPES = ["rl_only", "classif_rl", "classif_rl_readout_only"]
SEEDS = list(range(42, 72))

N_HIDDEN = 128
N_READOUT = 64  # hidden units [0:N_READOUT) feed vigour_head / value_head

# Weight matrices that define a hidden unit's *incoming* drive (row i = unit i).
INPUT_KEY = "backbone.input2h.weight"
RECURRENT_KEY = "backbone.h2h.weight"
# Downstream "read from hidden state" matrices (column i = contribution of unit i).
# Not every model_type/head combination is present in every checkpoint; filtered at load time.
# NOTE: backbone.h2o.weight (the RNN class's own built-in linear readout) is
# DELIBERATELY EXCLUDED here. VigourActorCritic.step() never calls
# self.backbone.h2o -- only vigour_head/value_head (reading the first
# N_READOUT units) and stim_head (reading all N_HIDDEN units) are used in the
# forward pass (see cxval/vigour.py's step()). h2o therefore gets no gradient
# and never updates: verified directly on a real checkpoint pair, its weight
# is bit-for-bit identical from the first to the last checkpoint. Including
# it in a "readout heads" aggregate would silently add a dead, untrained
# term to what's supposed to measure real learning.
HEAD_KEYS = [
    "vigour_head.weight",  # the actor: outputs the Gaussian policy mean over vigour
    "value_head.weight",   # the critic: state-value estimate
    "stim_head.weight",    # auxiliary SSL objective: classify the just-seen stimulus from hidden state
    "ctx_head.weight",     # auxiliary SSL objective (belief-state variant, unused by rl_only/classif_rl/classif_rl_readout_only)
]
HEAD_DISPLAY_NAMES = {
    "vigour_head.weight": "actor (vigour)",
    "value_head.weight": "critic (value)",
    "stim_head.weight": "SSL (stim id)",
    "ctx_head.weight": "SSL (context)",
}
# Short form used by circuit_diagram.py's node labels (that panel is already
# captioned "readout heads", so the parenthetical in HEAD_DISPLAY_NAMES above
# is redundant there and was cluttering the node labels).
HEAD_SHORT_NAMES = {
    "vigour_head.weight": "actor",
    "value_head.weight": "critic",
    "stim_head.weight": "stim",
    "ctx_head.weight": "ctx",
}

_COMBINED_ROOT = Path(__file__).resolve().parent.parent / "transfer"
MODEL_RUNS = _COMBINED_ROOT / "model_runs"  # symlink -> ../../model_runs_ckpt (pre-reversal)


def run_dir(model_type: str, seed: int) -> Path:
    return MODEL_RUNS / model_type / f"seed{seed}"


def list_checkpoints(rdir: Path) -> list[tuple[int, Path]]:
    ckpt_dir = rdir / "checkpoints"
    if not ckpt_dir.is_dir():
        return []
    out = []
    for p in ckpt_dir.glob("checkpoint_*.pt"):
        try:
            u = int(p.stem.split("_")[-1])
        except ValueError:
            continue
        out.append((u, p))
    return sorted(out)


def load_history(rdir: Path) -> dict:
    with open(rdir / "history.json") as f:
        return json.load(f)


def update_to_trial(u: int, history: dict) -> float:
    tot = max(history.get("total_updates", 1), 1)
    ntr = history.get("n_trials", 2500)
    return u * ntr / tot


def load_state_dict(path: Path) -> dict:
    sd = torch.load(str(path), map_location="cpu", weights_only=False)
    return {k: v.detach().cpu().numpy() for k, v in sd.items()}


def weight_change_trajectory(model_type: str, seed: int, rdir: Path | None = None) -> dict | None:
    """Consecutive-checkpoint weight-change trajectory for one (model_type, seed) run.

    Returns None if fewer than 2 checkpoints are available (e.g. no checkpoints/ dir).
    Otherwise a dict with:
      trials          : (T-1,) trial index the delta ending at this checkpoint corresponds to
      layer_delta     : {"input2h": (T-1,), "h2h": (T-1,), "heads": (T-1,)} Frobenius norms
      per_unit_delta  : (T-1, N_HIDDEN) combined ||Δinput2h[i,:]||+||Δh2h[i,:]|| per hidden unit
      input_col_delta_rate : (T-1, n_input_dims) Frobenius norm PER INPUT COLUMN (one input
                              feature -> all hidden units), rate per update. Column order
                              matches circuit_graph.INPUT_LABELS (0=context, 1..3=the three
                              stim features, 4=response cue).
      head_row_delta_rate  : {head_key: (T-1, n_rows)} Frobenius norm PER OUTPUT ROW (all
                              hidden units -> one output unit), rate per update, for each
                              head_key present in HEAD_KEYS. Row order matches that head's
                              own weight matrix (row 0 for vigour_head/value_head; for
                              stim_head, matches circuit_graph.STIM_HEAD_ROW_LABELS).

    These two extra per-column/per-row breakdowns are AGGREGATE quantities (a norm over all
    hidden units feeding/fed-by one input/output unit), not raw individual weight values --
    unlike a single hidden unit's raw weight, a norm like this has no seed-correspondence
    problem and CAN be validly averaged (mean +/- SEM) across seeds; see
    panels/weight_change.py's draw_individual_head_rates.
    """
    rdir = rdir or run_dir(model_type, seed)
    ckpts = list_checkpoints(rdir)
    if len(ckpts) < 2:
        return None
    history = load_history(rdir)
    trials = np.array([update_to_trial(u, history) for u, _ in ckpts])
    states = [load_state_dict(p) for _, p in ckpts]

    layer_delta = {"input2h": [], "h2h": [], "heads": []}
    per_unit_delta = []
    input_col_delta = []
    head_row_delta = {hk: [] for hk in HEAD_KEYS}
    n_updates = []  # updates spanned by each consecutive-checkpoint delta (for rate normalisation)
    for i in range(1, len(states)):
        prev, cur = states[i - 1], states[i]
        d_in = cur[INPUT_KEY] - prev[INPUT_KEY]
        d_h = cur[RECURRENT_KEY] - prev[RECURRENT_KEY]
        layer_delta["input2h"].append(float(np.linalg.norm(d_in)))
        layer_delta["h2h"].append(float(np.linalg.norm(d_h)))
        input_col_delta.append(np.linalg.norm(d_in, axis=0))  # (n_input_dims,)

        head_sq = []
        for hk in HEAD_KEYS:
            if hk in cur and hk in prev:
                d_hk = cur[hk] - prev[hk]
                head_sq.append(np.sum(d_hk ** 2))
                head_row_delta[hk].append(np.linalg.norm(d_hk, axis=1))  # (n_rows,)
        layer_delta["heads"].append(float(np.sqrt(np.sum(head_sq))) if head_sq else np.nan)

        per_unit = np.sqrt((d_in ** 2).sum(axis=1) + (d_h ** 2).sum(axis=1))  # (N_HIDDEN,)
        per_unit_delta.append(per_unit)
        n_updates.append(ckpts[i][0] - ckpts[i - 1][0])

    n_updates = np.asarray(n_updates, dtype=float)
    layer_delta = {k: np.asarray(v) for k, v in layer_delta.items()}
    input_col_delta = np.asarray(input_col_delta)  # (T-1, n_input_dims)
    head_row_delta = {hk: np.asarray(v) for hk, v in head_row_delta.items() if v}  # each (T-1, n_rows)
    return dict(
        trials=trials[1:],
        layer_delta=layer_delta,
        layer_delta_rate={k: v / n_updates for k, v in layer_delta.items()},  # weight change PER UPDATE
        n_updates=n_updates,
        per_unit_delta=np.asarray(per_unit_delta),  # (T-1, N_HIDDEN)
        input_col_delta_rate=input_col_delta / n_updates[:, None],
        head_row_delta_rate={hk: v / n_updates[:, None] for hk, v in head_row_delta.items()},
    )



def full_weight_change_trajectory(model_type: str, seed: int, rev_tag: str = "_5k") -> dict | None:
    """Continuous pre-reversal + reversal-phase weight-change trajectory,
    now that dense checkpoints exist through the WHOLE run (see
    model_runs_instrumented / reversal_5000_instrumented -- this supersedes
    the old dense-pre + single-coarse-post-point pattern, which existed only
    because the reversal phase used to save just model_init.pt/model.pt).

    Concatenates weight_change_trajectory()'s pre-reversal per-checkpoint
    deltas with the SAME computation run on the reversal phase's own
    checkpoints (via that function's `rdir` override), on one continuous
    array. Reversal-phase trials restart at ~0 relative to REVERSAL ONSET
    in that phase's own history.json -- NOT pre-reversal-absolute -- so
    this function does NOT offset them; the caller adds whatever shared
    reversal_x anchor it's already using (e.g. vigour_value._reversal_x, so
    this row lines up with every other row in the composite) to
    `reversal_relative_trials[n_pre:]` to get an absolute trial axis.

    Returns None if either phase is missing checkpoints. Otherwise a dict
    with the same array keys as weight_change_trajectory (layer_delta,
    layer_delta_rate, per_unit_delta, input_col_delta_rate,
    head_row_delta_rate) each now spanning both phases, plus:
      reversal_relative_trials : (T_pre + T_rev,) -- pre-reversal entries in
                                  absolute pre-reversal trial units, reversal
                                  entries in reversal-ONSET-relative units
                                  (see above -- caller must offset)
      n_pre                    : number of pre-reversal entries (index where
                                  the reversal phase begins)
    """
    pre = weight_change_trajectory(model_type, seed)
    rev_rdir = _COMBINED_ROOT / f"model_runs_reversal{rev_tag}" / model_type / f"seed{seed}"
    rev = weight_change_trajectory(model_type, seed, rdir=rev_rdir)
    if pre is None or rev is None:
        return None

    n_pre = len(pre["trials"])
    layer_delta = {k: np.concatenate([pre["layer_delta"][k], rev["layer_delta"][k]]) for k in pre["layer_delta"]}
    layer_delta_rate = {k: np.concatenate([pre["layer_delta_rate"][k], rev["layer_delta_rate"][k]])
                         for k in pre["layer_delta_rate"]}
    head_row_delta_rate = {}
    for hk in set(pre["head_row_delta_rate"]) & set(rev["head_row_delta_rate"]):
        head_row_delta_rate[hk] = np.concatenate(
            [pre["head_row_delta_rate"][hk], rev["head_row_delta_rate"][hk]], axis=0)

    return dict(
        reversal_relative_trials=np.concatenate([pre["trials"], rev["trials"]]),
        n_pre=n_pre,
        layer_delta=layer_delta,
        layer_delta_rate=layer_delta_rate,
        per_unit_delta=np.concatenate([pre["per_unit_delta"], rev["per_unit_delta"]], axis=0),
        input_col_delta_rate=np.concatenate([pre["input_col_delta_rate"], rev["input_col_delta_rate"]], axis=0),
        head_row_delta_rate=head_row_delta_rate,
    )


def readout_split(per_unit_delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """per_unit_delta: (T, N_HIDDEN) -> (mean over readout units, mean over non-readout units), each (T,)."""
    return per_unit_delta[:, :N_READOUT].mean(axis=1), per_unit_delta[:, N_READOUT:].mean(axis=1)


def reversal_delta(model_type: str, seed: int, rev_tag: str = "") -> dict | None:
    """Single before/after-reversal weight delta from model_init.pt -> model.pt.

    This is coarse (one step spanning the whole reversal run, not a trajectory) --
    it exists because post-reversal fine checkpoints are not currently saved.

    Also returns the same per-input-column / per-output-row breakdown as
    weight_change_trajectory (see that function's docstring) -- here as flat
    (n_input_dims,) / {head_key: (n_rows,)} arrays, since this is a single
    coarse delta rather than a trajectory.
    """
    rdir = _COMBINED_ROOT / f"model_runs_reversal{rev_tag}" / model_type / f"seed{seed}"
    init_p, final_p = rdir / "model_init.pt", rdir / "model.pt"
    if not (init_p.exists() and final_p.exists()):
        return None
    prev, cur = load_state_dict(init_p), load_state_dict(final_p)
    d_in = cur[INPUT_KEY] - prev[INPUT_KEY]
    d_h = cur[RECURRENT_KEY] - prev[RECURRENT_KEY]
    head_sq = [np.sum((cur[hk] - prev[hk]) ** 2) for hk in HEAD_KEYS if hk in cur and hk in prev]
    per_unit = np.sqrt((d_in ** 2).sum(axis=1) + (d_h ** 2).sum(axis=1))
    ro, nro = readout_split(per_unit[None, :])
    layer_delta = {
        "input2h": float(np.linalg.norm(d_in)),
        "h2h": float(np.linalg.norm(d_h)),
        "heads": float(np.sqrt(np.sum(head_sq))) if head_sq else np.nan,
    }
    total_updates = load_history(rdir).get("total_updates", 1) or 1
    head_row_delta = {
        hk: np.linalg.norm(cur[hk] - prev[hk], axis=1) for hk in HEAD_KEYS if hk in cur and hk in prev
    }
    return dict(
        layer_delta=layer_delta,
        layer_delta_rate={k: v / total_updates for k, v in layer_delta.items()},  # TOTAL change / total updates
        total_updates=total_updates,
        per_unit_delta=per_unit,
        readout_mean=float(ro[0]),
        non_readout_mean=float(nro[0]),
        input_col_delta_rate=np.linalg.norm(d_in, axis=0) / total_updates,
        head_row_delta_rate={hk: v / total_updates for hk, v in head_row_delta.items()},
    )


def final_pre_reversal_state(model_type: str, seed: int) -> dict:
    """State dict at the end of pre-reversal training (highest-update checkpoint,
    falls back to model.pt if no checkpoints/ dir)."""
    rdir = run_dir(model_type, seed)
    ckpts = list_checkpoints(rdir)
    if ckpts:
        return load_state_dict(ckpts[-1][1])
    return load_state_dict(rdir / "model.pt")


def final_post_reversal_state(model_type: str, seed: int, rev_tag: str = "") -> dict:
    """State dict at the end of reversal training (model.pt under
    model_runs_reversal{rev_tag}/{model_type}/seed{seed}/). Companion to
    final_pre_reversal_state() -- used e.g. for a post-reversal circuit diagram."""
    rdir = _COMBINED_ROOT / f"model_runs_reversal{rev_tag}" / model_type / f"seed{seed}"
    return load_state_dict(rdir / "model.pt")
