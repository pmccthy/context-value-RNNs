"""Post-reversal learning-RATE (gradient) analysis -- one number per stimulus
direction summarizing how fast lick rate / vigour climbs back after the
reversal, instead of having to read the full trial-resolved curves (see
chat: "quantify the gradient of the lick rate vs trials/vigour vs trials
line post reversal ... so we can compare learning rates").

Two INDEPENDENT gradient definitions, one per data source, by design (per
the chat): a fixed trial-window is the robust choice for noisy real
behavioural data (only ~13 sessions, no clean per-session "recovered"
asymptote to anchor a window on), whereas an RNN's densely-probed, far-less-
noisy training curve supports defining a genuine seed-specific recovery
point and taking the gradient up to THAT, which is a tighter (and BC
better-motivated) window than an arbitrary fixed one:

  experimental_fixed_window_gradient(window, stims=("100_to_0","0_to_100"))
      Per SESSION, per stimulus: ordinary-least-squares slope of per-trial
      anticipatory lick rate vs. trials-since-reversal, restricted to trials
      [0, window). One slope per (session, stimulus). `window` is in trials
      -- 100 and 200 both requested (see reversal_gradient_bars.py).

  rnn_recovery_gradient(model_type, stim, rel_thr=0.10, smooth_window=3)
      Per SEED (recovered seeds only -- same reward-based recovered_fraction
      >= 0.8 filter used by every other panel in this pipeline, see
      vigour_value.py): first find this seed's own RECOVERY POINT for
      `stim` (see _recovery_index), then take the OLS slope of vigour vs.
      trials from the reversal point (t=0, via the history_onset.json
      anchor) through that recovery point. NaN (excluded from the bar plot)
      if the stimulus never recovers within the post-reversal run, same
      convention as reversal/code/recovery_time.py's time-to-recovery
      functions.

      Recovery-point definition (per the chat): stimulus `stim`'s vigour is
      "recovered" once it comes within an ABSOLUTE threshold of its mirror-
      swapped pre-reversal target (target for 0%->100% stim = the OLD 100%
      stim's pre-reversal vigour, and vice versa -- only the VALUE mapping
      flips, physical stimulus identity doesn't). That absolute threshold is
      set as `rel_thr` (default 10%) of the INITIALLY-HIGH-value stimulus's
      OWN pre-reversal vigour (i.e. the "100"-stim's pre-reversal vigour),
      used for BOTH stimulus directions -- deliberately NOT each stimulus's
      own pre-reversal vigour, because the initially-LOW-value stimulus's
      pre-reversal vigour is near zero, so a percentage of THAT would be a
      vanishingly tiny (and noise-dominated) band. This mirrors
      reversal/code/recovery_time.py's time_to_recovery_vigour_target_per_stim,
      with one deliberate change: that function normalizes by the reversed-
      value SPAN (v_pre[2] - v_pre[0]), this one normalizes by v_pre[2]
      alone, per the chat's explicit request.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# EXPERIMENTAL side (real mouse lick-rate data, neuronal-representations repo)
# ---------------------------------------------------------------------------
# Cross-repo access is via reading its tidy CSV outputs directly (same
# convention as cross_model_vs_experiment/extract_experiment_group_counts.py),
# not by importing its Python modules.
NEURONAL_REPO = Path(os.environ.get("NEURONAL_REPO", str(Path.home() / "Documents" / "neuronal-representations")))
_EXP_DATA = NEURONAL_REPO / "results" / "transfer" / "data"
REV_CSV = _EXP_DATA / "lick" / "lick_rates_per_trial.csv"
SINGLE_TRIAL_META = _EXP_DATA / "single_trial" / "single_trial_metadata.csv"

REV_STIMS = ("100_to_0", "0_to_100")   # the two VALUE-REVERSING directions (50% is a control)
REV_PRETTY = {"100_to_0": "100%→0%", "0_to_100": "0%→100%"}


def _reversal_points():
    """{session: reversal boundary in session_trial_number units} -- verbatim
    logic from figures/panels/behaviour.py's own _reversal_points(), read
    straight off single_trial_metadata.csv's pre-Rev/post-Rev condition
    labels (midpoint between the last pre-Rev trial and first post-Rev trial)."""
    if not SINGLE_TRIAL_META.exists():
        return {}
    m = pd.read_csv(SINGLE_TRIAL_META)
    out = {}
    for sess, g in m.groupby("session"):
        pre = g[g.condition.str.contains("pre-Rev")].session_trial_number
        post = g[g.condition.str.contains("post-Rev")].session_trial_number
        if len(pre) and len(post):
            out[sess] = 0.5 * (pre.max() + post.min())
    return out


def experimental_fixed_window_gradient(window=100, stims=REV_STIMS, min_trials=5):
    """DataFrame with columns [session, stimulus, n_trials, slope] -- one row
    per (session, stimulus) with >= `min_trials` real per-trial lick-rate
    points inside [0, window) trials-since-reversal. `slope` is the OLS
    (np.polyfit degree-1) fit of anticipatory_lick_rate vs. trials-since-
    reversal over just that window -- i.e. Hz gained per trial, positive
    meaning lick rate climbing toward the new post-reversal target.

    No smoothing (unlike the RNN side) -- real single-trial lick rate is
    already the finest-grained unit available; a regression slope over the
    whole window is itself a form of denoising."""
    if not REV_CSV.exists():
        return pd.DataFrame(columns=["session", "stimulus", "n_trials", "slope"])
    df = pd.read_csv(REV_CSV)
    revs = _reversal_points()
    df = df[df.session.isin(revs)].copy()
    df["rel"] = df.trial_number - df.session.map(revs)
    rows = []
    for sess, sdf in df.groupby("session"):
        for stim in stims:
            s = sdf[(sdf.stimulus_label.astype(str) == stim) & (sdf.rel >= 0) & (sdf.rel < window)]
            if len(s) < min_trials:
                continue
            slope = float(np.polyfit(s.rel.values, s.anticipatory_lick_rate.values, 1)[0])
            rows.append({"session": sess, "stimulus": stim, "n_trials": len(s), "slope": slope})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RNN side (context-value-RNNs, this repo)
# ---------------------------------------------------------------------------
REV_TAG = os.environ.get("REV_TAG", "")
MODEL_RUNS_PRE = _HERE.parent / "transfer" / "model_runs"
MODEL_RUNS_POST = _HERE.parent / "transfer" / f"model_runs_reversal{REV_TAG}"
MODEL_TYPES = ["rl_only", "classif_rl", "classif_rl_readout_only"]
STIM_ORDER = ["0", "50", "100"]           # index 0="0%", 1="50%", 2="100%" (pre-reversal identity)
RNN_STIM_INDEX = {"0_to_100": 0, "100_to_0": 2}   # which STIM_ORDER slot is which direction
MIRROR = {0: 2, 1: 1, 2: 0}               # value-swap: 0<->2, 50% (1) untouched by the reversal
THR = 0.8                                  # recovered_fraction threshold, matches vigour_value.py

# Vendored from reversal/code/seed_groups.py -- see vigour_value.py's own
# vendoring of the analogous functions for why (that module's top-level
# imports pull in scipy.stats.mannwhitneyu + reversal_analysis/rsa/etc.,
# themselves pulling in torch, just to reach these small, dependency-free
# functions).
def _load_recovery_table(post_runs):
    table = {}
    for f in glob.glob(str(Path(post_runs) / "*" / "seed*" / "history.json")):
        p = Path(f); mt = p.parent.parent.name; seed = int(p.parent.name[4:])
        rf = json.loads(p.read_text()).get("recovered_fraction")
        if rf is not None:
            table.setdefault(mt, {})[seed] = rf
    return table


def _split_groups(table, thr):
    keep = {mt: {s for s, f in d.items() if f >= thr} for mt, d in table.items()}
    fail = {mt: {s for s, f in d.items() if f < thr} for mt, d in table.items()}
    return keep, fail


_recovery_cache = {}


def recovered_seeds(model_type):
    """set(seed) with recovered_fraction >= THR post-reversal -- same
    definition/cache pattern as vigour_value.py's own recovered_seeds()."""
    if model_type not in _recovery_cache:
        table = _load_recovery_table(MODEL_RUNS_POST)
        keep, _fail = _split_groups(table, THR)
        _recovery_cache[model_type] = keep.get(model_type, set())
    return _recovery_cache[model_type]


def _load_phase_seed(runs, key, include_onset=False):
    """Vendored verbatim from reversal/code/seed_groups.py -- see that
    file's docstring. {model_type: {'n_trials': int, 'seeds': [(seed,
    trials_x, vals (n_probe,3))]}}."""
    acc = {}
    for f in glob.glob(str(Path(runs) / "*" / "seed*" / "history.json")):
        p = Path(f); mt = p.parent.parent.name; seed = int(p.parent.name[4:])
        h = json.loads(p.read_text())
        if not h.get("probe_update") or not h.get("probe_" + key):
            continue
        tot = max(h.get("total_updates", 1), 1); ntr = h.get("n_trials", 2500)
        trials = np.asarray(h["probe_update"], float) * ntr / tot
        vals = np.asarray(h["probe_" + key], float)
        if include_onset:
            onset_f = p.parent / "history_onset.json"
            if onset_f.exists():
                onset = json.loads(onset_f.read_text()).get(key)
                if onset is not None:
                    trials = np.concatenate([[0.0], trials])
                    vals = np.concatenate([np.asarray(onset, float)[None, :], vals], axis=0)
        d = acc.setdefault(mt, {"n_trials": ntr, "seeds": []})
        d["seeds"].append((seed, trials, vals))
    return acc


def _smooth(y, window):
    """Centered moving average with an edge correction -- identical to
    reversal/code/recovery_time.py's own _smooth (vendored, same reasoning:
    avoid one noisy probe looking like "recovery" without pulling the whole
    module in)."""
    y = np.asarray(y, float)
    if window <= 1 or len(y) < 2:
        return y
    k = min(int(window), len(y))
    kernel = np.ones(k) / k
    num = np.convolve(y, kernel, mode="same")
    denom = np.convolve(np.ones_like(y), kernel, mode="same")
    return num / denom


def _recovery_index(tpost, vpost, v_pre_final, stim, rel_thr, smooth_window):
    """First index in `vpost` where stimulus `stim`'s (smoothed) distance to
    its mirror-swapped pre-reversal target drops to <= the ABSOLUTE
    threshold `rel_thr * v_pre_final[2]` (the initially-high-value, "100%",
    stimulus's own pre-reversal vigour -- see module docstring for why this
    one fixed reference is used for BOTH stimulus directions instead of each
    stimulus's own). Returns None if it never crosses."""
    threshold_abs = rel_thr * abs(float(v_pre_final[2]))
    if threshold_abs <= 1e-9:
        return None   # this seed never learned a meaningful pre-reversal "100%" vigour at all
    target = v_pre_final[MIRROR[stim]]
    err_t = np.abs(vpost[:, stim] - target)
    sm = _smooth(err_t, smooth_window)
    cross = np.where(sm <= threshold_abs)[0]
    return int(cross[0]) if len(cross) else None


def rnn_recovery_gradient_with_points(model_type, stim, rel_thr=0.10, smooth_window=3):
    """Like rnn_recovery_gradient(), but returns each seed's full post-
    reversal trajectory alongside the fit, instead of collapsing straight to
    the slope -- for reversal_gradient_bars.draw_rnn_fit_diagnostic_grid(),
    which needs to actually SHOW the [0, recovery_trial] window the OLS fit
    was computed from (not just report the resulting number). Returns
    [{seed, t, v, idx, slope, intercept}] -- t/v are the FULL post-reversal
    trajectory for this stim (idx marks the recovery point within it, i.e.
    the fit uses t[:idx+1]/v[:idx+1])."""
    stim_idx = RNN_STIM_INDEX[stim]
    PRE = _load_phase_seed(MODEL_RUNS_PRE, "vigour")
    POST = _load_phase_seed(MODEL_RUNS_POST, "vigour", include_onset=True)
    keep = recovered_seeds(model_type)
    out = []
    if model_type not in PRE or model_type not in POST:
        return out
    post_by_seed = {s: (t, v) for s, t, v in POST[model_type]["seeds"]}
    for seed, tpre, vpre in PRE[model_type]["seeds"]:
        if seed not in keep or seed not in post_by_seed:
            continue
        tpost, vpost = post_by_seed[seed]
        v_pre_final = vpre[-1]
        idx = _recovery_index(tpost, vpost, v_pre_final, stim_idx, rel_thr, smooth_window)
        if idx is None or idx < 1:
            continue
        slope, intercept = np.polyfit(tpost[: idx + 1], vpost[: idx + 1, stim_idx], 1)
        out.append(dict(seed=seed, t=tpost, v=vpost[:, stim_idx], idx=idx,
                         slope=float(slope), intercept=float(intercept)))
    return out


def rnn_recovery_gradient(model_type, stim, rel_thr=0.10, smooth_window=3):
    """DataFrame with columns [seed, recovery_trial, slope] for one
    (model_type, stim) -- `stim` is "0_to_100" or "100_to_0". `slope` is the
    OLS fit of vigour vs. trials-since-reversal from t=0 (reversal onset,
    via the history_onset.json anchor -- see _load_phase_seed) through the
    seed's own recovery trial. Only recovered seeds (reward-based, see
    recovered_seeds()) that ALSO cross the vigour-recovery threshold are
    included -- seeds that never cross it are dropped (mirrors
    recovery_time.py's NaN-if-never-crosses, here just excluded from the
    returned rows entirely so the caller's n-recovered count is exact)."""
    stim_idx = RNN_STIM_INDEX[stim]
    PRE = _load_phase_seed(MODEL_RUNS_PRE, "vigour")
    POST = _load_phase_seed(MODEL_RUNS_POST, "vigour", include_onset=True)
    keep = recovered_seeds(model_type)
    rows = []
    if model_type not in PRE or model_type not in POST:
        return pd.DataFrame(rows, columns=["seed", "recovery_trial", "slope"])
    post_by_seed = {s: (t, v) for s, t, v in POST[model_type]["seeds"]}
    for seed, tpre, vpre in PRE[model_type]["seeds"]:
        if seed not in keep or seed not in post_by_seed:
            continue
        tpost, vpost = post_by_seed[seed]
        v_pre_final = vpre[-1]
        idx = _recovery_index(tpost, vpost, v_pre_final, stim_idx, rel_thr, smooth_window)
        if idx is None or idx < 1:
            continue   # never recovered, or "recovered" at t=0 already (degenerate slope)
        slope = float(np.polyfit(tpost[: idx + 1], vpost[: idx + 1, stim_idx], 1)[0])
        rows.append({"seed": seed, "recovery_trial": float(tpost[idx]), "slope": slope})
    return pd.DataFrame(rows)
