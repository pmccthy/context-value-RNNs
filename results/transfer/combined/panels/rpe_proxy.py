"""rpe_proxy.py

A cheap, continuous, well-motivated reward-prediction-error PROXY, reconstructed
entirely from data already logged live during training -- no retraining, no
forward passes, no frozen-model reconstruction needed (contrast with
terminal_rpe.py's approach: repeatedly evaluating the frozen final pre/post
model on fresh trials, which only gives two point-clouds, not a trajectory).

Definition: at each probe point, for each stimulus, RPE_proxy = (the TRUE
available reward probability for that stimulus in that phase) - (the critic's
own probe_value estimate for that stimulus at that point). This is NOT the
textbook online bootstrapped TD-error delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
(cxval.vigour.infer_rpe computes that properly, but it needs consecutive
within-episode value estimates that were never logged for the main dataset) --
it's the STEADY-STATE residual that TD-error averages toward: how far the
critic's prediction is from the ground-truth reward statistics it's being
trained to match. It should decay toward 0 within a phase as the critic
converges, then jump sharply at the reversal (the true reward statistics just
flipped under stimuli whose critic value hasn't updated yet) and decay again --
the qualitative signature of a real RPE trajectory, and unlike terminal_rpe.py
it's a REAL trajectory at full probe resolution for both pre- and post-reversal,
because probe_value is logged live throughout both phases already (see
combined/panels/vigour_value.py's _has_real_probe_value check).

Two caveats worth keeping in mind: (1) it omits the vigour-cost term of the
actual reward function (true expected reward = p(reward) - cost(vigour), not
raw p(reward) alone), and (2) it's a residual against a periodically-probed
deterministic value estimate, not a moment-to-moment stochastic TD signal.
terminal_rpe.py's frozen-model estimate is an independent cross-check computed
a completely different way -- the two should broadly agree on the overall
pre/post pattern even though neither is the exact online TD-error.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "style"))
sys.path.insert(0, str(_HERE.parent / "transfer" / "code"))
sys.path.insert(0, str(_HERE.parent / "analysis"))

import json  # noqa: E402

from _tags import new_panel, save_panel  # noqa: E402
import style as S  # noqa: E402
import figure_config as FC  # noqa: E402
import vigour_value as VV  # noqa: E402

MODEL_TYPES = VV.MODEL_TYPES
STIM_ORDER = VV.STIM_ORDER  # ["0", "50", "100"]
VM_PRE = np.array([0.0, 0.5, 1.0])    # available reward probability by stim index, pre-reversal
VM_POST = VM_PRE[::-1].copy()          # post-reversal: probability swapped under the same stim identities

_COST_CACHE: dict = {}


def _vigour_cost_params(root, model_type):
    """(vigour_cost, cost_type, reward_lick, reward_fa) for this model_type, read
    from an actual run's config.json (same for every seed -- it's a training
    hyperparameter, not something that varies per seed). Falls back to the
    documented cxval defaults if no config can be found."""
    key = (str(root), model_type)
    if key in _COST_CACHE:
        return _COST_CACHE[key]
    import glob
    params = (0.8, "quadratic", 1.0, 0.0)
    for f in sorted(glob.glob(str(root / model_type / "seed*" / "config.json"))):
        try:
            cfg = json.load(open(f))
            tr = cfg.get("train", cfg)  # train_reversal.py's config.json nests differently
            params = (
                float(tr.get("vigour_cost", 0.8)),
                tr.get("cost_type", "quadratic"),
                float(tr.get("reward_lick", 1.0)),
                float(tr.get("reward_fa", 0.0)),
            )
            break
        except Exception:
            continue
    _COST_CACHE[key] = params
    return params


def _rpe_proxy_histories(root, model_type, keep, vm, cost_corrected=False):
    """Like vigour_value._load_seed_histories(..., "value", ...) but returns
    (probe_update, expected_reward - probe_value) per seed -- the reward-value
    residual. When cost_corrected=True, expected_reward also subtracts the
    vigour-effort term (0.5*cost*vigour^2 for the quadratic cost used
    throughout this project -- see cxval.vigour.BatchedVigourEnv.step),
    using the SAME probe's probe_vigour estimate at that point, instead of
    just the raw reward-availability probability vm."""
    raw_val = VV._load_seed_histories(root, model_type, "value", keep=keep)
    if not cost_corrected:
        return {seed: (x, vm[None, :] - y) for seed, (x, y) in raw_val.items()}

    raw_vig = VV._load_seed_histories(root, model_type, "vigour", keep=keep)
    cost, cost_type, reward_lick, reward_fa = _vigour_cost_params(root, model_type)
    out = {}
    for seed, (x, val) in raw_val.items():
        if seed not in raw_vig:
            continue
        _, vig = raw_vig[seed]
        n = min(len(val), len(vig))
        vig = vig[:n]
        val = val[:n]
        x = x[:n]
        effort = (0.5 * cost * vig ** 2) if cost_type == "quadratic" else (cost * vig)
        # expected_reward(stim) = p(stim)*reward_lick + (1-p(stim))*reward_fa - effort(vigour(stim))
        expected_reward = vm[None, :] * reward_lick + (1 - vm[None, :]) * reward_fa - effort
        out[seed] = (x, expected_reward - val)
    return out


def draw_rpe_proxy_vs_trials(model_type, ax=None, cost_corrected=False):
    fig, ax, _ = new_panel(ax, figsize=(6.5, 4.5))
    keep = VV.recovered_seeds(model_type)
    n_total = len(VV._load_seed_histories(VV.MODEL_RUNS_POST, model_type, "vigour"))
    reversal_x = VV._reversal_x(model_type, keep=keep, units="trials")

    pre_hist = _rpe_proxy_histories(VV.MODEL_RUNS_PRE, model_type, keep, VM_PRE, cost_corrected=cost_corrected)
    post_hist = _rpe_proxy_histories(VV.MODEL_RUNS_POST, model_type, keep, VM_POST, cost_corrected=cost_corrected)
    xp, mp, sp, n_pre = VV._stack_mean_sem(pre_hist)
    xq, mq, sq, n_post = VV._stack_mean_sem(post_hist)
    if xp is None and xq is None:
        ax.text(0.5, 0.5, "no probe_value data\n(recovered seeds)", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    r_pre = VV._trials_per_update(VV.MODEL_RUNS_PRE, model_type, keep=keep)
    r_post = VV._trials_per_update(VV.MODEL_RUNS_POST, model_type, keep=keep)
    for si, s in enumerate(STIM_ORDER):
        colour = S.STIM_COLOURS[s]
        if xp is not None:
            xp_u = xp * r_pre
            ax.plot(xp_u, mp[:, si], color=colour, lw=2, label=f"{s}%")
            ax.fill_between(xp_u, mp[:, si] - sp[:, si], mp[:, si] + sp[:, si], color=colour, alpha=0.25)
        if xq is not None:
            xq_off = xq * r_post + reversal_x
            ax.plot(xq_off, mq[:, si], color=colour, lw=2, linestyle="--")
            ax.fill_between(xq_off, mq[:, si] - sq[:, si], mq[:, si] + sq[:, si], color=colour, alpha=0.25)

    ax.axhline(0, color="0.3", lw=0.8)
    ax.axvline(reversal_x, color="0.2", linestyle=":", lw=1.5)
    ax.set_xlim(left=0)
    ax.set_xlabel("trials")
    label = FC.MODELS.get(model_type, {}).get("label", model_type)
    if cost_corrected:
        ax.set_ylabel("RPE proxy (cost-corrected)")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"{label}: RPE proxy incl. vigour cost (n={n_post}/{n_total} recovered)\n"
                     f"= [p(reward|stim) - 0.5\u00b7cost\u00b7vigour(stim)\u00b2] - critic V(stim)")
    else:
        ax.set_ylabel("RPE proxy")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"{label}: reconstructed RPE proxy (n={n_post}/{n_total} recovered)\n"
                     f"= available reward p(stim) - critic V(stim); real trajectory, not frozen-model")
    return fig


def build_all(show_tag=None):
    for mt in MODEL_TYPES:
        for cost_corrected in (False, True):
            try:
                fig = draw_rpe_proxy_vs_trials(mt, cost_corrected=cost_corrected)
            except Exception as e:
                print(f"  (skip RPE proxy vs trials for {mt} [cost_corrected={cost_corrected}]: {e})")
                continue
            tag = "MECH.RPEPROXY.cost" if cost_corrected else "MECH.RPEPROXY"
            name = f"{mt}_rpe_proxy_vs_trials_costcorrected" if cost_corrected else f"{mt}_rpe_proxy_vs_trials"
            save_panel(fig, "Mechanistic/RPEProxy", f"{tag}.{mt}", name, show_tag)


if __name__ == "__main__":
    build_all()
