"""Group 3: vigour (and value) vs. trials, mean +/- SEM across RECOVERED seeds
only, pre-reversal training concatenated with post-reversal training and the
reversal point marked with a vertical line. Uses the REAL per-checkpoint
training logs already on disk (model_runs/<model>/seed*/history.json's
probe_update/probe_vigour) -- not the terminal_rpe.py synthetic construct,
which exists only to fill in probe_value for runs that didn't log it live
(see chat).

Seed filtering: a seed is included only if it RECOVERED post-reversal, i.e.
recovered_fraction (post-reversal mean reward / that seed's own pre-reversal
mean reward, from history.json) >= THR. This matches reversal_study/code's
own seed_groups.py / terminal_rpe.py convention (THR = 0.8 there too) --
failed seeds never re-learn the reversed contingency, so folding them into
the trial-resolved average would blend a real learning curve with a flat
failure trace and distort both the vigour and value read-outs.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "style"))
sys.path.insert(0, str(_HERE.parent / "transfer" / "code"))

from _tags import new_panel, save_panel  # noqa: E402
import style as S  # noqa: E402
import figures as F  # noqa: E402

# Which post-reversal training-length run to read: "" = the original 2500-
# trials-per-episode run (model_runs_reversal), "_5k" = the longer 5000-
# trials-per-episode run (model_runs_reversal_5k), once you've generated it
# (see run_reversal_all.sh / run_all_analysis.sh). Set once via the
# environment so every panel and composite in one invocation uses the same
# horizon consistently: REV_TAG=_5k python make_panels.py && REV_TAG=_5k python compose.py
REV_TAG = os.environ.get("REV_TAG", "")
HORIZON_LABEL = f" [{REV_TAG.lstrip('_')} horizon]" if REV_TAG else ""


# ---- recovered-seed classification -----------------------------------------
# Vendored (not imported) from reversal_study/code/seed_groups.py's
# load_recovery_table / split_groups: that module's TOP-LEVEL imports pull in
# scipy.stats.mannwhitneyu plus reversal_analysis/population_similarity/rsa
# (themselves pulling in torch etc.) just to reach these two tiny, dependency
# -free functions -- so they're copied verbatim here instead of importing the
# whole module, exactly as analysis/model_group_categories.py already did for
# the responder-group counting functions.
def _load_recovery_table(post_runs):
    """{model_type: {seed: recovered_fraction}} straight from history.json."""
    table = {}
    for f in glob.glob(str(Path(post_runs) / "*" / "seed*" / "history.json")):
        p = Path(f); mt = p.parent.parent.name; seed = int(p.parent.name[4:])
        rf = json.loads(p.read_text()).get("recovered_fraction")
        if rf is not None:
            table.setdefault(mt, {})[seed] = rf
    return table


def _split_groups(table, thr):
    """{model_type: set(seed)} for recovered (>= thr) and failed (< thr)."""
    keep = {mt: {s for s, f in d.items() if f >= thr} for mt, d in table.items()}
    fail = {mt: {s for s, f in d.items() if f < thr} for mt, d in table.items()}
    return keep, fail

MODEL_RUNS_PRE = _HERE.parent / "transfer" / "model_runs"
MODEL_RUNS_POST = _HERE.parent / "transfer" / f"model_runs_reversal{REV_TAG}"
# probe_value only exists in the terminal_rpe.py synthetic output (repeated
# stochastic evals of the FROZEN final model, NOT a real training-time
# trajectory -- see terminal_rpe.py's own docstring), a different directory
# and a different x-axis semantics than the real probe_vigour training logs.
TERMINAL_RPE_PRE = (_HERE.parent.parent / "reversal" /
                     "terminal_rpe" / "model_runs")
TERMINAL_RPE_POST = (_HERE.parent.parent / "reversal" /
                      "terminal_rpe" / "model_runs_reversal")
MODEL_TYPES = ["rl_only", "classif_rl", "classif_rl_readout_only"]
STIM_ORDER = ["0", "50", "100"]

# recovered-seed threshold -- matches reversal_study/code's own convention
# (terminal_rpe.py's --thr default, seed_groups.py's usage elsewhere).
THR = 0.8

_recovery_cache = {}


def recovered_seeds(model_type):
    """set(seed) that recovered (recovered_fraction >= THR) post-reversal,
    computed from MODEL_RUNS_POST's history.json files via the same
    load_recovery_table/split_groups machinery reversal_study/code uses."""
    if model_type not in _recovery_cache:
        table = _load_recovery_table(MODEL_RUNS_POST)
        keep, _fail = _split_groups(table, THR)
        _recovery_cache[model_type] = keep.get(model_type, set())
    return _recovery_cache[model_type]


def _load_seed_histories(root, model_type, key, keep=None):
    """{seed: (probe_update array, probe_<key> array (n_probes, 3))}, restricted
    to `keep` seed ids when given."""
    out = {}
    for f in sorted(glob.glob(str(root / model_type / "seed*" / "history.json"))):
        seed = int(Path(f).parent.name.replace("seed", ""))
        if keep is not None and seed not in keep:
            continue
        h = json.load(open(f))
        if f"probe_{key}" not in h:
            continue
        out[seed] = (np.asarray(h["probe_update"], float), np.asarray(h[f"probe_{key}"], float))
    return out


def _stack_mean_sem(histories):
    """histories: {seed: (x, y (n,3))} with a COMMON x grid assumed (real
    training logs probe at fixed update intervals) -- trims to the shortest
    length seed-to-seed if a few runs logged a different number of points."""
    if not histories:
        return None, None, None, 0
    n_min = min(len(x) for x, _ in histories.values())
    x = next(iter(histories.values()))[0][:n_min]
    ys = np.stack([y[:n_min] for _, y in histories.values()])   # (n_seed, n_min, 3)
    mean = np.nanmean(ys, axis=0)
    sem = np.nanstd(ys, axis=0) / np.sqrt(max(ys.shape[0], 1))
    return x, mean, sem, ys.shape[0]


def _has_real_probe_value(model_type, keep):
    """True once a rerun has logged genuine probe_value entries (infer_value
    wired into make_probe() in train_model.py/train_reversal.py) for at
    least one kept seed in EITHER phase -- cheap existence check, doesn't
    load full histories."""
    for root in (MODEL_RUNS_PRE, MODEL_RUNS_POST):
        for f in glob.glob(str(root / model_type / "seed*" / "history.json")):
            seed = int(Path(f).parent.name.replace("seed", ""))
            if keep is not None and seed not in keep:
                continue
            h = json.load(open(f))
            if h.get("probe_value"):
                return True
    return False


def _trials_per_update(root, model_type, keep=None):
    """Trials-per-update ratio for one phase (n_trials / total_updates, read
    straight from history.json), used to convert probe_update x-axes to
    trial units -- e.g. rl_only pre-reversal: n_trials=2500,
    total_updates=850 -> ~2.94 trials/update. Constant per model type/phase
    for a fixed training config, but averaged across the available (kept)
    seeds for robustness in case of small per-run variation."""
    ratios = []
    for f in glob.glob(str(Path(root) / model_type / "seed*" / "history.json")):
        seed = int(Path(f).parent.name.replace("seed", ""))
        if keep is not None and seed not in keep:
            continue
        h = json.load(open(f))
        nt, tu = h.get("n_trials"), h.get("total_updates")
        if nt and tu:
            ratios.append(nt / tu)
    return float(np.mean(ratios)) if ratios else 1.0


def _reversal_x(model_type, keep=None, units="updates"):
    """The vigour curve's own pre-reversal endpoint, in either update or
    TRIAL units (units="trials" multiplies by the pre-reversal phase's own
    trials-per-update ratio -- see _trials_per_update), used to anchor/mark
    the reversal point."""
    pre_hist = _load_seed_histories(MODEL_RUNS_PRE, model_type, "vigour", keep=keep)
    xp, _, _, _ = _stack_mean_sem(pre_hist)
    x_end = float(xp[-1]) if xp is not None else 0.0
    if units == "trials":
        x_end *= _trials_per_update(MODEL_RUNS_PRE, model_type, keep=keep)
    return x_end


def _draw_trajectory(ax, model_type, key, keep, n_total, reversal_x, ylabel, units="trials"):
    """Shared connected-curve drawing for a real per-checkpoint probe metric
    (pre concatenated with post, reversal point marked) -- used for BOTH
    'vigour' and 'value' once a metric has genuine probe_<key> entries in
    MODEL_RUNS_PRE/POST's history.json (see train_model.py / train_reversal.py
    -- infer_value is now wired into make_probe() there, same rollout cost
    class as infer_vigour, so a rerun logs a real trial-resolved value curve,
    not just vigour).

    units="trials" (default) converts each phase's own probe_update x-values
    to trial counts via that phase's trials-per-update ratio (see
    _trials_per_update) before plotting -- `reversal_x` must already be in
    the matching units. units="updates" keeps the old raw-update x-axis."""
    pre_hist = _load_seed_histories(MODEL_RUNS_PRE, model_type, key, keep=keep)
    post_hist = _load_seed_histories(MODEL_RUNS_POST, model_type, key, keep=keep)
    xp, mp, sp, n_pre = _stack_mean_sem(pre_hist)
    xq, mq, sq, n_post = _stack_mean_sem(post_hist)
    if xp is None and xq is None:
        return False, 0
    if units == "trials":
        r_pre = _trials_per_update(MODEL_RUNS_PRE, model_type, keep=keep)
        r_post = _trials_per_update(MODEL_RUNS_POST, model_type, keep=keep)
    else:
        r_pre = r_post = 1.0
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
    ax.set_xlim(left=0)
    ax.set_xlabel("trials" if units == "trials" else "training update")
    ax.set_ylabel(ylabel)
    return True, n_post


def draw_metric_vs_trials(model_type, key="vigour", ax=None):
    """key='vigour': real per-checkpoint training logs, genuine trajectory.
    key='value': prefers the SAME kind of real per-checkpoint probe_value
    trajectory (see _draw_trajectory) now that infer_value is wired into
    make_probe() for both phases -- falls back to terminal_rpe.py's
    synthetic frozen-model reconstruction (repeated stochastic evals of the
    FROZEN final pre/post model, NOT a training trajectory -- see
    terminal_rpe.py's docstring: "point-to-point spread is sampling noise
    only") only until that rerun has been done, and says so in the title.

    Both branches use RECOVERED SEEDS ONLY (recovered_fraction >= THR post-
    reversal; see recovered_seeds() above) -- a failed seed's post-reversal
    trace is flat/uninformative and would distort the pooled mean+SEM."""
    fig, ax, _ = new_panel(ax, figsize=(6.5, 4.5))
    keep = recovered_seeds(model_type)
    n_total = len(glob.glob(str(MODEL_RUNS_POST / model_type / "seed*" / "history.json")))
    reversal_x = _reversal_x(model_type, keep=keep, units="trials")

    if key == "vigour":
        ok, n_post = _draw_trajectory(ax, model_type, "vigour", keep, n_total, reversal_x, "vigour",
                                      units="trials")
        if not ok:
            ax.text(0.5, 0.5, "no probe_vigour data\n(recovered seeds)", ha="center",
                    va="center", transform=ax.transAxes)
            return fig
        n_txt = f"n={n_post}/{n_total} seeds recovered"
        ax.set_title(f"{F.MODELS[model_type]['label']} — vigour vs. trials ({n_txt}){HORIZON_LABEL}")
    elif _has_real_probe_value(model_type, keep):
        ok, n_post = _draw_trajectory(ax, model_type, "value", keep, n_total, reversal_x,
                                      "critic value estimate V(s)", units="trials")
        n_txt = f"n={n_post}/{n_total} seeds recovered"
        ax.set_title(f"{F.MODELS[model_type]['label']} — value estimate vs. trials ({n_txt}){HORIZON_LABEL}")
    else:
        pre_hist = _load_seed_histories(TERMINAL_RPE_PRE, model_type, "value", keep=keep)
        post_hist = _load_seed_histories(TERMINAL_RPE_POST, model_type, "value", keep=keep)
        if not pre_hist and not post_hist:
            ax.text(0.5, 0.5, "no probe_value data\n(recovered seeds)", ha="center",
                    va="center", transform=ax.transAxes)
            return fig
        span = max(reversal_x * 0.06, 20.0)   # jitter width for the pseudo-trial blocks
        rng = np.random.default_rng(0)
        n_post_seeds = len(post_hist)
        for label, hist, xc in [("pre", pre_hist, reversal_x - span * 1.5),
                                 ("post", post_hist, reversal_x + span * 1.5)]:
            if not hist:
                continue
            # pool ALL (seed x block) evals per stimulus -> one mean+SEM per stimulus
            all_y = np.concatenate([y for _, y in hist.values()], axis=0)   # (n_seed*n_block, 3)
            mean = np.nanmean(all_y, axis=0)
            sem = np.nanstd(all_y, axis=0) / np.sqrt(max(all_y.shape[0], 1))
            for si, s in enumerate(STIM_ORDER):
                colour = S.STIM_COLOURS[s]
                jitter = rng.uniform(-span * 0.4, span * 0.4, size=all_y.shape[0])
                ax.scatter(xc + jitter, all_y[:, si], color=colour, s=8, alpha=0.35, zorder=2)
                ax.errorbar([xc], [mean[si]], yerr=[sem[si]], fmt="o", color=colour,
                            markersize=9, capsize=4, zorder=3,
                            label=f"{s}%" if label == "pre" else None)
        ax.set_xlim(reversal_x - span * 4, reversal_x + span * 4)
        ax.set_xticks([reversal_x - span * 1.5, reversal_x + span * 1.5])
        ax.set_xticklabels(["pre", "post"])
        ax.set_xlabel("")
        ax.set_ylabel("critic value estimate V(s)")
        ax.set_title(f"{F.MODELS[model_type]['label']} — terminal value estimate "
                     f"(n={n_post_seeds}/{n_total} recovered)\n"
                     f"PLACEHOLDER: frozen-model repeated evals, not a training trajectory "
                     f"-- rerun with the infer_value probe patch for the real curve")
        return fig

    ax.axvline(reversal_x, color="0.2", linestyle=":", lw=1.5)
    ax.text(reversal_x, 0.90, " reversal", transform=ax.get_xaxis_transform(),
            fontsize=8, va="top", ha="left", color="0.2")
    return fig


def draw_stim_decode_vs_trials(model_type, ax=None):
    """Fast numpy-only 3-way stimulus decode accuracy (probe_stim_decode --
    see _quick_stim_decode in train_model.py/train_reversal.py) vs. training
    update, pre concatenated with post, reversal marked -- answers "does the
    stimulus code stay decodable through the reversal, or does it dip while
    the representation gets rebuilt?" (per chat: expected to stay high
    throughout for classif_rl / classif_rl_readout_only, and to possibly dip
    for rl_only around the reversal point, since its code isn't a genuine
    identity code to begin with -- see the cross-context decode discussion).
    Chance = 1/3. Recovered seeds only, same as vigour/value."""
    fig, ax, _ = new_panel(ax, figsize=(6.5, 4.5))
    keep = recovered_seeds(model_type)
    n_total = len(glob.glob(str(MODEL_RUNS_POST / model_type / "seed*" / "history.json")))
    reversal_x = _reversal_x(model_type, keep=keep)

    pre_hist = _load_seed_histories(MODEL_RUNS_PRE, model_type, "stim_decode", keep=keep)
    post_hist = _load_seed_histories(MODEL_RUNS_POST, model_type, "stim_decode", keep=keep)
    xp, mp, sp, n_pre = _stack_mean_sem(pre_hist)
    xq, mq, sq, n_post = _stack_mean_sem(post_hist)
    if xp is None and xq is None:
        ax.text(0.5, 0.5, "no probe_stim_decode data\n(rerun training)", ha="center",
                va="center", transform=ax.transAxes)
        return fig
    colour = F.MODELS[model_type]["color"]
    if xp is not None:
        ax.plot(xp, mp, color=colour, lw=2)
        ax.fill_between(xp, mp - sp, mp + sp, color=colour, alpha=0.25)
    if xq is not None:
        xq_off = xq + reversal_x
        ax.plot(xq_off, mq, color=colour, lw=2, linestyle="--")
        ax.fill_between(xq_off, mq - sq, mq + sq, color=colour, alpha=0.25)
    ax.axhline(1 / 3, color="0.3", linestyle=":", lw=1.2, label="chance")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    ax.set_xlabel("training update")
    ax.set_ylabel("stimulus decode accuracy")
    n_txt = f"n={n_post}/{n_total} seeds recovered"
    ax.set_title(f"{F.MODELS[model_type]['label']} — stimulus decode vs. trials ({n_txt}){HORIZON_LABEL}")
    ax.axvline(reversal_x, color="0.2", linestyle=":", lw=1.5)
    ax.text(reversal_x, 0.90, " reversal", transform=ax.get_xaxis_transform(),
            fontsize=8, va="top", ha="left", color="0.2")
    ax.legend(frameon=False)
    return fig


def draw_crosscontext_decode_vs_trials(model_type, ax=None):
    """Pre->post accuracy: can a nearest-centroid classifier built from the
    FROZEN pre-reversal stim_hidden representation (probe_crosscontext_decode
    -- see train_reversal.py's _quick_crosscontext_decode) still recognise
    each physical stimulus in the model's CURRENT, still-adapting
    post-reversal representation, tracked at every probe point through the
    reversal? Chance = 1/3. Post-reversal only -- there is no "other
    context" to cross-decode against before the reversal has happened, so
    the x-axis is trials SINCE the reversal (0 = reversal onset), converted
    from probe_update via that phase's own trials-per-update ratio (see
    _trials_per_update). Needs a reversal-training rerun with the
    crosscontext-decode probe patch (added on request; existing
    model_runs_reversal{,_5k} runs predate it and will show the placeholder
    below until re-run). Recovered seeds only, same convention as
    vigour/value/stim_decode."""
    fig, ax, _ = new_panel(ax, figsize=(6.5, 4.5))
    keep = recovered_seeds(model_type)
    n_total = len(glob.glob(str(MODEL_RUNS_POST / model_type / "seed*" / "history.json")))
    post_hist = _load_seed_histories(MODEL_RUNS_POST, model_type, "crosscontext_decode", keep=keep)
    xq, mq, sq, n_post = _stack_mean_sem(post_hist)
    ax.set_xlabel("trials since reversal")
    ax.set_ylabel("Pre->post accuracy")
    if xq is None:
        ax.text(0.5, 0.5, "no probe_crosscontext_decode data\n(rerun reversal training with\nthe crosscontext-decode probe patch)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        return fig
    r_post = _trials_per_update(MODEL_RUNS_POST, model_type, keep=keep)
    xq_trials = xq * r_post
    colour = F.MODELS[model_type]["color"]
    ax.plot(xq_trials, mq, color=colour, lw=2)
    ax.fill_between(xq_trials, mq - sq, mq + sq, color=colour, alpha=0.25)
    ax.axhline(1 / 3, color="0.3", linestyle=":", lw=1.2, label="chance")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    n_txt = f"n={n_post}/{n_total} seeds recovered"
    ax.set_title(f"{F.MODELS[model_type]['label']} — pre->post accuracy vs. trials since reversal ({n_txt}){HORIZON_LABEL}")
    ax.axvline(0, color="0.2", linestyle=":", lw=1.5)
    ax.text(0, 0.90, " reversal", transform=ax.get_xaxis_transform(),
            fontsize=8, va="top", ha="left", color="0.2")
    ax.legend(frameon=False)
    return fig


def build_all(show_tag=None):
    for key in ["vigour", "value"]:
        for mt in MODEL_TYPES:
            try:
                fig = draw_metric_vs_trials(mt, key=key)
            except Exception as e:
                print(f"  (skip {key} vs trials for {mt}: {e})")
                continue
            save_panel(fig, "Transfer/Vigour+Value", f"TRANSFER.{key}_trials{REV_TAG}.{mt}",
                       f"{mt}_{key}_vs_trials{REV_TAG}", show_tag)
    for mt in MODEL_TYPES:
        try:
            fig = draw_stim_decode_vs_trials(mt)
        except Exception as e:
            print(f"  (skip stim decode vs trials for {mt}: {e})")
            continue
        save_panel(fig, "Transfer/Vigour+Value", f"TRANSFER.stimdecode_trials{REV_TAG}.{mt}",
                   f"{mt}_stimdecode_vs_trials{REV_TAG}", show_tag)
    for mt in MODEL_TYPES:
        try:
            fig = draw_crosscontext_decode_vs_trials(mt)
        except Exception as e:
            print(f"  (skip crosscontext decode vs trials for {mt}: {e})")
            continue
        save_panel(fig, "Transfer/Vigour+Value", f"TRANSFER.crosscontext_trials{REV_TAG}.{mt}",
                   f"{mt}_crosscontext_decode_vs_trials{REV_TAG}", show_tag)


if __name__ == "__main__":
    build_all()
