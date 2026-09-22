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
import figure_config as FC  # noqa: E402

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
# checkpoint_crosscontext_decode.json (see _load_seed_checkpoint_crosscontext
# below) was only ever produced by the OLDER checkpointed rerun
# (model_runs_ckpt) -- it does not exist under model_runs_instrumented (what
# MODEL_RUNS_PRE / the "model_runs" symlink now points at), so that one
# probe's pre-reversal segment silently disappeared when the rest of this
# codebase switched to the instrumented dataset. Recovered-seed counts are
# unchanged between the two reruns (same underlying training runs, just
# with/without the extra per-update logging -- see compose.py's
# figure_mechanistic_overview docstring), so reading this one file from the
# old run for the SAME seed is a valid, consistent combination.
MODEL_RUNS_PRE_CKPT = _HERE.parent.parent / "model_runs_ckpt"
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


def _load_seed_checkpoint_crosscontext(model_type, keep=None):
    """{seed: (checkpoint_update array, crosscontext_decode array)} from each
    seed's checkpoint_crosscontext_decode.json (MODEL_RUNS_PRE / model_type /
    seed*/) -- the PRE-reversal companion to probe_crosscontext_decode: the
    SAME frozen-decoder-vs-representation test (a nearest-centroid classifier
    built from the run's OWN final pre-reversal stim_hidden), but scored
    against periodic checkpoints taken DURING the original (pre-reversal)
    training instead of forward through post-reversal training -- so this is
    a representational-CONVERGENCE curve (how decodable does the eventual
    pre-reversal code become as training progresses?), not an independent
    validation metric; it necessarily reaches ~1.0 by the final checkpoint
    since the reference decoder IS the final checkpoint. Only present in the
    checkpointed reruns (model_runs_ckpt / see the chat: '...taking the
    pre-reversal decoder just before the reversal point and testing it...
    going back to the pre-reversal [activations]...')."""
    out = {}
    for f in sorted(glob.glob(str(MODEL_RUNS_PRE_CKPT / model_type / "seed*" /
                                  "checkpoint_crosscontext_decode.json"))):
        seed = int(Path(f).parent.name.replace("seed", ""))
        if keep is not None and seed not in keep:
            continue
        d = json.load(open(f))
        cu, cd = d.get("checkpoint_update"), d.get("crosscontext_decode")
        if not cu or not cd:
            continue
        out[seed] = (np.asarray(cu, float), np.asarray(cd, float))
    return out


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
    x_max = 0.0
    for si, s in enumerate(STIM_ORDER):
        colour = S.STIM_COLOURS[s]
        if xp is not None:
            xp_u = xp * r_pre
            ax.plot(xp_u, mp[:, si], color=colour, lw=2, label=f"{s}%")
            ax.fill_between(xp_u, mp[:, si] - sp[:, si], mp[:, si] + sp[:, si], color=colour, alpha=0.25)
            x_max = max(x_max, float(xp_u[-1]))
        if xq is not None:
            xq_off = xq * r_post + reversal_x
            ax.plot(xq_off, mq[:, si], color=colour, lw=2, linestyle="--")
            ax.fill_between(xq_off, mq[:, si] - sq[:, si], mq[:, si] + sq[:, si], color=colour, alpha=0.25)
            x_max = max(x_max, float(xq_off[-1]))
    # Explicit right bound (not just left=0): matplotlib's default 5% margin
    # would otherwise pad the axis past the last real data point on both
    # sides -- set_xlim(left=0) alone only pins the left edge.
    ax.set_xlim(0, x_max if x_max > 0 else 1)
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
    elif key == "rpe":
        # Real directly-PROBED RPE (probe_rpe in history.json, from the
        # model_runs_instrumented / reversal_5000_instrumented rerun's
        # infer_rpe probe) -- not a reconstruction, unlike rpe_proxy.py's
        # cost-corrected/uncorrected proxy (built from probe_value +
        # probe_vigour + the known reward/cost structure). Same real
        # per-checkpoint trajectory machinery as vigour/value above.
        ok, n_post = _draw_trajectory(ax, model_type, "rpe", keep, n_total, reversal_x,
                                      "reward prediction error (probed)", units="trials")
        if not ok:
            ax.text(0.5, 0.5, "no probe_rpe data\n(recovered seeds)", ha="center",
                    va="center", transform=ax.transAxes)
            return fig
        n_txt = f"n={n_post}/{n_total} seeds recovered"
        ax.set_title(f"{F.MODELS[model_type]['label']} — RPE (probed) vs. trials ({n_txt}){HORIZON_LABEL}")
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
    return fig


def draw_probe_metric_vs_trials(model_type, key, ylabel, title_metric, ax=None):
    """Generic version of draw_metric_vs_trials() for any metric that ALREADY
    has a genuine per-checkpoint probe_<key> array (n_probe, 3) logged live in
    history.json -- e.g. probe_frac_responsive, probe_pop_activity, both
    already present in every seed's history.json under the checkpointed
    reruns (model_runs_ckpt / model_runs_reversal{REV_TAG}_ckpt -- see
    train_model.py/train_reversal.py's make_probe()), so unlike probe_value
    this needs no fallback-to-terminal_rpe branch. Same recovered-seeds-only,
    pre+post concatenated, reversal-marked trajectory as vigour/value (see
    _draw_trajectory)."""
    fig, ax, _ = new_panel(ax, figsize=(6.5, 4.5))
    keep = recovered_seeds(model_type)
    n_total = len(glob.glob(str(MODEL_RUNS_POST / model_type / "seed*" / "history.json")))
    reversal_x = _reversal_x(model_type, keep=keep, units="trials")
    ok, n_post = _draw_trajectory(ax, model_type, key, keep, n_total, reversal_x, ylabel, units="trials")
    if not ok:
        ax.text(0.5, 0.5, f"no probe_{key} data\n(recovered seeds)", ha="center",
                va="center", transform=ax.transAxes)
        return fig
    n_txt = f"n={n_post}/{n_total} seeds recovered"
    ax.set_title(f"{F.MODELS[model_type]['label']} — {title_metric} vs. trials ({n_txt}){HORIZON_LABEL}")
    ax.axvline(reversal_x, color="0.2", linestyle=":", lw=1.5)
    return fig


def draw_frac_responsive_vs_trials(model_type, ax=None):
    """Fraction of units responsive to each stimulus, vs trials -- a
    representational-stability readout: does the set of stimulus-responsive
    units stay stable through learning and across the reversal, or does it
    reorganise? Real per-checkpoint data (probe_frac_responsive), no rerun
    needed."""
    return draw_probe_metric_vs_trials(model_type, "frac_responsive",
                                        "fraction of units responsive",
                                        "fraction responsive", ax=ax)


def draw_pop_activity_vs_trials(model_type, ax=None):
    """Mean population hidden activity for each stimulus, vs trials -- a
    second representational-stability readout alongside frac_responsive
    (this one sensitive to activity MAGNITUDE, not just whether a unit
    crosses the responsive threshold). Real per-checkpoint data
    (probe_pop_activity), no rerun needed."""
    return draw_probe_metric_vs_trials(model_type, "pop_activity",
                                        "mean population activity",
                                        "population activity", ax=ax)


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
    ax.legend(frameon=False)
    return fig


def _load_seed_svm_crosscontext(model_type, keep=None):
    """{seed: (update array, svm_crosscontext_decode array)} from
    svm_crosscontext_decode.json (compute_svm_crosscontext_decode.py, a
    frozen-pre-reversal-reference LinearSVC scored against every saved
    post-reversal checkpoint -- same convention as the nearest-centroid
    probe's post segment, just a different classifier), restricted to
    `keep` seed ids when given. Only present for seeds/model-types the
    script has been run on; silently empty otherwise (caller checks)."""
    out = {}
    for f in sorted(glob.glob(str(MODEL_RUNS_POST / model_type / "seed*" / "svm_crosscontext_decode.json"))):
        seed = int(Path(f).parent.name.replace("seed", ""))
        if keep is not None and seed not in keep:
            continue
        d = json.load(open(f))
        updates = np.asarray([r["update"] for r in d["results"]], float)
        accs = np.asarray([r["svm_crosscontext_decode"] for r in d["results"]], float)
        out[seed] = (updates, accs)
    return out


def _load_seed_svm_pretrain_crosscontext(model_type, keep=None):
    """{seed: (checkpoint_update array, svm_crosscontext_decode array)} from
    each seed's checkpoint_svm_crosscontext_decode.json (MODEL_RUNS_PRE_CKPT /
    model_type / seed*/, produced by compute_svm_pretrain_crosscontext_decode.py)
    -- the SVM-classifier counterpart to _load_seed_checkpoint_crosscontext
    (nearest-centroid): a StandardScaler+LinearSVC fit once on that seed's
    OWN final pre-reversal model.pt rollout, then scored -- without
    refitting -- against every checkpoint saved DURING that same
    pre-reversal training run. Reference IS the final checkpoint, so by
    construction checkpoint_update's max entry is the reversal onset --
    draw_crosscontext_decode_vs_trials offsets each seed's x by that max
    before combining with the post-reversal segment, same convention as
    the nearest-centroid version. Only present for seeds the checkpointed
    pre-reversal rerun (model_runs_ckpt) covers AND the pretrain SVM script
    has been run on; silently empty otherwise (caller checks)."""
    out = {}
    for f in sorted(glob.glob(str(MODEL_RUNS_PRE_CKPT / model_type / "seed*" /
                                  "checkpoint_svm_crosscontext_decode.json"))):
        seed = int(Path(f).parent.name.replace("seed", ""))
        if keep is not None and seed not in keep:
            continue
        d = json.load(open(f))
        cu, cd = d.get("checkpoint_update"), d.get("svm_crosscontext_decode")
        if not cu or not cd:
            continue
        out[seed] = (np.asarray(cu, float), np.asarray(cd, float))
    return out


def draw_crosscontext_decode_vs_trials(model_type, ax=None):
    """Pre→post accuracy: can a linear SVM trained on the FROZEN
    pre-reversal stim_hidden representation still recognise each physical
    stimulus in the model's representation, tracked both backward through
    the pre-reversal training that produced that reference and forward
    through post-reversal training? Chance = 1/3.

    Two segments, joined at the reversal (x = 0 = trials since reversal):
      x <= 0 (pre-reversal): compute_svm_pretrain_crosscontext_decode.py --
        StandardScaler+LinearSVC fit once on THIS SEED's own final
        pre-reversal model.pt, scored -- without refitting -- against
        checkpoints saved DURING that same pre-reversal run. Because the
        reference IS the final state being approached, this is necessarily
        a representational-CONVERGENCE curve (rises toward the reference's
        own near-ceiling training accuracy), not an independent validation
        metric -- same caveat as the nearest-centroid version this
        replaced.
      x > 0 (post-reversal): compute_svm_crosscontext_decode.py -- a
        SEPARATE StandardScaler+LinearSVC fit once on the frozen
        pre-reversal reference used to INITIALISE the reversal run
        (model_init.pt), scored against every saved post-reversal
        checkpoint on the same fixed trial battery (base_seed=10_000).
    The two segments' reference classifiers are each fit fresh, on nominally
    the same pre-reversal training run/seed but via two separately executed
    reruns (model_runs_ckpt vs. model_runs_instrumented) that are not
    byte-identical checkpoints (documented non-determinism already relied
    on elsewhere in this file for the nearest-centroid version -- see
    MODEL_RUNS_PRE_CKPT's own comment) -- so a small discontinuity exactly
    at x = 0 is expected and not a bug.

    On request, switched from the fast nearest-centroid training-time probe
    (_quick_crosscontext_decode) to the SVM classifier family used
    everywhere else in this repo's decoding, for consistency. Recovered
    seeds only, same convention as vigour/value/stim_decode."""
    fig, ax, _ = new_panel(ax, figsize=(6.5, 4.5))
    keep = recovered_seeds(model_type)
    n_total = len(glob.glob(str(MODEL_RUNS_POST / model_type / "seed*" / "history.json")))
    ax.set_xlabel("trials since reversal")
    ax.set_ylabel("Pre→post accuracy")
    colour = F.MODELS[model_type]["color"]

    post_hist = _load_seed_svm_crosscontext(model_type, keep=keep)
    xs, ms, ss, n_svm = _stack_mean_sem(post_hist)
    if xs is None:
        ax.text(0.5, 0.5, "no svm_crosscontext_decode data\n(run compute_svm_crosscontext_decode.py\nfor this model type/seeds)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        return fig
    r_post = _trials_per_update(MODEL_RUNS_POST, model_type, keep=keep)
    xs_trials = xs * r_post

    pre_hist_raw = _load_seed_svm_pretrain_crosscontext(model_type, keep=keep)
    # offset each seed's own checkpoint_update so its FINAL entry (that
    # seed's own reversal onset) sits at x = 0, before positional stacking
    pre_hist = {seed: (x - x[-1], y) for seed, (x, y) in pre_hist_raw.items()}
    xp, mp, sp, n_pre = _stack_mean_sem(pre_hist)
    r_pre = _trials_per_update(MODEL_RUNS_PRE, model_type, keep=keep)

    x_min = float(np.min(xs_trials))
    if xp is not None:
        xp_trials = xp * r_pre
        ax.plot(xp_trials, mp, color=colour, lw=2, label="linear SVM")
        ax.fill_between(xp_trials, mp - sp, mp + sp, color=colour, alpha=0.25)
        x_min = min(x_min, float(np.min(xp_trials)))

    ax.plot(xs_trials, ms, color=colour, lw=2, label=None if xp is not None else "linear SVM")
    ax.fill_between(xs_trials, ms - ss, ms + ss, color=colour, alpha=0.25)
    ax.axhline(1 / 3, color="0.3", linestyle=":", lw=1.2, label="chance")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(x_min, float(np.max(xs_trials)))
    n_txt = f"n={n_svm}/{n_total} seeds recovered" if xp is None else f"n={n_svm} post / {n_pre} pre / {n_total} seeds recovered"
    ax.set_title(f"{F.MODELS[model_type]['label']} → pre→post accuracy vs. trials since reversal ({n_txt}){HORIZON_LABEL}")
    ax.axvline(0, color="0.2", linestyle=":", lw=1.5)
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


def _load_seed_histories_with_ratio(root, model_type, key, keep=None):
    """Like _load_seed_histories, but also returns each seed's OWN
    trials-per-update ratio (n_trials / total_updates read straight from
    that seed's history.json) instead of a group-averaged one -- for a
    per-seed diagnostic plot we want each seed's raw trajectory converted to
    trial units as accurately as possible, not smeared by the small
    seed-to-seed variation _trials_per_update averages over.
    {seed: (probe_update array, probe_<key> array (n_probes, 3), ratio)}."""
    out = {}
    for f in sorted(glob.glob(str(root / model_type / "seed*" / "history.json"))):
        seed = int(Path(f).parent.name.replace("seed", ""))
        if keep is not None and seed not in keep:
            continue
        h = json.load(open(f))
        if f"probe_{key}" not in h:
            continue
        nt, tu = h.get("n_trials"), h.get("total_updates")
        ratio = (nt / tu) if (nt and tu) else 1.0
        out[seed] = (np.asarray(h["probe_update"], float),
                     np.asarray(h[f"probe_{key}"], float), ratio)
    return out


def draw_vigour_reversal_asymmetry_diagnostic(model_type, ax=None, keep=None,
                                               window_trials=1500):
    """Per-SEED (not seed-averaged) raw probe_vigour trajectories, zoomed on
    the early post-reversal window, for BOTH the 0%->100% condition (stim
    "0") and the 100%->0% condition (stim "100") overlaid -- diagnostic for
    the seed-MEAN asymmetry (0->100 rising to its new post-reversal level
    slower than 100->0 falls to ITS new level, seen for classif_rl
    specifically in draw_metric_vs_trials(key="vigour")).

    The question this answers: is that mean-level asymmetry a GENUINE
    per-seed phenomenon (each individual seed's 0->100 vigour really does
    ramp up gradually over ~hundreds of trials), or an AVERAGING ARTIFACT --
    each seed actually switches fast (near-step-function), but different
    seeds switch at different post-reversal trial counts, so overlaying and
    averaging staggered fast step transitions produces an apparent slow
    ramp in the mean that no individual seed actually shows. Plotting every
    kept seed's own raw trajectory (thin, translucent lines) underneath the
    seed-mean (bold) distinguishes the two: staggered-step seeds would show
    a fan of individually-steep, horizontally-offset step curves; a genuine
    slow ramp would show each thin line itself rising gradually, roughly in
    register with the others.

    Each seed's own n_trials/total_updates ratio (not a group-averaged one --
    see _load_seed_histories_with_ratio) is used to convert that seed's
    probe_update x-axis to trial units, for the most accurate possible
    per-seed x-position of each probe point.
    """
    fig, ax, _ = new_panel(ax, figsize=(6.5, 4.8))
    keep = keep if keep is not None else recovered_seeds(model_type)
    reversal_x = _reversal_x(model_type, keep=keep, units="trials")
    post_hist = _load_seed_histories_with_ratio(MODEL_RUNS_POST, model_type, "vigour", keep=keep)
    if not post_hist:
        raise FileNotFoundError(f"no post-reversal probe_vigour histories found for {model_type}")

    conditions = [("0", "0%→100%"), ("100", "100%→0%")]
    n_seeds_used = 0
    for stim, cond_label in conditions:
        si = STIM_ORDER.index(stim)
        colour = S.STIM_COLOURS[stim]
        seed_traces = []
        for seed, (xq, yq, ratio) in sorted(post_hist.items()):
            x_trials = xq * ratio
            mask = x_trials <= window_trials
            if mask.sum() < 2:
                continue
            ax.plot(x_trials[mask], yq[mask, si], color=colour, lw=0.9, alpha=0.45, zorder=2)
            seed_traces.append((x_trials[mask], yq[mask, si]))
        n_seeds_used = max(n_seeds_used, len(seed_traces))
        # Seed-mean overlay, restricted to the same zoomed window, on a common
        # grid built from the seed with the most points in-window (others are
        # interpolated onto it) -- purely for visual reference against the
        # per-seed fan, not a new computation path.
        if seed_traces:
            x_ref = max(seed_traces, key=lambda t: len(t[0]))[0]
            ys_interp = np.stack([np.interp(x_ref, x, y) for x, y in seed_traces])
            mean = ys_interp.mean(axis=0)
            ax.plot(x_ref, mean, color=colour, lw=3, alpha=1.0, zorder=5,
                    label=f"{cond_label} (seed mean, n={len(seed_traces)})")

    if n_seeds_used == 0:
        raise FileNotFoundError(f"no post-reversal seeds within the first {window_trials} "
                                f"trials for {model_type}")

    ax.axvline(0, color="0.2", linestyle=":", lw=1.5)
    ax.set_xlim(0, window_trials)
    ax.set_xlabel("trials since reversal")
    ax.set_ylabel("vigour (probed)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    label = FC.MODELS.get(model_type, {}).get("label", model_type)
    ax.set_title(f"{label}: per-seed raw vigour, first {window_trials} trials post-reversal")
    return fig
