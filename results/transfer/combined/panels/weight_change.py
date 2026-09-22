"""weight_change.py

Panels quantifying where learning happens in each RNN: how much the
input->hidden, hidden->hidden (recurrent), and hidden->readout-head weights
change through training, and whether that change is concentrated in
readout units (0:64) or non-readout units (64:128).

Fine-grained trajectories now cover the FULL pre+reversal run (dense
checkpoints exist through both phases as of the model_runs_instrumented /
reversal_5000_instrumented rerun -- see checkpoint_weights.
full_weight_change_trajectory). draw_weight_change_full,
draw_individual_head_rates, and draw_weight_change_readout_split all use
the newer _full_traj/_collect_seed_series machinery and span both phases.
draw_weight_change_layers is the one holdout still PRE-reversal-only (kept
for standalone single-phase use -- not part of the main composite).
draw_reversal_delta_bars (a simple before/after bar chart) is kept
separately for the cases where a single before/after summary is what's
wanted rather than the trajectory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "style"))
sys.path.insert(0, str(_HERE.parent / "transfer" / "code"))
sys.path.insert(0, str(_HERE.parent / "analysis"))

from _tags import new_panel, save_panel  # noqa: E402
import figure_config as FC  # noqa: E402
import checkpoint_weights as CW  # noqa: E402
import vigour_value as VV  # noqa: E402
import reversal_gradients as RG  # noqa: E402
import circuit_graph as CG  # noqa: E402
import style as S  # noqa: E402

MODEL_TYPES = CW.MODEL_TYPES
CW_REV_TAG = RG.REV_TAG

LAYER_NAMES = ["input2h", "h2h", "heads"]
LAYER_COLOURS = {"input2h": "#3377bb", "h2h": "#aa3377", "heads": "#229977"}
LAYER_LABELS = {
    "input2h": "input → hidden",
    "h2h": "hidden → hidden (recurrent)",
    "heads": "hidden → readout heads",
}

_TRAJ_CACHE: dict[tuple[str, int], dict | None] = {}


def _traj(model_type: str, seed: int) -> dict | None:
    key = (model_type, seed)
    if key not in _TRAJ_CACHE:
        _TRAJ_CACHE[key] = CW.weight_change_trajectory(model_type, seed)
    return _TRAJ_CACHE[key]


def draw_weight_change_layers(model_type, ax=None, seeds=None):
    """Pre-reversal weight-change-per-step, one line per layer group, mean +/- SEM
    across RECOVERED seeds only (see vigour_value.recovered_seeds; falls back to
    every seed if recovery filtering finds none)."""
    fig, ax, _ = new_panel(ax, figsize=(6.0, 4.5))
    seeds = seeds if seeds is not None else (sorted(VV.recovered_seeds(model_type)) or CW.SEEDS)

    collected = {ln: [] for ln in LAYER_NAMES}
    trials_ref = None
    n_seeds = 0
    for seed in seeds:
        tr = _traj(model_type, seed)
        if tr is None:
            continue
        if trials_ref is None:
            trials_ref = tr["trials"]
        if len(tr["trials"]) != len(trials_ref):
            continue
        n_seeds += 1
        for ln in LAYER_NAMES:
            collected[ln].append(tr["layer_delta"][ln])

    if trials_ref is None or n_seeds == 0:
        raise FileNotFoundError(f"no pre-reversal checkpoints found for {model_type}")

    for ln in LAYER_NAMES:
        arr = np.asarray(collected[ln])  # (n_seeds, T)
        mean = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(arr), axis=0).clip(min=1))
        ax.plot(trials_ref, mean, color=LAYER_COLOURS[ln], lw=2, label=LAYER_LABELS[ln])
        ax.fill_between(trials_ref, mean - sem, mean + sem, color=LAYER_COLOURS[ln], alpha=0.2, lw=0)

    ax.set_xlim(trials_ref[0], trials_ref[-1])
    ax.set_xlabel("trial (pre-reversal)")
    ax.set_ylabel("weight-change norm per step")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    label = FC.MODELS.get(model_type, {}).get("label", model_type)
    ax.set_title(f"{label}: weight change by layer  [n={n_seeds} seeds]")
    return fig


def draw_weight_change_readout_split(model_type, ax=None, seeds=None):
    """Per-unit INCOMING-weight-change proxy (||delta input2h[i,:]|| +
    ||delta h2h[i,:]||, i.e. checkpoint_weights.per_unit_delta), split
    readout units (0:N_READOUT, the only ones vigour_head/value_head ever
    read -- see checkpoint_weights.py's HEAD_KEYS note) vs non-readout units
    (N_READOUT:N_HIDDEN, read only by stim_head where present, otherwise
    fed by no readout at all) -- full pre+reversal trajectory, mean +/- SEM
    across recovered seeds, same interpolation-based seed collection as
    draw_weight_change_full (see _collect_seed_series).

    This is a WEIGHT-CHANGE proxy, not a raw gradient-norm split: per-hidden-
    unit gradients were never logged during training (only per-MODULE
    aggregate grad_norm_backbone/vigour_head/value_head/stim_head -- see
    gradient_loss_panels.py), so an exact "gradient norm restricted to
    readout units" would need a rerun with per-unit gradient hooks added.
    What this DOES show is how much each unit's own incoming weights ACTUALLY
    moved, which integrates the same information (gradient x learning-rate x
    Adam preconditioning) over each checkpoint interval -- the natural
    substitute question ("are readout units updated differently from
    non-readout ones?") without needing new instrumentation.

    Mechanistically: readout units feed vigour_head/value_head directly, so
    under detach_readout=True (classif_rl_readout_only) RL gradients cannot
    reach ANY unit's incoming weights this way (see cxval/vigour.py's
    ro.detach()) -- readout vs non-readout should look similar there, both
    driven only by the aux/activity loss. For classif_rl (no detach),
    readout units get aux+RL gradients directly while non-readout units get
    RL pressure only indirectly, via recurrent (h2h) backprop-through-time
    from the readout units they connect to -- so a readout/non-readout gap
    here is itself informative about how far RL's influence actually
    propagates through the recurrent population beyond the units it reads
    from directly.
    """
    fig, ax, _ = new_panel(ax, figsize=(6.5, 4.8))
    keep = VV.recovered_seeds(model_type)
    seeds = seeds if seeds is not None else (sorted(keep) if keep else CW.SEEDS)
    reversal_x = VV._reversal_x(model_type, keep=set(seeds), units="trials")

    n_seeds_used = 0
    x_min, x_max = None, None
    for extractor, color, label in [
        ((lambda tr: CW.readout_split(tr["per_unit_delta"])[0]), "#aa3377",
         f"readout units (0:{CW.N_READOUT})"),
        ((lambda tr: CW.readout_split(tr["per_unit_delta"])[1]), "#3377bb",
         f"non-readout units ({CW.N_READOUT}:{CW.N_HIDDEN})"),
    ]:
        result = _collect_seed_series(model_type, seeds, extractor)
        if result is None:
            continue
        t_pre, t_post, arr_pre, arr_post = result
        n_seeds_used = max(n_seeds_used, arr_pre.shape[0])
        mean_pre, sem_pre = _mean_sem_rows(arr_pre)
        mean_post, sem_post = _mean_sem_rows(arr_post)
        t_full = np.concatenate([t_pre, t_post + reversal_x])
        mean = np.concatenate([mean_pre, mean_post])
        sem = np.concatenate([sem_pre, sem_post])
        ax.plot(t_full, mean, color=color, lw=2, label=label)
        ax.fill_between(t_full, mean - sem, mean + sem, color=color, alpha=0.2, lw=0)
        x_min = t_full[0] if x_min is None else min(x_min, t_full[0])
        x_max = t_full[-1] if x_max is None else max(x_max, t_full[-1])

    if n_seeds_used == 0:
        raise FileNotFoundError(f"no pre+reversal checkpoints found for {model_type}")

    ax.axvline(reversal_x, color="0.2", linestyle=":", lw=1.5)
    ax.set_xlabel("trials")
    ax.set_ylabel("mean per-unit weight-change rate\n(incoming, Frobenius norm)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    label = FC.MODELS.get(model_type, {}).get("label", model_type)
    ax.set_title(f"{label}: readout vs non-readout units, full pre+reversal "
                 f"trajectory [n={n_seeds_used} seeds]")
    if x_min is not None:
        ax.set_xlim(x_min, x_max)
    return fig


def draw_reversal_delta_bars(ax=None, rev_tag=None, seeds=None):
    """Coarse before/after-reversal weight delta (model_init.pt -> model.pt), all 3 model types,
    one bar group per layer. Not a trajectory -- see module docstring."""
    fig, ax, _ = new_panel(ax, figsize=(6.0, 4.5))
    rev_tag = rev_tag if rev_tag is not None else ""

    x = np.arange(len(LAYER_NAMES))
    width = 0.25
    for mi, mt in enumerate(MODEL_TYPES):
        vals = {ln: [] for ln in LAYER_NAMES}
        # recovered-only per model_type (each model_type can have a different
        # recovered-seed set), falling back to every seed if recovery
        # filtering finds none
        seeds_mt = seeds if seeds is not None else (sorted(VV.recovered_seeds(mt)) or CW.SEEDS)
        for seed in seeds_mt:
            rd = CW.reversal_delta(mt, seed, rev_tag=rev_tag)
            if rd is None:
                continue
            for ln in LAYER_NAMES:
                vals[ln].append(rd["layer_delta"][ln])
        means = [np.nanmean(vals[ln]) if vals[ln] else np.nan for ln in LAYER_NAMES]
        sems = [
            (np.nanstd(vals[ln], ddof=1) / np.sqrt(len(vals[ln]))) if len(vals[ln]) > 1 else 0.0
            for ln in LAYER_NAMES
        ]
        color = FC.MODELS.get(mt, {}).get("color", f"C{mi}")
        label = FC.MODELS.get(mt, {}).get("label", mt)
        ax.bar(x + (mi - 1) * width, means, width, yerr=sems, color=color, label=label, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([LAYER_LABELS[ln] for ln in LAYER_NAMES], rotation=15, ha="right")
    ax.set_ylabel("total weight-change norm (init to final)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"Before vs after reversal weight change{' [' + rev_tag.lstrip('_') + ' horizon]' if rev_tag else ''}\n(coarse: single delta, no fine post-reversal checkpoints yet)")
    return fig


_FULL_TRAJ_CACHE: dict[tuple[str, int], dict | None] = {}


def _full_traj(model_type: str, seed: int) -> dict | None:
    key = (model_type, seed)
    if key not in _FULL_TRAJ_CACHE:
        _FULL_TRAJ_CACHE[key] = CW.full_weight_change_trajectory(model_type, seed, rev_tag=CW_REV_TAG)
    return _FULL_TRAJ_CACHE[key]


def _collect_seed_series(model_type, seeds, extractor):
    """extractor(tr) -> 1D array aligned with tr["reversal_relative_trials"],
    or None to skip this seed for this particular series (e.g. a stim_head
    row that doesn't exist for rl_only). Interpolates every seed onto the
    FIRST successful seed's own trial grid, separately per phase, instead of
    the strict length-match used elsewhere in this file -- necessary because
    roughly half the seeds have 57 vs 58 pre-reversal checkpoints (a strict
    match would silently drop half the seeds from every average). The
    reversal-phase checkpoint count is uniform (74) across seeds so this
    interpolation is a no-op there, but is applied for symmetry/safety
    anyway. Returns (trials_pre_ref, trials_post_ref, arr_pre (n,T_pre),
    arr_post (n,T_post)) or None if no seed had this series.
    """
    ref_pre = ref_post = None
    rows_pre, rows_post = [], []
    for seed in seeds:
        tr = _full_traj(model_type, seed)
        if tr is None:
            continue
        vals = extractor(tr)
        if vals is None:
            continue
        n_pre = tr["n_pre"]
        t_pre = tr["reversal_relative_trials"][:n_pre]
        t_post = tr["reversal_relative_trials"][n_pre:]
        v_pre, v_post = vals[:n_pre], vals[n_pre:]
        if ref_pre is None:
            ref_pre, ref_post = t_pre, t_post
        rows_pre.append(np.interp(ref_pre, t_pre, v_pre))
        rows_post.append(np.interp(ref_post, t_post, v_post))
    if ref_pre is None:
        return None
    return ref_pre, ref_post, np.asarray(rows_pre), np.asarray(rows_post)


def _mean_sem_rows(arr):
    n = arr.shape[0]
    mean = arr.mean(axis=0)
    sem = arr.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return mean, sem


def draw_weight_change_full(model_type, ax=None, seeds=None):
    """Weight-change RATE (per training update -- see checkpoint_weights.
    layer_delta_rate) across the WHOLE run, pre-reversal AND reversal phase
    both now real dense-checkpoint trajectories (model_runs_instrumented /
    reversal_5000_instrumented), mean +/- SEM across recovered seeds, one
    line per layer group. Supersedes the old dense-pre + single-coarse-
    post-point version (kept only a before/after delta post-reversal,
    because that's all that used to be saved there) now that dense
    post-reversal checkpoints exist -- see full_weight_change_trajectory's
    docstring. Aligned to vigour_value.py's own reversal point (same
    recovered-seed set) so this row lines up with the vigour/value/
    coding-angle rows in the same composite figure.
    """
    fig, ax, _ = new_panel(ax, figsize=(6.5, 4.8))
    keep = VV.recovered_seeds(model_type)
    seeds = seeds if seeds is not None else (sorted(keep) if keep else CW.SEEDS)
    reversal_x = VV._reversal_x(model_type, keep=set(seeds), units="trials")

    n_seeds_used = 0
    x_min, x_max = None, None
    for ln in LAYER_NAMES:
        result = _collect_seed_series(model_type, seeds, lambda tr, ln=ln: tr["layer_delta_rate"][ln])
        if result is None:
            continue
        t_pre, t_post, arr_pre, arr_post = result
        n_seeds_used = max(n_seeds_used, arr_pre.shape[0])
        mean_pre, sem_pre = _mean_sem_rows(arr_pre)
        mean_post, sem_post = _mean_sem_rows(arr_post)
        t_full = np.concatenate([t_pre, t_post + reversal_x])
        mean = np.concatenate([mean_pre, mean_post])
        sem = np.concatenate([sem_pre, sem_post])
        ax.plot(t_full, mean, color=LAYER_COLOURS[ln], lw=2, label=LAYER_LABELS[ln])
        ax.fill_between(t_full, mean - sem, mean + sem, color=LAYER_COLOURS[ln], alpha=0.2, lw=0)
        x_min = t_full[0] if x_min is None else min(x_min, t_full[0])
        x_max = t_full[-1] if x_max is None else max(x_max, t_full[-1])

    ax.axvline(reversal_x, color="0.2", linestyle=":", lw=1.5)
    ax.set_xlabel("trials")
    ax.set_ylabel("weight change rate\n(Frobenius norm. diff matrix)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    label = FC.MODELS.get(model_type, {}).get("label", model_type)
    ax.set_title(f"{label}: weight-change rate, full pre+reversal trajectory [n={n_seeds_used} seeds]")
    # Explicit data-delimited xlim -- otherwise matplotlib's default 5%
    # margin pads whitespace past the first/last real trial on both sides.
    if x_min is not None:
        ax.set_xlim(x_min, x_max)
    return fig

# Which (head_key, row_idx, display_label) an "output" string in
# draw_individual_weight_trajectory refers to -- same physical-stimulus
# transition-aware labels as circuit_graph.STIM_HEAD_ROW_LABELS, plus the
# two scalar heads. Built lazily per model_type since stim_head is only
# present for classif_rl / classif_rl_readout_only.
def _resolve_output_spec(model_type, output, seed):
    sd = CG._final_state(model_type, seed, "pre")
    if output in ("actor", "vigour_head.weight"):
        return "vigour_head.weight", 0, "actor"
    if output in ("critic", "value_head.weight"):
        return "value_head.weight", 0, "critic"
    for ri, lbl in enumerate(CG.STIM_HEAD_ROW_LABELS):
        if output in (lbl, lbl.replace("stim ", ""), f"stim:{ri}", f"stim_head[{ri}]"):
            if "stim_head.weight" not in sd:
                raise FileNotFoundError(f"{model_type} has no stim_head (RL-only model)")
            return "stim_head.weight", ri, lbl
    raise ValueError(f"unrecognised output spec {output!r} -- try 'actor', 'critic', "
                      f"or one of {CG.STIM_HEAD_ROW_LABELS}")


def draw_individual_weight_trajectory(model_type, output="0%\u2192100%", seed=None, granularity="broad",
                                      ax=None):
    """Higher-granularity companion to draw_weight_change_full(): instead of
    one aggregated Frobenius-norm rate per layer, this plots each INDIVIDUAL
    hidden-unit's weight onto one chosen output row (default: the stim_head
    row for physical stimulus 0, "0%->100%" -- pass output="actor",
    "critic", or one of circuit_graph.STIM_HEAD_ROW_LABELS for others)
    across the dense pre-reversal checkpoints, one thin line per hidden
    unit, coloured by that unit's PRE-reversal functional subgroup (same
    categories/colours as circuit_diagram.py) so you can see whether e.g.
    "pref 0%->100%" units' weights onto this output move together or
    diverge -- something the aggregated norm can't show.

    NECESSARILY single-seed (not seed-averaged): individual hidden units
    have no correspondence across independently-initialised/trained
    networks, so averaging individual weight VALUES across seeds would mix
    unrelated units. Defaults to the lowest-numbered recovered seed for
    reproducibility; pass seed= explicitly to pick another. The single
    coarse post-reversal endpoint (model_init.pt -> model.pt, same
    before/after-only limitation as draw_weight_change_full) is shown as an
    open marker per unit, coloured the same way.
    """
    fig, ax, _ = new_panel(ax, figsize=(6.5, 4.8))
    if seed is None:
        rec = VV.recovered_seeds(model_type)
        seed = min(rec) if rec else CW.SEEDS[0]

    head_key, row_idx, out_label = _resolve_output_spec(model_type, output, seed)

    rdir = CW.run_dir(model_type, seed)
    ckpts = CW.list_checkpoints(rdir)
    if len(ckpts) < 2:
        ax.text(0.5, 0.5, f"no dense checkpoints for {model_type}/seed{seed}",
                ha="center", va="center", transform=ax.transAxes)
        return fig
    history = CW.load_history(rdir)
    trials = np.array([CW.update_to_trial(u, history) for u, _ in ckpts])
    states = [CW.load_state_dict(p) for _, p in ckpts]
    weights = np.stack([sd[head_key][row_idx, :] for sd in states])  # (T, N_HIDDEN)

    cats, n_groups = CG._cats_for(model_type, seed, granularity, "pre")
    labels_full = CG.GRANULARITIES[granularity]["labels"]
    colours_full = {"broad": [S.NONRESPONSIVE_COLOUR, S.STIM_COLOURS["0"], S.STIM_COLOURS["50"], S.STIM_COLOURS["100"]]}.get(
        granularity, [f"C{i}" for i in range(n_groups)])
    GROUP_LABEL_OVERRIDE = {"pref 0%": "pref 0%\u2192100%", "pref 100%": "pref 100%\u21920%"}

    # vigour_head/value_head only read from the first N_READOUT (=64) hidden
    # units (see checkpoint_weights.py's own comment on this) -- stim_head
    # reads all 128, so n_cols below is 64 or 128 depending on which output
    # was picked; restrict to units this output actually has a weight for.
    n_cols = weights.shape[1]
    for g in range(n_groups):
        unit_idx = np.where(cats == g)[0]
        unit_idx = unit_idx[unit_idx < n_cols]
        if unit_idx.size == 0:
            continue
        disp = GROUP_LABEL_OVERRIDE.get(labels_full[g], labels_full[g])
        # NOTE: label deliberately omits this panel's own per-seed n= count
        # (unlike its standalone title, which keeps it) -- compose.py's
        # _combine_legends() dedupes the whole composite's legend by EXACT
        # label string, and this row appears once per model_type/column
        # with a different n per group each time; a dynamic "(n=...)" would
        # defeat the dedupe and flood the shared legend with near-duplicates.
        for j, ui in enumerate(unit_idx):
            ax.plot(trials, weights[:, ui], color=colours_full[g], lw=0.7, alpha=0.55,
                    label=disp if j == 0 else None)

    # coarse post-reversal endpoint, same units, same colours
    rd_post = CW.final_post_reversal_state(model_type, seed, rev_tag=CW_REV_TAG)
    if head_key in rd_post:
        w_post = rd_post[head_key][row_idx, :]
        reversal_x = VV._reversal_x(model_type, keep={seed}, units="trials")
        span = max(reversal_x * 0.06, 30.0)
        n_cols_post = w_post.shape[0]
        for g in range(n_groups):
            unit_idx = np.where(cats == g)[0]
            unit_idx = unit_idx[unit_idx < n_cols_post]
            if unit_idx.size == 0:
                continue
            ax.scatter(np.full(unit_idx.size, reversal_x + span), w_post[unit_idx],
                       color=colours_full[g], s=10, alpha=0.7, zorder=4, marker="o",
                       edgecolors="none")
        ax.axvline(reversal_x, color="0.2", linestyle=":", lw=1.2)

    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xlabel("trials")
    ax.set_ylabel(f"individual weight -> {out_label}")
    ax.legend(frameon=False, fontsize=7, loc="best")
    label = FC.MODELS.get(model_type, {}).get("label", model_type)
    ax.set_title(f"{label}: individual unit weights \u2192 {out_label}\n"
                 f"(single example seed{seed}, dense pre-reversal + coarse post-reversal point)")
    return fig



# ---------------------------------------------------------------------------
# Redesigned individual-weight view, v2 (per user feedback on v1): each line
# is now an AGGREGATE norm over one input/output unit's whole weight vector
# (not one line per hidden unit), so there are only as many lines as there
# are input/output units -- 3 stim inputs, 2 (rl_only: actor+critic only) to
# 5 (actor+critic+3 stim outputs) outputs, plus ONE aggregate recurrent
# (h2h) line -- and, being norms rather than raw per-hidden-unit weight
# values, they CAN be validly averaged (mean +/- SEM) across RECOVERED
# seeds, exactly like draw_weight_change_full's per-layer lines (no
# single-seed restriction here; see checkpoint_weights.
# weight_change_trajectory's docstring). This supersedes v1's per-hidden-
# unit grid (draw_individual_weight_trajectory is kept for anyone who still
# wants the raw single-seed per-unit view, but is no longer used by
# build_all/compose.py).
# ---------------------------------------------------------------------------

def _stim_key(label: str) -> str:
    """Map a stimulus label (any of INPUT_LABELS' "stim ..." entries or
    STIM_HEAD_ROW_LABELS) to the style.STIM_COLOURS key ("0", "50", "100")
    for that physical stimulus, so input and output lines for the SAME
    stimulus share a colour."""
    stripped = label.replace("stim ", "").replace("%", "").strip()
    if stripped.startswith("50"):
        return "50"
    if stripped.startswith("100"):
        return "100"
    if stripped.startswith("0"):
        return "0"
    raise ValueError(f"unrecognised stimulus label {label!r}")


_OUTPUT_OTHER_COLOURS = {"actor": "#333333", "critic": "#888888"}


def draw_individual_head_rates(model_type, ax=None, seeds=None):
    """Weight-change RATE (per update, mean +/- SEM across recovered seeds),
    one line per individual INPUT unit (the 3 stim features) and per
    individual OUTPUT unit (actor, critic, and -- where present -- the 3
    stim outputs), plus ONE aggregate line for the whole recurrent (h2h)
    block, across the FULL pre+reversal trajectory (real dense checkpoints
    now exist through reversal too -- see full_weight_change_trajectory).
    Each input/output line is the Frobenius norm of that one unit's whole
    weight vector (all N_HIDDEN entries), NOT a per-hidden-unit value --
    see module note above for why that makes seed-averaging valid here.
    Input lines are solid, output lines dashed, recurrent dotted;
    input/output lines for the SAME physical stimulus share a colour
    (style.STIM_COLOURS) so e.g. "in: stim 50%" and "out: stim 50%" are
    visually paired. rl_only has no stim_head, so it shows only 3 input +
    2 output (actor, critic) + 1 recurrent = 6 lines instead of 9.
    """
    fig, ax, _ = new_panel(ax, figsize=(7.2, 4.8))
    keep = VV.recovered_seeds(model_type)
    seeds = seeds if seeds is not None else (sorted(keep) if keep else CW.SEEDS)
    reversal_x = VV._reversal_x(model_type, keep=set(seeds), units="trials")

    stim_inputs = [(i, lbl) for i, lbl in enumerate(CG.INPUT_LABELS) if lbl.startswith("stim ")]
    output_specs = [("vigour_head.weight", 0, "actor"), ("value_head.weight", 0, "critic")]
    output_specs += [("stim_head.weight", ri, lbl) for ri, lbl in enumerate(CG.STIM_HEAD_ROW_LABELS)]

    n_seeds_used = 0
    x_min, x_max = None, None

    def _plot_series(label, color, linestyle, extractor):
        nonlocal n_seeds_used, x_min, x_max
        result = _collect_seed_series(model_type, seeds, extractor)
        if result is None:
            return
        t_pre, t_post, arr_pre, arr_post = result
        n_seeds_used = max(n_seeds_used, arr_pre.shape[0])
        mean_pre, sem_pre = _mean_sem_rows(arr_pre)
        mean_post, sem_post = _mean_sem_rows(arr_post)
        t_full = np.concatenate([t_pre, t_post + reversal_x])
        mean = np.concatenate([mean_pre, mean_post])
        sem = np.concatenate([sem_pre, sem_post])
        ax.plot(t_full, mean, color=color, lw=1.8, linestyle=linestyle, label=label)
        ax.fill_between(t_full, mean - sem, mean + sem, color=color, alpha=0.15, lw=0)
        x_min = t_full[0] if x_min is None else min(x_min, t_full[0])
        x_max = t_full[-1] if x_max is None else max(x_max, t_full[-1])

    for idx, lbl in stim_inputs:
        _plot_series(f"in: {lbl}", S.STIM_COLOURS[_stim_key(lbl)], "-",
                     lambda tr, idx=idx: tr["input_col_delta_rate"][:, idx])
    for hk, ri, lbl in output_specs:
        colour = _OUTPUT_OTHER_COLOURS.get(lbl) or S.STIM_COLOURS[_stim_key(lbl)]
        _plot_series(
            f"out: {lbl}", colour, "--",
            lambda tr, hk=hk, ri=ri: (tr["head_row_delta_rate"][hk][:, ri]
                                       if hk in tr["head_row_delta_rate"] else None))
    _plot_series("recurrent (h2h, aggregate)", LAYER_COLOURS["h2h"], ":",
                 lambda tr: tr["layer_delta_rate"]["h2h"])

    if n_seeds_used == 0:
        raise FileNotFoundError(f"no pre+reversal checkpoints found for {model_type}")

    ax.axvline(reversal_x, color="0.2", linestyle=":", lw=1.5)
    ax.set_xlabel("trials")
    ax.set_ylabel("weight change rate\n(Frobenius norm, per input/output unit)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=2)
    label = FC.MODELS.get(model_type, {}).get("label", model_type)
    ax.set_title(f"{label}: individual input/output weight-change rates, full pre+reversal "
                 f"trajectory [n={n_seeds_used} seeds]\n"
                 f"(solid=input, dashed=output, dotted=recurrent aggregate; paired colours = same stimulus)")
    if x_min is not None:
        ax.set_xlim(x_min, x_max)
    return fig

def build_all(show_tag=None):
    for mt in MODEL_TYPES:
        try:
            fig = draw_weight_change_layers(mt)
        except FileNotFoundError as e:
            print(f"  (skip weight-change layers for {mt}: {e})")
        else:
            save_panel(fig, "Mechanistic/WeightChange", f"MECH.WC.layers.{mt}",
                       f"{mt}_weight_change_layers", show_tag)

        try:
            fig = draw_weight_change_readout_split(mt)
        except FileNotFoundError as e:
            print(f"  (skip weight-change readout split for {mt}: {e})")
        else:
            save_panel(fig, "Mechanistic/WeightChange", f"MECH.WC.readout.{mt}",
                       f"{mt}_weight_change_readout_split", show_tag)

    for mt in MODEL_TYPES:
        try:
            fig = draw_weight_change_full(mt)
        except Exception as e:
            print(f"  (skip full pre+post weight-change trajectory for {mt}: {e})")
            continue
        save_panel(fig, "Mechanistic/WeightChange", f"MECH.WC.full.{mt}",
                   f"{mt}_weight_change_full", show_tag)

    for rev_tag in ("", "_5k"):
        try:
            fig = draw_reversal_delta_bars(rev_tag=rev_tag)
        except Exception as e:
            print(f"  (skip reversal delta bars [{rev_tag}]: {e})")
        else:
            save_panel(fig, "Mechanistic/WeightChange", f"MECH.WC.reversaldelta{rev_tag}",
                       f"reversal_weight_delta_bars{rev_tag}", show_tag)

    for mt in MODEL_TYPES:
        try:
            fig = draw_individual_head_rates(mt)
        except Exception as e:
            print(f"  (skip individual input/output weight-change rates for {mt}: {e})")
        else:
            save_panel(fig, "Mechanistic/WeightChange", f"MECH.WIND.{mt}",
                       f"{mt}_individual_head_rates", show_tag)


if __name__ == "__main__":
    build_all()

