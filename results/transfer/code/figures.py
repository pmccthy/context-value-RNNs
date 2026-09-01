#!/usr/bin/env python
"""Figure library for the final 3s1c models — reads ONLY figure_data.pkl (no models,
no cxval). Importable for interactive use (see the companion notebook) and runnable
as a script to regenerate every figure.

Design
------
* One model registry (`MODELS`) is the single source for each model's colour + label.
* Small REUSABLE primitives that take the data + which model(s) to plot and draw into
  an Axes (so they compose and can be reused in any context):
      bar_metric(D, metric, model_types=..., ax=...)      one grouped-bar panel
      heatmap_seed(D, model_type, seed, ax=...)           one model+seed tuning heatmap
      group_grid(D, model_type, ...)                      responsiveness-group PSTH grid
* Thin WRAPPERS that save publication files (PNG preview + PDF vector):
      fig_bar_metric(...)        combined + per-model panels, shared y-axis
      fig_heatmap_montage(...)   per-seed heatmaps tiled for one model
      fig_group_grid(...)        the cell-type grid for one model

Style is set by an editable matplotlib style sheet (--style).

Usage:
  python scripts/16_06_26_figures.py --data results/16_06_26_final/figure_data \
         --out results/16_06_26_final/figures
"""
from __future__ import annotations
import argparse, pickle
from pathlib import Path
import numpy as np
import matplotlib; import matplotlib.pyplot as plt   # backend set to Agg in main() (script use)

# Colours + labels live in the editable config module (figure_config.py); fonts/sizes
# live in the matplotlib style sheet. figure_config sits next to this file, so it is
# importable both as a script (`python figures.py`) and from the notebook.
from figure_config import MODELS, STIM_COLORS, STIM_LABELS

DEFAULT_STYLE = Path(__file__).with_name("figure_style.mplstyle")
VEC = "pdf"                                          # vector format for Illustrator


# ── io ────────────────────────────────────────────────────────────────────────
def load(data_dir):
    """Load the figure_data.pkl dict."""
    with open(Path(data_dir) / "figure_data.pkl", "rb") as f:
        return pickle.load(f)


def save_fig(fig, outdir, name):
    """Save a figure as PNG (preview) + PDF (vector)."""
    base = Path(outdir) / name
    fig.savefig(f"{base}.png", bbox_inches="tight")
    fig.savefig(f"{base}.{VEC}", bbox_inches="tight")
    plt.close(fig); print(f"  -> {name}.png / .{VEC}")


def _present(D, model_types):
    """Default to the model types actually present in the data, in registry order."""
    avail = [m for m in MODELS if m in D["model_types"]]
    return [m for m in (model_types or avail) if m in D["model_types"]]


def tuning_vmax(D, model_types=None, pct=99):
    """A COMMON heatmap colour ceiling across the given model types, so their
    per-unit tuning heatmaps are directly comparable. Pass the result as `vmax` to
    heatmap_seed / heatmap_montage (make_all uses this across all models by default)."""
    ti = {m: i for i, m in enumerate(D["model_types"])}
    data = np.concatenate([D["tuning"]["data"][ti[m]].reshape(-1)
                           for m in _present(D, model_types)])
    return float(np.nanpercentile(data, pct))


# ══════════════════════════════════════════════════════════════════════════════
# REUSABLE PRIMITIVES  (draw into an Axes; specify which data + which models)
# ══════════════════════════════════════════════════════════════════════════════
def bar_metric(D, metric, *, model_types=None, ax=None, ylim=None, ref=None,
               ylabel=None, title=None, annotate=False, legend_fontsize=13):
    """Grouped bars of a scalar `metric` per stimulus, mean ± SEM across seeds, one
    bar group per stimulus and one coloured bar per model. Colours come from MODELS.

    Args:
        D:           loaded figure_data dict.
        metric:      key in D['scalars'] (e.g. 'n_responsive', 'vigour').
        model_types: which models to draw (default: all present, registry order).
        ax:          draw into this Axes (created if None).
        ylim, ref:   optional y-limits and a dashed reference line.
        ylabel,title:optional labels (default from metric/units).
        annotate:    if True, write each bar's value above it. For 'n_responsive' this
                     is the neuron count plus its percentage of all units.
        legend_fontsize: legend text size (kept small so it doesn't cover the bars).
    Returns: (fig, ax).
    """
    S = D["scalars"]; stim = S["coords"]["stim"]; x = np.arange(len(stim))
    arr = S[metric]; ti = {m: i for i, m in enumerate(D["model_types"])}
    mts = _present(D, model_types)
    H = D["n_units"]
    w = 0.8 / max(len(mts), 1)
    fig = ax.figure if ax is not None else plt.figure(figsize=(7.0, 5.0))
    ax = ax or fig.add_subplot(111)
    for i, m in enumerate(mts):
        a = arr[ti[m]]                                     # (seed, stim)
        n = int(np.isfinite(a[:, 0]).sum())
        mean = np.nanmean(a, 0); sem = np.nanstd(a, 0) / np.sqrt(max(n, 1))
        xpos = x + (i - (len(mts) - 1) / 2) * w
        ax.bar(xpos, mean, w, yerr=sem, capsize=4,
               color=MODELS[m]["color"], edgecolor="white", alpha=0.9,
               label=f"{MODELS[m]['label']} (n={n})")
        if annotate:                                       # neuron count (+ % of units)
            for xp, mv, sv in zip(xpos, mean, sem):
                txt = (f"{mv:.0f}\n{100*mv/H:.0f}%" if metric == "n_responsive"
                       else f"{mv:.2f}")
                ax.text(xp, mv + sv, txt, ha="center", va="bottom", fontsize=10,
                        linespacing=0.9)
    if ref is not None:
        ax.axhline(ref, color="0.6", lw=1, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(stim)
    ax.set_xlim(-0.5, len(stim) - 0.5); ax.margins(x=0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel or S["units"].get(metric, metric))
    if title:
        ax.set_title(title + (f" — {MODELS[mts[0]]['label']}" if len(mts) == 1 else ""))
    if len(mts) > 1:
        ax.legend(fontsize=legend_fontsize, loc="upper left", framealpha=0.85,
                  borderaxespad=0.3, handlelength=1.2, handletextpad=0.5)
    return fig, ax


def heatmap_seed(D, model_type, seed, *, ax=None, vmax=None, sort=True,
                 xtick_fontsize=None, ylabel="hidden unit", title=None):
    """Per-unit stim-window tuning heatmap (stimulus × unit) for ONE model + seed,
    units sorted by preferred stimulus. The y-axis is the hidden units.

    Args:
        ax:             draw into this Axes (created, sized for a standalone plot, if None).
        vmax:           shared colour ceiling (default: this model+seed's max).
        xtick_fontsize: stimulus-label size (small in montages to avoid overlap).
        ylabel:         y-axis label (None to omit, e.g. inside a montage).
        title:          panel title (default "seed <NN>").
    Returns: (fig, ax, im), or (None, None, None) if that (model, seed) is absent.
    """
    ti = D["model_types"].index(model_type)
    if seed not in D["seeds"]:
        return None, None, None
    si = D["seeds"].index(seed)
    md = D["tuning"]["data"][ti, si]                       # (3, H)
    if not np.isfinite(md).all():
        return None, None, None
    stim = D["tuning"]["coords"]["stim"]
    if sort:
        peak = md.max(0); pref = md.argmax(0); silent = peak < 0.05 * peak.max()
        order = np.lexsort((-peak, np.where(silent, 3, pref)))
    else:
        order = np.arange(md.shape[1])
    standalone = ax is None
    fig = ax.figure if ax is not None else plt.figure(figsize=(4.3, 4.2))
    ax = ax or fig.add_subplot(111)
    im = ax.imshow(md[:, order].T, aspect="auto", cmap="hot", vmin=0, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(len(stim)))
    ax.set_xticklabels(stim, **({"fontsize": xtick_fontsize} if xtick_fontsize else {}))
    if ylabel:                                            # mark the hidden-unit axis
        ax.set_ylabel((ylabel + " (sorted)") if standalone else ylabel)
    else:
        ax.set_yticks([])
    ax.set_title(title if title is not None else f"seed {seed}",
                 fontsize=None if standalone else 9)
    if standalone:                                       # single-seed plot gets its own colorbar
        fig.colorbar(im, ax=ax, label="mean activation", shrink=0.85)
    return fig, ax, im


# responsiveness-pattern -> cell-type group, laid out lower-triangular
GROUPS = {(0,): "0%-only", (1,): "50%-only", (2,): "100%-only",
          (0, 1): "0% & 50%", (0, 2): "0% & 100%", (1, 2): "50% & 100%",
          (0, 1, 2): "all three"}
CELL = {(0, 0): (0,), (1, 1): (1,), (2, 2): (2,),
        (1, 0): (0, 1), (2, 0): (0, 2), (2, 1): (1, 2), (0, 2): (0, 1, 2)}


def _group_of(pattern):
    members = tuple(np.where(pattern)[0].tolist())
    return members if members in GROUPS else None


def _pool_units(D, model_type):
    """Pool per-unit responsiveness patterns + trial-aligned traces over a model's
    seeds. Returns pats (N,3) bool, traces (N, n_align, 3), or (None, None)."""
    ti = D["model_types"].index(model_type)
    resp = D["responsive"]["data"][ti]                    # (seed, unit, stim)
    aligned = D["aligned_mean"]["data"][ti]               # (seed, unit, time, stim)
    pats, traces = [], []
    for si in range(resp.shape[0]):
        if not np.isfinite(aligned[si]).all():            # absent seed -> skip
            continue
        pats.append(resp[si]); traces.append(aligned[si])
    if not pats:
        return None, None
    return np.concatenate(pats, 0), np.concatenate(traces, 0)


def group_grid_ylim(D, model_types=None, pad=0.05):
    """A COMMON y-range for the trial-aligned group grids across the given models,
    scaled to the largest-magnitude group of ANY model, so the per-model grids are
    directly comparable. Pass the result as `ylim=` to group_grid / fig_group_grid
    (make_all uses this across all models by default)."""
    hi = lo = 0.0
    for m in _present(D, model_types):
        pats, traces = _pool_units(D, m)
        if pats is None:
            continue
        for key in CELL.values():
            members = np.array([_group_of(p) == key for p in pats])
            if members.sum() == 0:
                continue
            g = traces[members]; gm = g.mean(0); gsem = g.std(0) / np.sqrt(g.shape[0])
            hi = max(hi, float((gm + gsem).max())); lo = min(lo, float((gm - gsem).min()))
    p = pad * max(hi - lo, 1e-9)
    return (lo - p, hi + p)


def group_grid(D, model_type, *, fig=None, show_counts=True, label_fontsize=16, ylim=None):
    """Draw the lower-triangular responsiveness-group grid for ONE model: each panel
    is a group's mean trial-aligned activation (pooled over units & seeds) to each
    stimulus, across ITI -> stim -> outcome. Returns the Figure (or None).

    Args:
        show_counts:    if True, each panel title includes the group's neuron count and
                        its percentage of all units.
        label_fontsize: size of the x-tick (ITI/stim/outcome) and y-axis labels.
        ylim:           shared y-limits for every panel. If None (default for a lone
                        call) it is scaled to THIS model's largest group; pass
                        group_grid_ylim(D) to share one scale across all models.
    """
    pats, traces = _pool_units(D, model_type)
    if pats is None:
        return None
    n_iti, stim_ts, rew_ts = (D["period"][k] for k in ("n_iti_pre", "stim_ts", "rew_ts"))
    n_total = n_iti + stim_ts + rew_ts; x = np.arange(n_total); n_units = len(pats)
    stim = D["stim_labels"]
    # precompute each group's mean ± SEM trace, and a SHARED y-range scaled to the
    # largest-magnitude group, so every panel uses the same y-axis.
    panels = {}; hi = lo = 0.0
    for key in CELL.values():
        members = np.array([_group_of(p) == key for p in pats])
        if members.sum() > 0:
            g = traces[members]
            gm = g.mean(0); gsem = g.std(0) / np.sqrt(g.shape[0])
            panels[key] = (members, gm, gsem)
            hi = max(hi, float((gm + gsem).max())); lo = min(lo, float((gm - gsem).min()))
        else:
            panels[key] = (members, None, None)
    if ylim is None:                                 # default: scale to THIS model
        pad = 0.05 * max(hi - lo, 1e-9); ylim = (lo - pad, hi + pad)
    fig = fig or plt.figure(figsize=(12, 11))
    axes = fig.subplots(3, 3, sharex=True)
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            if (i, j) not in CELL:
                ax.axis("off"); continue
            key = CELL[(i, j)]
            members, gm, gsem = panels[key]
            # mark the ITI|stim and stim|outcome boundaries with dotted lines only
            # (no block shading, per request)
            ax.axvline(n_iti - 0.5, color="0.6", lw=0.8, ls=":")
            ax.axvline(n_iti + stim_ts - 0.5, color="0.6", lw=0.8, ls=":")
            if gm is not None:
                for s in range(3):
                    ax.fill_between(x, gm[:, s] - gsem[:, s], gm[:, s] + gsem[:, s],
                                    color=STIM_COLORS[s], alpha=0.2, lw=0)
                    ax.plot(x, gm[:, s], "-", color=STIM_COLORS[s], lw=2.0, alpha=0.95)
            ax.set_xlim(0, n_total - 1); ax.margins(x=0)     # tight: no L/R gap
            ax.set_ylim(*ylim)                               # shared y-axis across panels
            frac = 100.0 * members.sum() / max(n_units, 1)
            name = GROUPS[key] + (f"   n={int(members.sum())} ({frac:.1f}%)" if show_counts else "")
            ax.set_title(name, fontsize=12)
            ax.tick_params(labelsize=label_fontsize)
            if j == 0:
                ax.set_ylabel("mean activation", fontsize=label_fontsize)
            if i == 2:
                ax.set_xticks([n_iti / 2 - 0.5, n_iti + stim_ts / 2 - 0.5,
                               n_iti + stim_ts + rew_ts / 2 - 0.5])
                ax.set_xticklabels(D["period"]["segments"], fontsize=label_fontsize)
    leg = axes[0, 1]; leg.axis("off")
    leg.legend(handles=[plt.Line2D([0], [0], color=STIM_COLORS[s], lw=2.5, label=stim[s])
                        for s in range(3)], loc="center", fontsize=15, frameon=False,
               title="stimulus", title_fontsize=13)
    fig.suptitle(f"Functional cell-type groups & trial-aligned responses — "
                 f"{MODELS[model_type]['label']}\nlower-triangular: exclusives on diagonal, "
                 f"pairs below, all-three top-right  ·  pooled over seeds (n={n_units} units)",
                 y=0.99, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# responder groups (exclusives -> pairwise-mixed -> all-three), and their bit codes
# for [0%, 50%, 100%] membership (1|2|4)
GROUP_ORDER = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
_GROUP_CODES = [1, 2, 4, 3, 5, 6, 7]


def responder_group_counts(D, model_type):
    """Per-seed counts of units in each responder group (GROUP_ORDER) for one model.
    A unit's group is the set of stimuli it significantly responds to. Returns
    (n_present_seed, 7); only seeds actually present for the model are included."""
    ti = D["model_types"].index(model_type)
    resp = D["responsive"]["data"][ti]                       # (seed, unit, stim) bool
    seed_idx = [D["seeds"].index(s) for s in D["seeds_present"][model_type]]
    bits = np.array([1, 2, 4])
    rows = [[int((resp[si].astype(int) @ bits == c).sum()) for c in _GROUP_CODES]
            for si in seed_idx]
    return np.array(rows, float) if rows else np.zeros((0, len(_GROUP_CODES)))


def bar_responder_groups(D, *, model_types=None, ax=None, fraction=False,
                         ylabel=None, title="Responder groups", legend_fontsize=13):
    """Grouped bars of the number (or %) of units in each responder group — the three
    exclusive types, the three pairwise-mixed types and the all-three type — one
    coloured bar per model, mean ± SEM across seeds. Returns (fig, ax)."""
    mts = _present(D, model_types)
    labels = [GROUPS[k] for k in GROUP_ORDER]; x = np.arange(len(GROUP_ORDER))
    w = 0.8 / max(len(mts), 1); H = D["n_units"]
    fig = ax.figure if ax is not None else plt.figure(figsize=(11, 5.2))
    ax = ax or fig.add_subplot(111)
    for i, m in enumerate(mts):
        c = responder_group_counts(D, m)
        if fraction:
            c = 100.0 * c / H
        n = c.shape[0]
        mean = c.mean(0) if n else np.zeros(len(x))
        sem = c.std(0) / np.sqrt(max(n, 1)) if n else np.zeros(len(x))
        ax.bar(x + (i - (len(mts) - 1) / 2) * w, mean, w, yerr=sem, capsize=3,
               color=MODELS[m]["color"], edgecolor="white", alpha=0.9,
               label=f"{MODELS[m]['label']} (n={n})")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.tick_params(axis="x", labelsize=14); ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylabel(ylabel or ("% of units" if fraction else "# units"))
    ax.set_title(title)
    ax.legend(fontsize=legend_fontsize)
    fig.tight_layout()
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# WRAPPERS  (save publication files)
# ══════════════════════════════════════════════════════════════════════════════
def fig_bar_metric(D, metric, *, outdir, stem, ylabel=None, title=None, ref=None,
                   model_types=None, annotate=False):
    """Save a metric as combined + one-per-model panels, all sharing a y-axis
    (derived from the metric across the chosen models, plus headroom). `annotate`
    writes the neuron count (+ %) above each bar."""
    mts = _present(D, model_types)
    top = float(np.nanmax(D["scalars"][metric]))
    factor = 1.38 if len(mts) > 1 else 1.15      # extra headroom so the legend clears the bars
    if annotate:
        factor = max(factor, 1.30)
    ymax = (ref if (ref is not None and ref > top) else top) * factor
    ylim = (0.0, ymax)
    fig, _ = bar_metric(D, metric, model_types=mts, ylim=ylim, ref=ref,
                        ylabel=ylabel, title=title, annotate=annotate)
    save_fig(fig, outdir, stem)
    for m in mts:
        fig, _ = bar_metric(D, metric, model_types=[m], ylim=ylim, ref=ref,
                            ylabel=ylabel, title=title, annotate=annotate)
        save_fig(fig, outdir, f"{stem}_{m}")


def heatmap_montage(D, model_type, *, vmax=None, ncol=6):
    """Tile heatmap_seed across all of a model's seeds; returns the Figure (or None).
    Shared colour scale (vmax) across panels."""
    ti = D["model_types"].index(model_type)
    seeds = [s for s in D["seeds"]
             if np.isfinite(D["tuning"]["data"][ti, D["seeds"].index(s)]).all()]
    if not seeds:
        return None
    if vmax is None:
        vmax = float(np.nanpercentile(D["tuning"]["data"][ti], 99))
    nrow = int(np.ceil(len(seeds) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.8 * ncol, 1.7 * nrow), squeeze=False,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.25})
    for ax in axes.flat:
        ax.axis("off")
    im = None
    for k, seed in enumerate(seeds):
        ax = axes[k // ncol][k % ncol]; ax.axis("on")
        # small stimulus labels (large ones overlap in the tiny panels); no y-label here
        _, _, im = heatmap_seed(D, model_type, seed, ax=ax, vmax=vmax,
                                xtick_fontsize=8, ylabel=None)
        if (k + ncol) < len(seeds):                       # x-labels on bottom row only
            ax.set_xticks([])
    fig.colorbar(im, ax=axes, shrink=0.5, label="mean activation")
    fig.suptitle(f"Per-unit stim-window activation (sorted by preferred stim) — "
                 f"{MODELS[model_type]['label']}  [{len(seeds)} seeds]", y=1.0, fontsize=13)
    return fig


def fig_heatmap_montage(D, model_type, *, outdir, vmax=None, ncol=6):
    """Save the per-seed heatmap montage for one model."""
    fig = heatmap_montage(D, model_type, vmax=vmax, ncol=ncol)
    if fig is not None:
        save_fig(fig, outdir, f"activation_heatmaps_{model_type}")


def fig_group_grid(D, model_type, *, outdir, show_counts=True, ylim=None):
    """Save the responsiveness-group trial-aligned grid for one model. Pass `ylim`
    (e.g. group_grid_ylim(D)) to share one y-axis across models."""
    fig = group_grid(D, model_type, show_counts=show_counts, ylim=ylim)
    if fig is not None:
        save_fig(fig, outdir, f"celltype_group_grid_{model_type}")


def fig_heatmap_seed(D, model_type, seed, *, outdir, vmax=None):
    """Save a SINGLE per-unit tuning heatmap for one model + seed (hidden units on y)."""
    fig, _, _ = heatmap_seed(D, model_type, seed, vmax=vmax)
    if fig is not None:
        save_fig(fig, outdir, f"heatmap_{model_type}_seed{seed}")


def fig_responder_groups(D, *, outdir, fraction=False):
    """Save the responder-group bar plot (units per group, one coloured bar per model)."""
    fig, _ = bar_responder_groups(D, fraction=fraction)
    save_fig(fig, outdir, "responder_groups" + ("_frac" if fraction else ""))


# ══════════════════════════════════════════════════════════════════════════════
def make_all(D, outdir, *, annotate=False, show_counts=True, single_seed=42,
             shared_grid_ylim=True):
    """Regenerate every figure from the data.

    Args:
        annotate:        write the neuron count (+ %) above each bar in the count plot.
        show_counts:     include each group's neuron count + % in the group-grid panels.
        single_seed:     seed to use for the single per-model heatmap (None to skip).
        shared_grid_ylim: if True (default), every model's trial-aligned grid uses ONE
                         y-axis, scaled to the highest-magnitude model, so they are
                         directly comparable; if False, each model self-scales.
    """
    # bar metrics (combined + per-model, shared y-axis)
    fig_bar_metric(D, "n_responsive", outdir=outdir, stem="responsive_per_stim",
                   ylabel="# responsive units", title="Significantly responsive units",
                   annotate=annotate)
    fig_bar_metric(D, "pop_activity", outdir=outdir, stem="population_activity_per_stim",
                   ylabel="mean activation", title="Population activity")
    fig_bar_metric(D, "vigour", outdir=outdir, stem="vigour_per_stim",
                   ylabel="mean vigour", title="Lick vigour")
    fig_responder_groups(D, outdir=outdir)         # units per responder group, per model
    # per-model heatmaps (montage + one single seed) on a COMMON colour scale, + group grids
    vmax = tuning_vmax(D)                          # shared so the heatmaps are comparable
    grid_ylim = group_grid_ylim(D) if shared_grid_ylim else None   # shared grid y-axis
    for m in _present(D, None):
        fig_heatmap_montage(D, m, outdir=outdir, vmax=vmax)
        fig_group_grid(D, m, outdir=outdir, show_counts=show_counts, ylim=grid_ylim)
        if single_seed is not None:                # one representative single-seed heatmap
            fig_heatmap_seed(D, m, single_seed, outdir=outdir, vmax=vmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/16_06_26_final/figure_data")
    ap.add_argument("--out", default="results/16_06_26_final/figures")
    ap.add_argument("--style", default=str(DEFAULT_STYLE))
    ap.add_argument("--annotate", action="store_true",
                    help="write neuron count + %% above each bar in the responder plot")
    ap.add_argument("--no-counts", action="store_true",
                    help="hide the per-group neuron count/%% in the group-grid titles")
    ap.add_argument("--per-model-grid-ylim", action="store_true",
                    help="self-scale each model's trial-aligned grid (default: one shared "
                         "y-axis across models, scaled to the highest-magnitude model)")
    args = ap.parse_args()
    matplotlib.use("Agg")                                # headless file output for the script
    if Path(args.style).exists():
        plt.style.use(args.style); print(f"style: {args.style}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    make_all(load(args.data), out, annotate=args.annotate, show_counts=not args.no_counts,
             shared_grid_ylim=not args.per_model_grid_ylim)


if __name__ == "__main__":
    main()
