#!/usr/bin/env python3
"""Compose individual panels into multi-panel figures, following the exact
machinery (and visual conventions -- panel tags/letters, one combined
deduped legend, shared gridspec, no per-row title repetition, NO font
shrinking -- fonts are the real transfer.mplstyle scale throughout) of
~/Documents/neuronal-representations/results/transfer/figures/compose.py.

    python compose.py
    SHOW_TAG=0 python compose.py   # hide the per-panel tag labels

Font sizes: model_figure_style.mplstyle (applied globally by panels/_tags.py
at import time) already ports transfer.mplstyle's real values verbatim
(axes.labelsize 22, xtick/ytick.labelsize 18, legend.fontsize 14,
axes.titlesize 12) -- there is NO separate "compact" font context here
any more. Like the real compose.py, collision at these large sizes is
avoided by (a) generous per-panel width/height (not a shrunk font), (b)
manual hspace/wspace (no constrained_layout, which fights large fixed-size
tick labels), and (c) dropping repeated per-row titles in favour of one
column header per model (see _strip_row_titles below).
"""
from __future__ import annotations

import os
import string
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "panels"))
sys.path.insert(0, str(_HERE / "style"))
sys.path.insert(0, str(_HERE / "analysis"))
sys.path.insert(0, str(_HERE / "transfer" / "code"))
sys.path.insert(0, str(_HERE / "cross_model_vs_experiment"))

import _tags
_tags.SHOW_TAG = os.environ.get("SHOW_TAG", "1") not in ("0", "false", "False")
from _tags import FIG_ROOT, finalize_axes  # noqa: E402

import transfer as T          # noqa: E402
import reversal_fine as RF    # noqa: E402
import reversal_broad as RB   # noqa: E402
import chi2_bars as C2        # noqa: E402
import chi2_diagnostics as CD  # noqa: E402
import decoding as DEC        # noqa: E402
import decoding_timeresolved as DTR  # noqa: E402
import population_timeresolved as PT  # noqa: E402
import vigour_value as VV     # noqa: E402
import figures as F           # noqa: E402
import model_responders as MR  # noqa: E402
import _method                 # noqa: E402
import group_sizes as GS       # noqa: E402
import reversal_gradient_bars as RGB  # noqa: E402
import population_pca_panels  # noqa: E402
import population_tdr_panels  # noqa: E402
import coding_angle_panel as CANG  # noqa: E402
import weight_change as WCH    # noqa: E402
import terminal_rpe as TRPE    # noqa: E402
import circuit_diagram as CDG  # noqa: E402
import rpe_proxy as RPEP  # noqa: E402
import weight_matrix_panel as WMAT  # noqa: E402
import gradient_loss_panels as GLP  # noqa: E402
import coding_angle_comparison as CACMP  # noqa: E402
import noise_robustness as NR  # noqa: E402

OUT = FIG_ROOT / "composites"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_TYPES = ["rl_only", "classif_rl", "classif_rl_readout_only"]

# "" = build from the 2500-trial reversal run, "_5k" = the longer one.
# Every module here (vigour_value, reversal_broad/fine, decoding) reads the
# SAME env var, so one invocation is internally consistent:
#   REV_TAG=_5k python make_panels.py && REV_TAG=_5k python compose.py
REV_TAG = os.environ.get("REV_TAG", "")


def _panel(fig, gs, r, c, tag, fn, letter=None):
    ax = fig.add_subplot(gs[r, c])
    fn(ax=ax)
    if letter is not None:
        ax.text(-0.1, 1.22, letter, transform=ax.transAxes, fontsize=30,
                fontweight="bold", va="top", ha="left")
    elif _tags.SHOW_TAG:
        ax.text(0.015, 0.985, tag, transform=ax.transAxes, fontsize=6, va="top",
                ha="left", color="0.35", family="monospace", zorder=1000)
    return ax


def _panel_letter(idx: int) -> str:
    """Excel-style lowercase panel label: a, b, ..., z, aa, ab, ... -- plain
    string.ascii_lowercase[idx] overflows past 26 panels, which
    figure_mechanistic_overview now exceeds (12 rows x 3 cols = 36)."""
    idx += 1
    letters = ""
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        letters = string.ascii_lowercase[rem] + letters
    return letters


def _build_grid(fig, gs, rows, letters):
    idx = 0
    for r, row in enumerate(rows):
        for c, (tag, fn) in enumerate(row):
            _panel(fig, gs, r, c, tag, fn,
                   letter=_panel_letter(idx) if letters else None)
            idx += 1


def _column_headers(fig, model_types, n_cols):
    """Plain fig.text() header above each column naming its model -- used
    instead of a per-axes title (which _save()'s finalize_axes now strips
    from every composite, matching the real repo's own convention of never
    showing an axes title) so a cross-model grid like MODELFIG1 doesn't lose
    which column is which model. Reads each column's actual axes position so
    it stays aligned regardless of gridspec width_ratios."""
    for c, mt in enumerate(model_types[:n_cols]):
        bbox = fig.axes[c].get_position()
        xc = (bbox.x0 + bbox.x1) / 2
        fig.text(xc, 0.985, F.MODELS[mt]["label"], ha="center", va="top",
                 fontsize=20)


def _row_headers(fig, model_types, n_cols):
    """Plain fig.text() header to the left of each row naming its model --
    the row-major counterpart of _column_headers, for composites where MODEL
    is the row dimension instead of the column dimension."""
    for r, mt in enumerate(model_types):
        bbox = fig.axes[r * n_cols].get_position()
        yc = (bbox.y0 + bbox.y1) / 2
        fig.text(0.005, yc, F.MODELS[mt]["label"], ha="left", va="center",
                 fontsize=18, rotation=90)


def _dedupe_row_ylabels(fig, n_rows, n_cols=3):
    """Blank the y-axis label on every column except the first in a plain
    r*n_cols-major grid of single-Axes cells (no interleaved colorbar axes --
    only safe to call on composites where every panel fn was called with
    colorbar=False / defaults to no own colorbar when given an ax, as is the
    case for every panel in figure_mechanistic_overview). Every column in a
    row shares the same y-axis meaning (same metric, different model_type),
    so repeating the label 3x per row was pure clutter."""
    axes = fig.axes
    for r in range(n_rows):
        for c in range(1, n_cols):
            i = r * n_cols + c
            if i < len(axes):
                axes[i].set_ylabel("")


def _harmonize_row_ylim(fig, row_indices, n_cols=3):
    """Force every axes in each given (0-indexed) row to share the same
    y-limits -- the union of that row's own per-column auto-scaled limits
    -- so a difference in a mechanistic quantity's SCALE across model types
    isn't visually confused with a difference in its column layout. Only
    meaningful for rows of ordinary line-plot Axes (real y data, shared
    units across columns); heatmap and circuit-diagram rows are excluded by
    the caller since their "y-axis" isn't a comparable quantity. Relies on
    the same plain r*n_cols-major fig.axes indexing as _dedupe_row_ylabels
    (safe here for the same reason: no panel in this grid owns a colorbar
    axes of its own)."""
    axes = fig.axes
    for r in row_indices:
        row_axes = [axes[r * n_cols + c] for c in range(n_cols) if r * n_cols + c < len(axes)]
        if not row_axes:
            continue
        lims = [ax.get_ylim() for ax in row_axes]
        lo = min(l[0] for l in lims)
        hi = max(l[1] for l in lims)
        for ax in row_axes:
            ax.set_ylim(lo, hi)


def _clear_heatmap_titles(fig):
    """Explicitly blank the title on every image-bearing axes (heatmaps) in
    the figure -- used where a composite wants a heatmap's own per-panel
    title gone even though finalize_axes(force_remove_titles=True)
    otherwise exempts image axes (see _save/finalize_axes) so heatmaps keep
    theirs everywhere else. Filters by ax.get_images() rather than a fixed
    row*n_cols slice, since each heatmap's own colorbar is a SEPARATE axes
    that fig.colorbar() appends to fig.axes right after it -- a positional
    slice silently drifts out of alignment as soon as more than one heatmap
    is in the grid."""
    for ax in fig.axes:
        if ax.get_images():
            ax.set_title("")


def _strip_row_titles(fig, n_cols, keep_row=0):
    """At real (18-22pt) font sizes a title on every one of e.g. 5 stacked
    rows repeats the same model name 5x down each column and collides with
    the axes above -- the real repo's own compose.py avoids this by
    dropping composite titles altogether (finalize_axes force_remove_titles
    =True) and relying on tags/letters instead. Here MODEL TYPE is the
    primary axis (columns), so keep exactly one title per column -- on
    `keep_row` -- as a de-facto column header, and clear the rest."""
    axes = fig.axes
    for i, ax in enumerate(axes):
        if ax.get_images():
            continue  # heatmaps keep their titles regardless
        row = i // n_cols
        if row != keep_row:
            ax.set_title("")


def _combine_legends(fig):
    seen = {}
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        for hh, ll in zip(h, l):
            if ll and not ll.startswith("_") and ll not in seen:
                seen[ll] = hh
    if seen:
        fig.legend(seen.values(), seen.keys(), loc="center left",
                   bbox_to_anchor=(0.99, 0.5), frameon=False, fontsize=15)


def _strip_all_legends(fig):
    """No-legend variant: remove every axes' own legend outright (no shared
    figure-level legend either)."""
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()


def _per_row_legends(fig, row_indices, n_cols=3, x_pad=0.012):
    """Per-row alternative to _combine_legends: instead of ONE shared legend
    for the whole figure (unwieldy once a composite has this many rows, each
    with its OWN legend vocabulary -- stim colours in one row, layer colours
    in another, loss-term colours in a third), give each line-plot row its
    own small legend in the right margin next to that row's own axes. Built
    from the union of unique (deduped, order-preserving) labels across that
    row's columns, so e.g. rl_only's 6-line WIND subset and classif_rl's
    9-line superset still produce one correct legend for that row. Strips
    each axes' own inline legend as it goes -- the row legend replaces it,
    it isn't in addition to it."""
    for r in row_indices:
        seen = {}
        row_axes = []
        for c in range(n_cols):
            i = r * n_cols + c
            if i >= len(fig.axes):
                continue
            ax = fig.axes[i]
            row_axes.append(ax)
            h, l = ax.get_legend_handles_labels()
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
            for hh, ll in zip(h, l):
                if ll and not ll.startswith("_") and ll not in seen:
                    seen[ll] = hh
        if not seen or not row_axes:
            continue
        bbox = row_axes[-1].get_position()   # rightmost column in this row
        yc = (bbox.y0 + bbox.y1) / 2
        fig.legend(seen.values(), seen.keys(), loc="center left",
                   bbox_to_anchor=(bbox.x1 + x_pad, yc), frameon=False, fontsize=9)


def _row_side_label(fig, ax, text, fontsize=20):
    """Big, non-bold, vertical left-margin row label positioned from a
    single given axes' own bbox -- the same visual weight as
    _column_headers' model-name headers (fontsize=20), used for row
    groupings (e.g. "pre-reversal" / "post-reversal") that _column_headers/
    _row_headers (which label MODEL, not phase) don't cover."""
    bbox = ax.get_position()
    yc = (bbox.y0 + bbox.y1) / 2
    fig.text(0.005, yc, text, ha="left", va="center", fontsize=fontsize, rotation=90)


def _add_weight_matrix_rows(fig, gs, row_pre, row_post, model_types, granularity="broad",
                            letters=False, start_idx=0):
    """Append the pre/post-reversal weight-MATRIX figures (weight_matrix_panel.
    draw_weight_matrix_by_subgroup -- the actual unit x unit recurrent
    matrix plus its input/output strips, unit-sorted by functional
    subgroup) as two more rows at given (already-reserved, larger-height)
    GridSpec row indices. Necessarily built OUTSIDE _build_grid's plain
    one-Axes-per-cell loop: each cell here is itself a nested 2x2 sub-grid
    (see draw_weight_matrix_by_subgroup's own subplot_spec.subgridspec
    call), not a single Axes, so calling it appends 3 Axes per cell (9 per
    row) onto the END of fig.axes -- harmless as long as every OTHER
    fig.axes-index-based helper here (_harmonize_row_ylim, _dedupe_row_
    ylabels) is called with the MAIN grid's own n_rows, not this extended
    total, so it never walks into these. NECESSARILY single-seed, like the
    circuit-diagram rows (see weight_matrix_panel's own module docstring for
    why) -- each cell is titled with its example seed by
    draw_weight_matrix_by_subgroup itself; that per-cell title is an IMAGE
    axes title (on ax_in), so _save()'s force_remove_titles=True alone
    doesn't drop it -- the caller must also call _clear_heatmap_titles(fig)
    AFTER this function returns (it can't be called here since it filters
    by ax.get_images() at call time, over whatever's already in fig.axes).
    A big vertical "pre-reversal"/"post-reversal" row label (_row_side_label,
    matching _column_headers' font size) replaces the old small bold
    horizontal one. letters/start_idx: when letters=True, stamps one
    Excel-style panel letter per (phase, model) cell -- continuing the
    sequence from start_idx (typically the calling composite's own
    n_rows * n_cols, i.e. right after the main grid's own last letter) --
    on each cell's top axes (ax_in), same visual convention as _panel()."""
    idx = start_idx
    for r, phase, row_label in [(row_pre, "pre", "pre-reversal"),
                                (row_post, "post", "post-reversal")]:
        for c, mt in enumerate(model_types):
            WMAT.draw_weight_matrix_by_subgroup(mt, phase=phase, granularity=granularity,
                                                fig=fig, subplot_spec=gs[r, c])
            if letters:
                # ax_in, ax_main, ax_out were just appended, in that order. ax_in
                # is a short strip (height_ratio 1.6 of 8.6 in draw_weight_matrix_
                # by_subgroup's own outer subgridspec), so a modest axes-fraction
                # offset in ITS OWN transAxes is a tiny absolute gap -- not enough
                # clearance for a 30pt letter, which then visually overlapped the
                # strip itself. Use a bigger offset.
                ax_in = fig.axes[-3]
                ax_in.text(-0.1, 2.3, _panel_letter(idx), transform=ax_in.transAxes,
                           fontsize=30, fontweight="bold", va="top", ha="left")
            idx += 1
        # Left-margin row label, vertically centred on the FULL cell (ax_in's
        # small top strip UNION ax_main's big bottom square -- together they
        # span the whole cell height; using ax_in's bbox alone, as before,
        # centred the label on just that short top strip instead of the
        # whole cell, same misalignment _row_side_label's own single-axes
        # convention avoids for the circuit-diagram rows).
        first_cell_axes = fig.axes[-(3 * len(model_types)):-(2 * len(model_types))]
        if len(first_cell_axes) >= 2:
            bbox_in = first_cell_axes[0].get_position()
            bbox_main = first_cell_axes[1].get_position()
            y0 = min(bbox_in.y0, bbox_main.y0)
            y1 = max(bbox_in.y1, bbox_main.y1)
            yc = (y0 + y1) / 2
            fig.text(0.005, yc, row_label, ha="left", va="center", fontsize=20, rotation=90)


def _save(fig, tag, name, legend=True, subdir=None):
    if legend:
        _combine_legends(fig)
    else:
        _strip_all_legends(fig)
    # Real repo's own _save() always force-removes titles from every
    # composite (relies on panel tags/letters + axis labels + one shared
    # legend instead) -- matched here on request.
    finalize_axes(fig, force_remove_titles=True)
    out_dir = (OUT / subdir) if subdir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}__{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.25)   # vector companion
    plt.close(fig)
    rel = f"composites/{subdir}/{path.name}" if subdir else f"composites/{path.name}"
    print(f"  {tag:12} -> {rel} (+ .pdf)")


def _group_bar_and_sankey_fns(model_type_placeholder, D, Dpost, group_mode, include_nonresp, method="temporal"):
    """Shared group_bar_fn/sankey_fn selection (fine vs. broad responder
    categories), used by figure1/figure1_exact/figure1_top/figure1_bottom so
    the group-mode switch stays in one place. Returns
    (group_bar_fn(ax, mt, D_), sankey_fn(ax, mt)) -- both take the model
    type explicitly since these composites vary it per-column/row."""
    if group_mode == "fine":
        group_bar_fn = (lambda ax, mt, D_: T.draw_responder_group_bar(
            mt, D=D_, include_nonresp=include_nonresp, ax=ax))
        sankey_fn = (lambda ax, mt: RF.draw_fine_sankey(mt, Dpre=D, Dpost=Dpost, ax=ax, method=method))
    else:
        group_bar_fn = (lambda ax, mt, D_: T.draw_responder_group_bar_broad(
            mt, D=D_, include_nonresp=include_nonresp, ax=ax))
        sankey_fn = (lambda ax, mt: RB.draw_broad_sankey(mt, Dpre=D, Dpost=Dpost, ax=ax, method=method))
    return group_bar_fn, sankey_fn


def figure1(letters=False, group_mode="broad", include_nonresp=True, legend=True, method="temporal"):
    """Model overview: one column per model type, one row per analysis --
    the model-side analogue of neuronal-representations' FIG1
    (expert-vs-reversal overview). Here MODEL TYPE is the primary comparison
    axis (phase is folded into each panel: the vigour/value-vs-trials and
    Sankey panels already span pre->post within themselves), rather than
    FIG1's phase-as-primary-axis layout, since the modelling side's central
    question is "which model", not "which phase".

    group_mode: "broad" (3 winner-take-all preferred-stimulus categories,
    default) or "fine" (7-way mixed-selectivity groups) -- selects both the
    responder-group-size bar and the Sankey category scheme, matching
    figure1_exact's own group_mode convention.
    include_nonresp: pass False to drop the non-responsive category from
    the responder-group-size bar (row 2) -- both draw_responder_group_bar
    and draw_responder_group_bar_broad already support this directly.

    Rows: population activity / responder-group sizes / pooled tuning
    heatmap (all seeds' pre-reversal expert data) / vigour vs. trials
    (recovered seeds only) / value estimate vs. trials (recovered seeds
    only) / Sankey (pre->post transitions) / pre->post (cross-context)
    stimulus decode accuracy vs. trials since reversal."""
    D = T._load_D(method)
    Dpost = _method.load_D(F, MR, _HERE / "transfer" / f"figure_data_reversal{REV_TAG}", method)
    group_bar_fn, sankey_fn = _group_bar_and_sankey_fns(None, D, Dpost, group_mode, include_nonresp, method=method)
    rows = [
        [(f"TRANSFER.popact.{mt}", (lambda ax, mt=mt: T.draw_population_activity_bar(mt, D=D, ax=ax)))
         for mt in MODEL_TYPES],
        [(f"TRANSFER.respgroups.{mt}",
          (lambda ax, mt=mt: group_bar_fn(ax, mt, D)))
         for mt in MODEL_TYPES],
        [(f"SUB.heatmappool.{mt}",
          (lambda ax, mt=mt: PT.draw_pooled_tuning_heatmap(mt, D, ax=ax, phase_label="pre",
                                                            exclude_nonresp=not include_nonresp)))
         for mt in MODEL_TYPES],
        [(f"TRANSFER.vigour_trials.{mt}",
          (lambda ax, mt=mt: VV.draw_metric_vs_trials(mt, key="vigour", ax=ax)))
         for mt in MODEL_TYPES],
        [(f"TRANSFER.value_trials.{mt}",
          (lambda ax, mt=mt: VV.draw_metric_vs_trials(mt, key="value", ax=ax)))
         for mt in MODEL_TYPES],
        [(f"TRANSFER.sankey.{mt}",
          (lambda ax, mt=mt: sankey_fn(ax, mt)))
         for mt in MODEL_TYPES],
        [(f"TRANSFER.crosscontext_trials.{mt}",
          (lambda ax, mt=mt: VV.draw_crosscontext_decode_vs_trials(mt, ax=ax)))
         for mt in MODEL_TYPES],
    ]
    n_rows = len(rows)
    fig = plt.figure(figsize=(7.6 * 3, 5.8 * n_rows))
    # Sankey panels (row 5, "TRANSFER.sankey") look small relative to everything
    # else at 1x -- give that row extra vertical room (columns are shared across
    # model types here, not panel types, so height is the lever, not width).
    row_heights = [1.0] * n_rows
    row_heights[5] = 1.4
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.55, wspace=0.4, top=0.94,
                           height_ratios=row_heights)
    _build_grid(fig, gs, rows, letters)
    _column_headers(fig, MODEL_TYPES, n_cols=3)
    suffix = (REV_TAG + ("_lettered" if letters else "") + ("_fine" if group_mode == "fine" else "")
              + ("_no_nonresp" if not include_nonresp else "") + ("_nolegend" if not legend else "")
              + f"_{method}")
    tag = ("MODELFIG1" + ("_fine" if group_mode == "fine" else "")
           + ("_EXCL_NONRESP" if not include_nonresp else "") + f"_{method}")
    _save(fig, tag, "model_overview" + suffix, legend=legend,
          subdir=f"{method}/{'fine' if group_mode == 'fine' else 'coarse'}")


def figure1_top(letters=False, group_mode="broad", legend=True, method="temporal"):
    """Top 3 rows of figure1() as their own composite -- population
    activity / responder-group sizes (non-responsive category dropped,
    unlike figure1()'s own group-size row) / pooled tuning heatmap (no
    per-panel title -- the column header above already names the model) --
    the expert/pre-reversal summary: everything in figure1() that describes
    the model's steady-state pre-reversal representation, without the
    reversal-dynamics rows below it."""
    D = T._load_D(method)
    Dpost = _method.load_D(F, MR, _HERE / "transfer" / f"figure_data_reversal{REV_TAG}", method)
    group_bar_fn, _ = _group_bar_and_sankey_fns(None, D, Dpost, group_mode, False, method=method)
    rows = [
        [(f"TRANSFER.popact.{mt}", (lambda ax, mt=mt: T.draw_population_activity_bar(mt, D=D, ax=ax)))
         for mt in MODEL_TYPES],
        [(f"TRANSFER.respgroups.{mt}",
          (lambda ax, mt=mt: group_bar_fn(ax, mt, D)))
         for mt in MODEL_TYPES],
        [(f"SUB.heatmappool.{mt}",
          (lambda ax, mt=mt: PT.draw_pooled_tuning_heatmap(mt, D, ax=ax, phase_label="pre",
                                                            exclude_nonresp=True)))
         for mt in MODEL_TYPES],
    ]
    n_rows = len(rows)
    fig = plt.figure(figsize=(7.6 * 3, 5.8 * n_rows))
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.55, wspace=0.4, top=0.94)
    _build_grid(fig, gs, rows, letters)
    _clear_heatmap_titles(fig)   # heatmaps: drop their own title, column header names the model
    _column_headers(fig, MODEL_TYPES, n_cols=3)
    suffix = (REV_TAG + ("_lettered" if letters else "") + ("_fine" if group_mode == "fine" else "")
              + ("_nolegend" if not legend else "") + f"_{method}")
    tag = "MODELFIG1_EXPERT" + ("_fine" if group_mode == "fine" else "") + f"_{method}"
    _save(fig, tag, "model_expert_summary" + suffix, legend=legend,
          subdir=f"{method}/{'fine' if group_mode == 'fine' else 'coarse'}")


def figure1_bottom(letters=False, group_mode="broad", legend=True, method="temporal"):
    """Bottom 4 rows of figure1() as their own composite -- vigour vs.
    trials / value vs. trials / Sankey (pre->post transitions) / pre->post
    (cross-context) stimulus decode accuracy vs. trials since reversal --
    the post-reversal dynamics: everything in figure1() that tracks the
    model as it adapts through the reversal."""
    D = T._load_D(method)
    Dpost = _method.load_D(F, MR, _HERE / "transfer" / f"figure_data_reversal{REV_TAG}", method)
    _, sankey_fn = _group_bar_and_sankey_fns(None, D, Dpost, group_mode, True, method=method)
    rows = [
        [(f"TRANSFER.vigour_trials.{mt}",
          (lambda ax, mt=mt: VV.draw_metric_vs_trials(mt, key="vigour", ax=ax)))
         for mt in MODEL_TYPES],
        [(f"TRANSFER.value_trials.{mt}",
          (lambda ax, mt=mt: VV.draw_metric_vs_trials(mt, key="value", ax=ax)))
         for mt in MODEL_TYPES],
        [(f"TRANSFER.sankey.{mt}",
          (lambda ax, mt=mt: sankey_fn(ax, mt)))
         for mt in MODEL_TYPES],
        [(f"TRANSFER.crosscontext_trials.{mt}",
          (lambda ax, mt=mt: VV.draw_crosscontext_decode_vs_trials(mt, ax=ax)))
         for mt in MODEL_TYPES],
    ]
    n_rows = len(rows)
    fig = plt.figure(figsize=(7.6 * 3, 5.8 * n_rows))
    # Same Sankey row-height bump as figure1() (row 2 here, "TRANSFER.sankey").
    row_heights = [1.0] * n_rows
    row_heights[2] = 1.4
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.55, wspace=0.4, top=0.94,
                           height_ratios=row_heights)
    _build_grid(fig, gs, rows, letters)
    _column_headers(fig, MODEL_TYPES, n_cols=3)
    suffix = (REV_TAG + ("_lettered" if letters else "") + ("_fine" if group_mode == "fine" else "")
              + ("_nolegend" if not legend else "") + f"_{method}")
    tag = "MODELFIG1_POSTREV" + ("_fine" if group_mode == "fine" else "") + f"_{method}"
    _save(fig, tag, "model_postreversal_dynamics" + suffix, legend=legend,
          subdir=f"{method}/{'fine' if group_mode == 'fine' else 'coarse'}")


def figure1_bottom_gradient(letters=False, group_mode="broad", legend=True, method="temporal"):
    """Same as figure1_bottom(), but the first row (vigour vs. trials, the
    full trial-resolved curve) is replaced with reversal_gradient_bars.
    draw_rnn_gradient_bars: the per-stimulus recovery-point vigour GRADIENT
    as one bar each, per model -- the RNN-side analogue of the experimental
    compose.py's figure1_gradient() panel-f swap (see chat). Every other
    row, and every parameter, is identical to figure1_bottom()."""
    D = T._load_D(method)
    Dpost = _method.load_D(F, MR, _HERE / "transfer" / f"figure_data_reversal{REV_TAG}", method)
    _, sankey_fn = _group_bar_and_sankey_fns(None, D, Dpost, group_mode, True, method=method)
    rows = [
        [(f"TRANSFER.gradient.{mt}",
          (lambda ax, mt=mt: RGB.draw_rnn_gradient_bars(mt, ax=ax)))
         for mt in MODEL_TYPES],
        [(f"TRANSFER.value_trials.{mt}",
          (lambda ax, mt=mt: VV.draw_metric_vs_trials(mt, key="value", ax=ax)))
         for mt in MODEL_TYPES],
        [(f"TRANSFER.sankey.{mt}",
          (lambda ax, mt=mt: sankey_fn(ax, mt)))
         for mt in MODEL_TYPES],
        [(f"TRANSFER.crosscontext_trials.{mt}",
          (lambda ax, mt=mt: VV.draw_crosscontext_decode_vs_trials(mt, ax=ax)))
         for mt in MODEL_TYPES],
    ]
    n_rows = len(rows)
    fig = plt.figure(figsize=(7.6 * 3, 5.8 * n_rows))
    row_heights = [1.0] * n_rows
    row_heights[2] = 1.4
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.55, wspace=0.4, top=0.94,
                           height_ratios=row_heights)
    _build_grid(fig, gs, rows, letters)
    _column_headers(fig, MODEL_TYPES, n_cols=3)
    suffix = (REV_TAG + ("_lettered" if letters else "") + ("_fine" if group_mode == "fine" else "")
              + ("_nolegend" if not legend else "") + f"_{method}")
    tag = "MODELFIG1_POSTREV_GRAD" + ("_fine" if group_mode == "fine" else "") + f"_{method}"
    _save(fig, tag, "model_postreversal_dynamics_gradient" + suffix, legend=legend,
          subdir=f"{method}/{'fine' if group_mode == 'fine' else 'coarse'}")


def figure_gradient_all_sources(letters=False, legend=True, window=100):
    """1x4 grid: real mice (fixed-window lick-rate gradient) alongside all 3
    RNN model types (per-seed recovery-point vigour gradient), side by side
    -- the final panel the user asked for after the two composite splices
    above (figure1_gradient() on the experimental side's own compose.py,
    figure1_bottom_gradient() here): one place to compare the reversal
    learning-rate "gradient" summary across every data source at once.
    window: passed through to draw_experimental_gradient_bars (100 or 200,
    matching that function's own build_all() sweep) -- doesn't affect the
    RNN panels, which use the seed-specific recovery-point window instead
    (see reversal_gradient_bars.py's module docstring for why the two data
    sources use different window definitions)."""
    row = [("GRADIENT.exp", (lambda ax: RGB.draw_experimental_gradient_bars(window=window, ax=ax)))]
    row += [(f"GRADIENT.{mt}", (lambda ax, mt=mt: RGB.draw_rnn_gradient_bars(mt, ax=ax)))
            for mt in MODEL_TYPES]
    fig = plt.figure(figsize=(4.8 * 4, 5.6))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.55, top=0.86)
    _build_grid(fig, gs, [row], letters)
    labels = ["Real mice"] + [F.MODELS[mt]["label"] for mt in MODEL_TYPES]
    for c, lab in enumerate(labels):
        bbox = fig.axes[c].get_position()
        xc = (bbox.x0 + bbox.x1) / 2
        fig.text(xc, 0.985, lab, ha="center", va="top", fontsize=20)
    suffix = (REV_TAG + ("_lettered" if letters else "") + ("_nolegend" if not legend else "")
              + f"_w{window}")
    tag = f"GRADIENT_ALLSOURCES_w{window}{REV_TAG}"
    _save(fig, tag, "gradient_all_sources" + suffix, legend=legend, subdir="gradient")


def figure_gradient_fit_diagnostic():
    """Supplementary: RGB.draw_rnn_fit_diagnostic_grid() as its own saved
    figure, one per model type -- every recovered seed's own recovery-point
    fit (trajectory + highlighted fit window + fit line + recovery-point
    marker), not just the resulting slope bar (see figure1_bottom_gradient's
    first row for that). Own function (not a _panel()/gridspec composite)
    since draw_rnn_fit_diagnostic_grid already builds and returns its own
    complete multi-seed grid figure -- mirrors the experimental side's
    figure_gradient_fit_diagnostic() in neuronal-representations' own
    compose.py."""
    out_dir = FIG_ROOT / "composites" / "supplementary"
    out_dir.mkdir(parents=True, exist_ok=True)
    for mt in MODEL_TYPES:
        fig = RGB.draw_rnn_fit_diagnostic_grid(mt)
        path_ = out_dir / f"GRADIENT_FITDIAG_{mt}{REV_TAG}__rnn_gradient_fit_diagnostic_{mt}{REV_TAG}.png"
        fig.savefig(path_, dpi=150, bbox_inches="tight")
        fig.savefig(path_.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"  GRADIENT_FITDIAG_{mt} -> supplementary/{path_.name} (+ .pdf)")


def figure_dimensionality_seed_grids():
    """Supplementary: population_pca_panels.draw_pca_phase_space_grid() /
    population_tdr_panels.draw_tdr_phase_space_grid() as their own saved
    figures, one per model type -- every seed's own phase-space trajectory
    (record of every seed, per the chat), the RNN-side analogue of
    figure_dimensionality_grids() in neuronal-representations' own
    compose.py (multi-SESSION grids there, multi-SEED grids here)."""
    out_dir = FIG_ROOT / "composites" / "supplementary"
    out_dir.mkdir(parents=True, exist_ok=True)
    for mt in MODEL_TYPES:
        figs = [
            (f"DIM_PCA_GRID_{mt}", f"{mt}_pca_phase_grid_all_seeds{REV_TAG}",
             population_pca_panels.draw_pca_phase_space_grid(mt)),
            (f"DIM_TDR_GRID_{mt}", f"{mt}_tdr_phase_grid_all_seeds{REV_TAG}",
             population_tdr_panels.draw_tdr_phase_space_grid(model_type=mt)),
        ]
        for tag, name, fig in figs:
            path_ = out_dir / f"{tag}{REV_TAG}__{name}.png"
            fig.savefig(path_, dpi=150, bbox_inches="tight")
            fig.savefig(path_.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(fig)
            print(f"  {tag} -> supplementary/{path_.name} (+ .pdf)")


def figure1_exact(model_type, letters=False, group_mode="broad", include_nonresp=True, legend=True, method="temporal"):
    """Literal per-model equivalent of the real repo's FIG1
    (expert_reversal_overview): PHASE (pre/post) as the primary axis (rows),
    analysis type as columns -- matching neuronal-representations' own
    compose.py:figure1() layout exactly (behaviour bar / population-mean
    trace / responder-group sizes / [stimpair-decode-bar | Sankey] / pooled
    tuning heatmap). One composite per model type, since here MODEL is what
    varies rather than the real FIG1's single dataset.

    group_mode: "broad" (3 winner-take-all preferred-stimulus categories,
    default) or "fine" (7-way mixed-selectivity groups) -- selects both the
    responder-size bar and the Sankey's category scheme, saved under
    MODELFIG1EQ_<model> / MODELFIG1EQ_fine_<model> respectively.

    Row 1 (pre, the "expert" analogue): vigour bar, population mean trace,
        responder-group sizes, time-pooled stimulus-pair decode bar (pre
        only), pooled tuning heatmap.
    Row 2 (post, the "reversal" analogue): the same four analyses on
        post-reversal data, but with the pre->post Sankey filling the
        decode-bar's column slot -- exactly mirroring how the real FIG1
        itself swaps that one column's panel type between its expert row
        (stimpair decode bar) and its reversal row (Sankey)."""
    D = T._load_D(method)
    Dpost = _method.load_D(F, MR, _HERE / "transfer" / f"figure_data_reversal{REV_TAG}", method)
    have_dec = DEC.DECODE_JSON.exists()
    dec_results = DEC._load() if have_dec else None

    if group_mode == "fine":
        group_bar_fn = (lambda ax, D_: T.draw_responder_group_bar(
            model_type, D=D_, include_nonresp=include_nonresp, ax=ax))
        sankey_fn = (lambda ax: RF.draw_fine_sankey(model_type, Dpre=D, Dpost=Dpost, ax=ax, method=method))
    else:
        group_bar_fn = (lambda ax, D_: T.draw_responder_group_bar_broad(
            model_type, D=D_, include_nonresp=include_nonresp, ax=ax))
        sankey_fn = (lambda ax: RB.draw_broad_sankey(model_type, Dpre=D, Dpost=Dpost, ax=ax, method=method))

    def _stimpair_pre_only(ax):
        if not have_dec or model_type not in dec_results:
            ax.text(0.5, 0.5, "no decoding json\n(run analysis/run_decoding.py)",
                    ha="center", va="center", transform=ax.transAxes)
            return
        r = dec_results[model_type]
        wmc = np.array(r["within_mean_by_context"])
        pairs = [(0, 1), (0, 2), (1, 2)]
        x = np.arange(len(pairs))
        pre_vals = [wmc[0, i, j] for i, j in pairs]
        ax.bar(x, pre_vals, 0.5, color=F.MODELS[model_type]["color"])
        ax.axhline(0.5, color="0.3", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([DEC.PAIR_LABELS[p] for p in pairs], rotation=20, ha="right")
        ax.set_ylim(0.4, 1.02)
        ax.set_ylabel("decode accuracy")

    top = [
        ("BEH.vigourbar.pre", (lambda ax: PT.draw_metric_bar(
            model_type, "vigour", D, ax=ax, ylabel="vigour", phase_label="pre"))),
        ("POP.mean_tr.pre", (lambda ax: PT.draw_population_mean_trace(
            model_type, D, ax=ax, phase_label="pre"))),
        ("TRANSFER.respgroups.pre", (lambda ax: group_bar_fn(ax, D))),
        ("DEC.stimpair_pre", _stimpair_pre_only),
        ("SUB.heatmappool.pre", (lambda ax: PT.draw_pooled_tuning_heatmap(
            model_type, D, ax=ax, phase_label="pre", exclude_nonresp=not include_nonresp))),
    ]
    bottom = [
        ("BEH.vigourbar.post", (lambda ax: PT.draw_metric_bar(
            model_type, "vigour", Dpost, ax=ax, ylabel="vigour", phase_label="post"))),
        ("POP.mean_tr.post", (lambda ax: PT.draw_population_mean_trace(
            model_type, Dpost, ax=ax, phase_label="post"))),
        ("TRANSFER.respgroups.post", (lambda ax: group_bar_fn(ax, Dpost))),
        ("TRANSFER.sankey", sankey_fn),
        ("SUB.heatmappool.post", (lambda ax: PT.draw_pooled_tuning_heatmap(
            model_type, Dpost, ax=ax, phase_label="post", exclude_nonresp=not include_nonresp))),
    ]
    fig = plt.figure(figsize=(6.2 * 5, 5.6 * 2))
    # Column 3 ("DEC.stimpair_pre" / "TRANSFER.sankey") holds the Sankey in the
    # post row -- widen it so the Sankey isn't squeezed relative to the others.
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.6, wspace=0.42, top=0.94,
                           width_ratios=[1, 1, 1, 1.4, 1])
    _build_grid(fig, gs, [top, bottom], letters)
    suffix = (REV_TAG + ("_lettered" if letters else "")
              + ("_no_nonresp" if not include_nonresp else "") + ("_nolegend" if not legend else "")
              + f"_{method}")
    tag = (f"MODELFIG1EQ_{'fine_' if group_mode == 'fine' else ''}{model_type}{REV_TAG}"
           + ("_EXCL_NONRESP" if not include_nonresp else "") + f"_{method}")
    name = "expert_reversal_overview" + ("_fine" if group_mode == "fine" else "") + suffix
    _save(fig, tag, name, legend=legend,
          subdir=f"{method}/{'fine' if group_mode == 'fine' else 'coarse'}")


def figure2(letters=False, legend=True, method="temporal"):
    """Cross-model vs. experiment summary: chi2 fit (pop. activity, fine
    groups, broad groups, vigour correlation) plus cross-context stimulus
    decoding -- the model-side analogue of FIG2 (reversal decoding) -- PLUS
    a second row: representational similarity (RSA / coding-angle) matrix
    comparison -- one combined pre+post-reversal coding-angle heatmap per
    model type, the same real experimental heatmap, and a bar chart of each
    model's Pearson correlation with the experimental RSM (see
    cross_model_vs_experiment/coding_angle_comparison.py's module
    docstring for the full second-order-similarity methodology). Added on
    request as a second row of this same figure rather than a separate one,
    since it's the same kind of model-vs-experiment comparison as row 1,
    just for representational geometry instead of chi2/decoding metrics.

    method: responder-significance criterion for the two group-count chi2
    panels (fine/broad) -- matched on both sides (model D and the real-data
    group-count json), see chi2_bars.py. pop_activity/vigour_corr/decoding
    panels are method-independent, as is the new RSA row."""
    if not DEC.DECODE_JSON.exists():
        print(f"  (skip MODELFIG2{REV_TAG}: {DEC.DECODE_JSON} not found "
              f"-- run analysis/run_decoding.py in cxval env)")
        return
    D = T._load_D(method)
    row1 = [
        ("CHI2.pop_activity", (lambda ax: C2.draw_pop_activity_chi2(D=D, ax=ax))),
        ("CHI2.groups_fine", (lambda ax: C2.draw_fine_groups_chi2(D=D, ax=ax, method=method))),
        ("CHI2.groups_broad", (lambda ax: C2.draw_broad_groups_chi2(D=D, ax=ax, method=method))),
        ("CHI2.vigour_corr", (lambda ax: C2.draw_vigour_correlation(D=D, ax=ax))),
        ("DEC.crosscontext_bar", DEC.draw_crosscontext_bar),
    ]
    row2 = [
        (f"MECH.ANGLE.combined.{mt}", (lambda ax, mt=mt: CANG.draw_coding_angle_heatmap_combined(mt, ax=ax)))
        for mt in MODEL_TYPES
    ] + [
        ("CACMP.ANGLE.experimental", (lambda ax: CACMP.draw_experimental_heatmap_triangular(ax=ax))),
        ("CACMP.ANGLE.vsexp_bar", (lambda ax: CACMP.draw_comparison_bar(
            ax=ax, show_errorbars=False, ylim=(0.0, 1.0)))),
    ]
    rows = [row1, row2]
    fig = plt.figure(figsize=(6.4 * 5, 7.2 + 7.6))
    gs = gridspec.GridSpec(2, 5, figure=fig, wspace=0.45, hspace=0.55, top=0.92,
                           height_ratios=[1.0, 1.15])
    _build_grid(fig, gs, rows, letters)
    # Row 2's 4 heatmap panels (3 per-model combined coding-angle matrices +
    # the experimental one) are image axes, so finalize_axes/_save's
    # force_remove_titles=True exempts them and their own (fairly verbose)
    # per-panel titles would otherwise survive -- stripped explicitly here
    # instead, on request, same helper figure_mechanistic_overview uses for
    # its own coding-angle row. A short plain fig.text() label (NOT an axes
    # title, so it isn't a candidate for stripping) replaces it so each
    # heatmap is still identifiable without the verbose text.
    _clear_heatmap_titles(fig)
    row2_labels = [F.MODELS[mt]["label"] for mt in MODEL_TYPES] + ["Neural data", ""]
    for c, label in enumerate(row2_labels):
        if not label:
            continue
        bbox = fig.axes[5 + c].get_position()
        xc = (bbox.x0 + bbox.x1) / 2
        fig.text(xc, bbox.y1 + 0.008, label, ha="center", va="bottom", fontsize=16, fontweight="bold")
    # Row 2's four heatmaps (f-i) all share the same "cosine similarity"
    # y-axis meaning -- keep the label only on the leftmost (f) instead of
    # repeating it 4x. Panel j (the correlation bar, index 9) has a
    # different y-axis and is untouched.
    for c in range(1, 4):
        fig.axes[5 + c].set_ylabel("")
    # Shared colorbar for row 2's heatmaps, placed next to the rightmost one
    # (i -- the neural-data matrix). Added LAST, after every other
    # index-based fig.axes lookup above, since fig.colorbar() appends a new
    # axes immediately after the one it's attached to and would otherwise
    # shift every later panel's index (in particular panel j, index 9).
    # Panel j sat in row 2 (height_ratio 1.15, sized for the big near-square
    # heatmaps beside it), so it rendered as a tall, narrow bar chart --
    # visually inconsistent with panels a-e in row 1 (height_ratio 1.0).
    # Redraw it in a fresh axes matching row 1's own cell HEIGHT (taken from
    # panel a), vertically centred within its original row-2 cell; same
    # x-position/width as before (columns are already shared/uniform across
    # both rows). Done before the colorbar step below so the colorbar's own
    # appended axes doesn't need to be accounted for here.
    ax_j_old = fig.axes[9]
    old_bbox = ax_j_old.get_position()
    # Match row 2's own heatmap height (panel f, index 5) instead of row 1's
    # bar-chart height -- matching row 1 left panel j's axes short and
    # vertically stranded mid-row, with its letter floating above empty
    # space (the row-2 cell is taller than the axes drawn inside it).
    # Top-aligned with the heatmaps (not centred) so j's letter sits at the
    # same height as f/g/h/i's.
    heat_bbox = fig.axes[5].get_position()
    ref_height = heat_bbox.height
    y0 = heat_bbox.y1 - ref_height
    ax_j_old.remove()
    ax_j = fig.add_axes([old_bbox.x0, y0, old_bbox.width, ref_height])
    CACMP.draw_comparison_bar(ax=ax_j, show_errorbars=False, ylim=(0.0, 1.0))
    if letters:
        ax_j.text(-0.1, 1.22, _panel_letter(9), transform=ax_j.transAxes, fontsize=30,
                 fontweight="bold", va="top", ha="left")
    elif _tags.SHOW_TAG:
        ax_j.text(0.015, 0.985, "CACMP.ANGLE.vsexp_bar", transform=ax_j.transAxes, fontsize=6,
                 va="top", ha="left", color="0.35", family="monospace", zorder=1000)

    exp_ax = fig.axes[8]
    exp_im = exp_ax.get_images()[0]
    # make_axes_locatable's divider (rather than plt.colorbar(ax=...)) so the
    # colorbar is carved directly out of panel i's own axes box -- with 5
    # gridspec columns and a neighbouring panel (j) immediately to the
    # right, colorbar(ax=exp_ax)'s automatic placement crowded into the
    # inter-column gap and collided with panel j's y-axis label.
    cax = make_axes_locatable(exp_ax).append_axes("right", size="6%", pad=0.12)
    plt.colorbar(exp_im, cax=cax)
    # No floating "cosine similarity" label above the colorbar -- the shared
    # scale is already named by panel f's own y-axis label (the only one of
    # f-i that keeps it, per the dedupe above), so a second label here was
    # redundant clutter.
    suffix = REV_TAG + ("_lettered" if letters else "") + ("_nolegend" if not legend else "") + f"_{method}"
    _save(fig, f"MODELFIG2_{method}", "crossmodel_summary" + suffix, legend=legend, subdir=method)


def figure2_per_model(model_type, letters=False, legend=True):
    """Per-model FIG2 (reversal_decoding) equivalent. Always builds the
    time-pooled-only version (stimpair / genmat / context / value /
    stim-identity decode, one row) as FIG2EQ_<model>. ADDITIONALLY builds
    the full 2-row exact analogue of the real FIG2 (time-pooled top row,
    time-resolved bottom row -- stimpair TR pre, stimpair TR post, context
    TR, and the time-pooled stim-identity bar reused in the last slot,
    exactly matching neuronal-representations' own figure2() layout) as
    FIG2EQ_full_<model>, once analysis/run_time_resolved_decoding.py has
    been run (needs sklearn, same as run_decoding.py) -- gracefully skipped
    with a clear message otherwise."""
    if not DEC.DECODE_JSON.exists():
        print(f"  (skip FIG2-equiv for {model_type}: {DEC.DECODE_JSON} not found "
              f"-- run analysis/run_decoding.py in cxval env)")
        return
    results = DEC._load()
    if "value_decode" not in results.get(model_type, {}):
        print(f"  (skip FIG2-equiv for {model_type}: crosscontext_decode json is missing "
              f"the new context/value/stimidentity decode keys -- re-run the UPDATED "
              f"analysis/run_decoding.py in cxval env to add them)")
        return
    row = [
        (f"DEC.stimpair.{model_type}", (lambda ax: DEC.draw_stimpair_bar(model_type, results=results, ax=ax))),
        (f"DEC.genmat.{model_type}", (lambda ax: DEC.draw_generalisation_matrix(model_type, results=results, ax=ax))),
        (f"DEC.context.{model_type}", (lambda ax: DEC.draw_context_bar(model_type, results=results, ax=ax))),
        (f"DEC.value.{model_type}", (lambda ax: DEC.draw_value_bar_single(model_type, results=results, ax=ax))),
        (f"DEC.stimid.{model_type}", (lambda ax: DEC.draw_stimidentity_bar_single(model_type, results=results, ax=ax))),
    ]
    fig = plt.figure(figsize=(5.6 * 5, 6.4))
    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.5, top=0.86,
                          width_ratios=[1.2, 1.0, 1.2, 0.7, 0.7])
    _build_grid(fig, gs, [row], letters)
    suffix = ("_lettered" if letters else "") + ("_nolegend" if not legend else "")
    _save(fig, f"FIG2EQ_{model_type}{REV_TAG}", "reversal_decoding_timepooled" + suffix, legend=legend)

    # -- full 2-row version, exact analogue of the real FIG2 (time-pooled top,
    # time-resolved bottom) -- only once analysis/run_time_resolved_decoding.py
    # has been run (needs sklearn; same requirement as run_decoding.py). Bottom
    # row mirrors the real figure2()'s own 4 slots exactly: stimpair TR pre,
    # stimpair TR post, context TR, and the TIME-POOLED stim-identity bar
    # reused to fill the last slot (the real repo's bottom-right panel is
    # ALSO time-pooled, not time-resolved -- see decoding_timeresolved.py's
    # docstring for why no time-resolved value/stim-identity decoder exists).
    if not DTR.TR_JSON.exists():
        print(f"  (skip full FIG2-equiv for {model_type}: {DTR.TR_JSON} not found -- "
              f"run analysis/run_time_resolved_decoding.py in cxval env for the "
              f"time-resolved bottom row)")
        return
    tr_results = DTR._load()
    if model_type not in tr_results:
        print(f"  (skip full FIG2-equiv for {model_type}: not present in {DTR.TR_JSON})")
        return
    top_row = row
    bottom_row = [
        (f"DEC.stimpair_tr_pre.{model_type}",
         (lambda ax: DTR.draw_stimpair_tr_pre(model_type, results=tr_results, ax=ax))),
        (f"DEC.stimpair_tr_post.{model_type}",
         (lambda ax: DTR.draw_stimpair_tr_post(model_type, results=tr_results, ax=ax))),
        (f"DEC.context_tr.{model_type}",
         (lambda ax: DTR.draw_context_tr(model_type, results=tr_results, ax=ax))),
        (f"DEC.stimid.{model_type}",
         (lambda ax: DEC.draw_stimidentity_bar_single(model_type, results=results, ax=ax))),
    ]
    fig2 = plt.figure(figsize=(5.6 * 5, 6.4 * 2))
    gs2 = gridspec.GridSpec(2, 5, figure=fig2, wspace=0.5, hspace=0.55, top=0.92,
                            width_ratios=[1.2, 1.0, 1.2, 0.7, 0.7])
    _build_grid(fig2, gs2, [top_row, bottom_row], letters)
    _save(fig2, f"FIG2EQ_full_{model_type}{REV_TAG}", "reversal_decoding_full" + suffix, legend=legend)


def figure2_top_by_model(letters=False, legend=True):
    """The real FIG2 top row (time-pooled decoding), but with MODEL as rows
    instead of one composite per model -- all 3 models stacked so they're
    directly comparable at a glance. Panel b is the pre-vs-post scatter/line
    plot (draw_stim_scatter, the real repo's own decoding_pooled.
    draw_reversal_stim_scatter analogue) in place of the generalisation-
    matrix heatmap, per request."""
    if not DEC.DECODE_JSON.exists():
        print(f"  (skip FIG2 top-by-model: {DEC.DECODE_JSON} not found -- "
              f"run analysis/run_decoding.py in cxval env)")
        return
    results = DEC._load()
    rows = []
    for mt in MODEL_TYPES:
        rows.append([
            (f"DEC.stimpair.{mt}", (lambda ax, mt=mt: DEC.draw_stimpair_bar(mt, results=results, ax=ax))),
            (f"DEC.stimscatter.{mt}", (lambda ax, mt=mt: DEC.draw_stim_scatter(mt, results=results, ax=ax))),
            (f"DEC.context.{mt}", (lambda ax, mt=mt: DEC.draw_context_bar(mt, results=results, ax=ax))),
            (f"DEC.value.{mt}", (lambda ax, mt=mt: DEC.draw_value_bar_single(mt, results=results, ax=ax))),
            (f"DEC.stimid.{mt}", (lambda ax, mt=mt: DEC.draw_stimidentity_bar_single(mt, results=results, ax=ax))),
        ])
    fig = plt.figure(figsize=(5.6 * 5, 5.4 * len(MODEL_TYPES)))
    gs = gridspec.GridSpec(len(MODEL_TYPES), 5, figure=fig, wspace=0.5, hspace=0.5, top=0.95,
                          width_ratios=[1.2, 1.0, 1.2, 0.7, 0.7])
    _build_grid(fig, gs, rows, letters)
    _row_headers(fig, MODEL_TYPES, n_cols=5)
    suffix = REV_TAG + ("_lettered" if letters else "") + ("_nolegend" if not legend else "")
    _save(fig, "FIG2EQ_top_by_model", "reversal_decoding_timepooled_by_model" + suffix, legend=legend)


def figure2_bottom_by_model(letters=False, legend=True):
    """The real FIG2 bottom row (time-resolved decoding), with MODEL as rows
    instead of one composite per model. Height is squashed relative to the
    top-row composite -- per request, since these curves mostly sit flat at
    chance (0.5) or ceiling (1.0) and don't carry much visual information
    beyond that, unlike the time-pooled bars/scatter above."""
    if not DTR.TR_JSON.exists():
        print(f"  (skip FIG2 bottom-by-model: {DTR.TR_JSON} not found -- "
              f"run analysis/run_time_resolved_decoding.py in cxval env)")
        return
    if not DEC.DECODE_JSON.exists():
        print(f"  (skip FIG2 bottom-by-model: {DEC.DECODE_JSON} not found -- "
              f"run analysis/run_decoding.py in cxval env)")
        return
    tr_results = DTR._load()
    results = DEC._load()
    rows = []
    for mt in MODEL_TYPES:
        if mt not in tr_results:
            continue
        rows.append([
            (f"DEC.stimpair_tr_pre.{mt}", (lambda ax, mt=mt: DTR.draw_stimpair_tr_pre(mt, results=tr_results, ax=ax))),
            (f"DEC.stimpair_tr_post.{mt}", (lambda ax, mt=mt: DTR.draw_stimpair_tr_post(mt, results=tr_results, ax=ax))),
            (f"DEC.context_tr.{mt}", (lambda ax, mt=mt: DTR.draw_context_tr(mt, results=tr_results, ax=ax))),
            (f"DEC.stimid.{mt}", (lambda ax, mt=mt: DEC.draw_stimidentity_bar_single(mt, results=results, ax=ax))),
        ])
    if not rows:
        print("  (skip FIG2 bottom-by-model: no models present in time-resolved JSON)")
        return
    fig = plt.figure(figsize=(5.6 * 4, 3.2 * len(rows)))
    gs = gridspec.GridSpec(len(rows), 4, figure=fig, wspace=0.5, hspace=0.5, top=0.94,
                          width_ratios=[1.2, 1.2, 1.2, 0.7])
    _build_grid(fig, gs, rows, letters)
    _row_headers(fig, MODEL_TYPES[:len(rows)], n_cols=4)
    suffix = REV_TAG + ("_lettered" if letters else "") + ("_nolegend" if not legend else "")
    _save(fig, "FIG2EQ_bottom_by_model", "reversal_decoding_timeresolved_by_model" + suffix, legend=legend)



def figure_group_sizes_grid(letters=False):
    """4x2 grid: raw responder-group SIZE distributions (the counts feeding
    chi2_bars.py's chi2 fit statistic, shown directly rather than collapsed
    into one chi2 number per model -- see figure2() for that). Rows: RNN
    models pooled across the 3 architectures (temporal, then time_averaged),
    then real experimental data (temporal, then time_averaged). Columns:
    broad (3-way, winner-take-all preferred stimulus), then fine (7-way,
    mixed-selectivity powerset). Pre-reversal ("expert") data only on both
    sides -- no REV_TAG dependence."""
    row_specs = [
        ("RNN", "temporal", GS.draw_rnn),
        ("RNN", "time_averaged", GS.draw_rnn),
        ("experimental", "temporal", GS.draw_real),
        ("experimental", "time_averaged", GS.draw_real),
    ]
    grid = []
    for source, method, draw_fn in row_specs:
        grid.append([
            (f"GROUPSIZES.{source}.{method}.broad",
             (lambda ax, draw_fn=draw_fn, method=method: draw_fn(method, "broad", ax=ax))),
            (f"GROUPSIZES.{source}.{method}.fine",
             (lambda ax, draw_fn=draw_fn, method=method: draw_fn(method, "fine", ax=ax))),
        ])
    fig = plt.figure(figsize=(5.8 * 2, 5.6 * 4))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=1.15, wspace=0.35, top=0.90)
    _build_grid(fig, gs, grid, letters)
    # Column headers ("broad" / "fine") and row headers (source + method) --
    # _save() force-strips every per-axes title (composite convention), and
    # unlike the MODEL_TYPES-column composites there's no existing helper for
    # arbitrary row/col labels, so these are placed directly from each
    # panel's own axes position (same technique as _column_headers/_row_headers).
    # Row headers go ABOVE each row (spanning both columns), not to the left,
    # since the left column's own "Count" y-axis label already occupies that
    # margin and a left-side label collides with it.
    col_labels = ["Broad (3-way, preferred stimulus)", "Fine (7-way, mixed selectivity)"]
    for c, label in enumerate(col_labels):
        bbox = fig.axes[c].get_position()
        xc = (bbox.x0 + bbox.x1) / 2
        fig.text(xc, 0.965, label, ha="center", va="top", fontsize=16)
    method_label = {"temporal": "temporal-cluster", "time_averaged": "time-averaged"}
    for r, (source, method, _fn) in enumerate(row_specs):
        bbox_l = fig.axes[r * 2].get_position()
        bbox_r = fig.axes[r * 2 + 1].get_position()
        xc = (bbox_l.x0 + bbox_r.x1) / 2
        yc = bbox_l.y1 + 0.02
        label = f"{'RNN models (pooled across architectures)' if source == 'RNN' else 'Experimental data'} \u2014 {method_label[method]}"
        fig.text(xc, yc, label, ha="center", va="bottom", fontsize=15, fontweight="bold")
    suffix = "_lettered" if letters else ""
    _save(fig, "GROUPSIZES", "responder_group_sizes_grid" + suffix, legend=False)


def figure_group_sizes_by_model(letters=False, method="time_averaged", include_nonresp=False):
    """2x4 grid: responder-group SIZE distributions broken down by
    INDIVIDUAL model (unlike figure_group_sizes_grid()'s RNN-pooled-across-
    architectures row/column), for one responder-classification method only
    (time_averaged by default, on request). Rows: broad (3-way, preferred
    stimulus), then fine (7-way, mixed selectivity). Columns: the 3 RNN
    model types, then experimental data. RNN panels reuse transfer.py's own
    draw_responder_group_bar/_broad (same window-mean t-test convention as
    figure1_top's own group-size row, and the same non-responsive-excluded
    default as that row/MODELFIG1_EXPERT's caption); the experimental-data
    column reuses group_sizes.py's draw_real() (same source as
    figure_group_sizes_grid()'s own experimental columns, so the counts
    here match that figure's time_averaged row exactly). Pre-reversal
    ("expert") data only on both sides -- no REV_TAG dependence."""
    D = T._load_D(method)
    rows = [
        [(f"GROUPBYMODEL.broad.{mt}",
          (lambda ax, mt=mt: T.draw_responder_group_bar_broad(
              mt, D=D, include_nonresp=include_nonresp, ax=ax)))
         for mt in MODEL_TYPES] +
        [("GROUPBYMODEL.broad.experimental",
          (lambda ax: GS.draw_real(method, "broad", ax=ax)))],
        [(f"GROUPBYMODEL.fine.{mt}",
          (lambda ax, mt=mt: T.draw_responder_group_bar(
              mt, D=D, include_nonresp=include_nonresp, ax=ax)))
         for mt in MODEL_TYPES] +
        [("GROUPBYMODEL.fine.experimental",
          (lambda ax: GS.draw_real(method, "fine", ax=ax)))],
    ]
    fig = plt.figure(figsize=(6.6 * 4, 6.5 * 2))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=1.15, wspace=0.45, top=0.86)
    _build_grid(fig, gs, rows, letters)
    # Column headers: the 3 model types via the usual F.MODELS lookup, plus a
    # manually-placed 4th header for the experimental-data column, since
    # _column_headers only knows how to look up MODEL_TYPES keys and can't
    # handle a mixed model/real-data column list.
    col_labels = [F.MODELS[mt]["label"] for mt in MODEL_TYPES] + ["Experimental data"]
    for c, label in enumerate(col_labels):
        bbox = fig.axes[c].get_position()
        xc = (bbox.x0 + bbox.x1) / 2
        fig.text(xc, 0.98, label, ha="center", va="top", fontsize=18)
    # Row headers placed ABOVE each row (not to the left) -- same collision-
    # avoidance as figure_group_sizes_grid(): the left column's own "number
    # of units" y-axis label already occupies the left margin.
    row_labels = ["Broad (3-way, preferred stimulus)", "Fine (7-way, mixed selectivity)"]
    for r, label in enumerate(row_labels):
        bbox_l = fig.axes[r * 4].get_position()
        bbox_r = fig.axes[r * 4 + 3].get_position()
        axes_h = bbox_l.y1 - bbox_l.y0
        xc = (bbox_l.x0 + bbox_r.x1) / 2
        # Clear the panel-letter position (placed at 1.22 * axes-height above
        # each axes' own top edge by _panel()) rather than a flat offset --
        # with only 2 rows here (vs. figure_group_sizes_grid's 4), each row's
        # axes occupy a much bigger absolute share of the figure height, so
        # a flat 0.02 headroom (that figure's own value) isn't enough and
        # the row header collides with the letter/row above.
        yc = bbox_l.y1 + 0.22 * axes_h + 0.035
        fig.text(xc, yc, label, ha="center", va="bottom", fontsize=14, fontweight="bold")
    fig.text(0.5, 1.0, f"{_method.METHOD_LABEL[method]} responder classification",
             ha="center", va="bottom", fontsize=11, style="italic")
    suffix = ("_lettered" if letters else "") + ("" if include_nonresp else "_responders") + f"_{method}"
    _save(fig, "GROUPSIZES_MODEL", "group_sizes_by_model" + suffix, legend=False)



def figure_pairwise_chi2_grid(letters=False):
    """4x4 symmetric heatmap(s) of pairwise chi2-test-of-homogeneity
    statistics between the 4 group-size distributions from
    figure_group_sizes_grid() (RNN pooled across architectures / experimental
    data x temporal (time-pooled) / time_averaged) -- one heatmap for broad
    (3-way) groups, one for fine (7-way) groups, side by side. Unlike
    chi2_bars.py's draw_fine_groups_chi2/draw_broad_groups_chi2 (one-
    directional: model vs. real, real treated as the expected distribution),
    every pair here is symmetric and comparable, including RNN-vs-RNN
    (across method) and real-vs-real (across method), not just model-vs-real.
    Each cell shows the chi2 statistic and a significance-star summary of its
    p-value; higher chi2 / smaller p = more different distributions. Diagonal
    is blank (self-comparison). Pre-reversal ("expert") data only, like
    figure_group_sizes_grid() -- no REV_TAG dependence."""
    fig = plt.figure(figsize=(7.2 * 2, 6.8))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.55, top=0.82, bottom=0.32, left=0.16, right=0.94)
    titles = {"broad": "Broad (3-way, preferred stimulus)", "fine": "Fine (7-way, mixed selectivity)"}
    for c, group_mode in enumerate(["broad", "fine"]):
        ax = fig.add_subplot(gs[0, c])
        chi2, pval = GS.pairwise_chi2_matrix(group_mode)
        n = chi2.shape[0]
        vmax = float(np.nanmax(chi2))
        im = ax.imshow(chi2, cmap="magma_r", vmin=0, vmax=vmax)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                p = pval[i, j]
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                colour = "white" if chi2[i, j] > vmax * 0.55 else "black"
                ax.text(j, i, f"{chi2[i, j]:.0f}\n{stars}", ha="center", va="center",
                        fontsize=10, color=colour)
        ax.set_xticks(range(n))
        ax.set_xticklabels(GS.ENTITY_LABELS, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels(GS.ENTITY_LABELS, fontsize=9)
        ax.set_title(titles[group_mode], fontsize=13)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
        cb.set_label(r"$\chi^2$ (pairwise test of homogeneity)", fontsize=9)
        if letters:
            ax.text(-0.18, 1.16, string.ascii_lowercase[c], transform=ax.transAxes, fontsize=26,
                    fontweight="bold", va="top", ha="left")
    suffix = "_lettered" if letters else ""
    _save(fig, "CHI2MATRIX", "pairwise_group_chi2_matrix" + suffix, legend=False)



def figure_chi2_bars_grid(letters=False):
    """2x2 grid of chi2_bars.py's existing model-vs-real chi2 FIT bar charts
    (draw_broad_groups_chi2 / draw_fine_groups_chi2 -- one bar per RNN model
    type, real data treated as the expected distribution for the goodness-
    of-fit test) tiled across every viable combination of responder-
    significance method (temporal/time-pooled, time_averaged) and group-
    taxonomy granularity (broad 3-way, fine 7-way). Rows: method. Columns:
    granularity. This is the classic per-model chi2 bar view (see MODELFIG2
    for the single-method version) -- as opposed to figure_group_sizes_grid()
    (raw counts, no test) or figure_pairwise_chi2_grid() (symmetric NxN
    matrix across data sources rather than one bar per model)."""
    row_specs = [("temporal", "time-pooled"), ("time_averaged", "time-averaged")]
    col_specs = [("broad", C2.draw_broad_groups_chi2), ("fine", C2.draw_fine_groups_chi2)]
    grid = []
    for method, _mlabel in row_specs:
        D = C2._load_D(method)
        row = []
        for group_mode, draw_fn in col_specs:
            row.append((f"CHI2.{group_mode}.{method}",
                        (lambda ax, draw_fn=draw_fn, D=D, method=method: draw_fn(D=D, ax=ax, method=method))))
        grid.append(row)
    fig = plt.figure(figsize=(6.8 * 2, 5.8 * 2))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.85, wspace=0.4, top=0.90, left=0.15)
    _build_grid(fig, gs, grid, letters)
    col_labels = ["Broad (3-way, preferred stimulus)", "Fine (7-way, mixed selectivity)"]
    for c, label in enumerate(col_labels):
        bbox = fig.axes[c].get_position()
        xc = (bbox.x0 + bbox.x1) / 2
        fig.text(xc, 0.965, label, ha="center", va="top", fontsize=16)
    # Row labels go in the LEFT margin (rotated), not above the row -- an
    # above-row fig.text() collides with the panel-letter machinery's own
    # (-0.1, 1.22 axes-fraction) placement once there are only 2 rows (each
    # row's axes occupy a large figure-fraction, so that fixed axes-fraction
    # offset becomes a large absolute one). Same technique as _row_headers().
    method_label = {"temporal": "Temporal-cluster", "time_averaged": "Time-averaged"}
    for r, (method, _mlabel) in enumerate(row_specs):
        bbox = fig.axes[r * 2].get_position()
        yc = (bbox.y0 + bbox.y1) / 2
        fig.text(0.02, yc, f"Responder test:\n{method_label[method]}", ha="left", va="center",
                 fontsize=13, fontweight="bold", rotation=90)
    suffix = "_lettered" if letters else ""
    _save(fig, "CHI2GRID", "model_vs_real_chi2_grid" + suffix, legend=False)



def figure_chi2_diagnostics(method, group_mode, letters=False):
    """Diagnostic companion to chi2_bars.py's draw_broad_groups_chi2 /
    draw_fine_groups_chi2 bar charts: for one (method, group_mode), one row
    per RNN model type, columns are (observed vs. rescaled-expected group-
    size bars) and (per-category standardized residuals) -- lets the actual
    distributions and per-category errors behind a single chi2 number be
    inspected directly, including the rescaling step (real counts scaled up
    to the model's own total N) that the summary chi2 bar charts don't show.
    See panels/chi2_diagnostics.py for the chi2/dof/p/Cohen's w numbers
    annotated on each left-column panel."""
    grid = []
    for mt in MODEL_TYPES:
        grid.append([
            (f"CHI2DIAG.obsexp.{mt}",
             (lambda ax, mt=mt: CD.draw_obs_vs_expected(mt, method, group_mode, ax=ax))),
            (f"CHI2DIAG.resid.{mt}",
             (lambda ax, mt=mt: CD.draw_residuals(mt, method, group_mode, ax=ax))),
        ])
    fig = plt.figure(figsize=(7.0 * 2, 4.8 * 3))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.8, wspace=0.35, top=0.86, left=0.15)
    _build_grid(fig, gs, grid, letters)
    col_labels = ["Observed vs. rescaled-expected", "Per-category standardized residual"]
    for c, label in enumerate(col_labels):
        bbox = fig.axes[c].get_position()
        xc = (bbox.x0 + bbox.x1) / 2
        fig.text(xc, 0.965, label, ha="center", va="top", fontsize=15)
    method_label = {"temporal": "Temporal-cluster", "time_averaged": "Time-averaged"}
    gm_label = {"broad": "broad (3-way)", "fine": "fine (7-way)"}
    fig.text(0.5, 1.0, f"{method_label[method]} responder test  \u00d7  {gm_label[group_mode]} groups",
              ha="center", va="top", fontsize=17, fontweight="bold")
    # Row labels (model type) in the left margin -- see figure_chi2_bars_grid's
    # comment for why this goes in the margin rather than above each row.
    for r, mt in enumerate(MODEL_TYPES):
        bbox = fig.axes[r * 2].get_position()
        yc = (bbox.y0 + bbox.y1) / 2
        fig.text(0.02, yc, F.MODELS[mt]["label"], ha="left", va="center",
                 fontsize=13, fontweight="bold", rotation=90)
    suffix = "_lettered" if letters else ""
    _save(fig, f"CHI2DIAG_{method}_{group_mode}", "chi2_diagnostics" + suffix, legend=True,
          subdir=f"{method}/{'fine' if group_mode == 'fine' else 'coarse'}")




def figure_vigour_reversal_diagnostic(letters=False, legend=True, window_trials=1500):
    """Standalone companion to MECHFIG1's VIGOUR row: per-SEED (not
    seed-averaged) raw probe_vigour trajectories, one column per model type,
    zoomed on the first `window_trials` trials after the reversal, both the
    0%->100% and 100%->0% conditions overlaid per seed -- see
    vigour_value.draw_vigour_reversal_asymmetry_diagnostic's docstring for
    what question this answers (genuine per-seed slow ramp vs. an averaging
    artifact from staggered fast per-seed transitions), asked on request
    after the seed-mean VIGOUR row showed classif_rl's 0->100 condition
    converging to its new post-reversal level slower than 100->0 converges
    to ITS new level, a asymmetry not seen for rl_only or
    classif_rl_readout_only.

    Not part of the main MECHFIG1 composite (a diagnostic for one specific
    question, not a general mechanistic-analysis row) -- saved separately as
    VIGDIAG.
    """
    rows = [
        [(f"VIGDIAG.{mt}", (lambda ax, mt=mt: VV.draw_vigour_reversal_asymmetry_diagnostic(
            mt, ax=ax, window_trials=window_trials))) for mt in MODEL_TYPES],
    ]
    n_rows = len(rows)
    fig = plt.figure(figsize=(7.6 * 3, 5.8 * n_rows))
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.6, wspace=0.4, top=0.9)
    _build_grid(fig, gs, rows, letters)
    _harmonize_row_ylim(fig, range(n_rows), n_cols=3)
    _column_headers(fig, MODEL_TYPES, n_cols=3)
    suffix = REV_TAG + ("_lettered" if letters else "") + ("_nolegend" if not legend else "")
    _save(fig, "VIGDIAG", "vigour_reversal_diagnostic" + suffix, legend=legend)


def figure_mechanistic_overview(letters=False, legend=True):
    """Mechanistic dissection overview: one column per model type, one row
    per mechanistic analysis. Rows, top to bottom:

    1. Combined pre+post-reversal coding-vector cosine-similarity heatmap (6
       conditions -- 3 pre-reversal stimuli then 3 post-reversal -- in ONE
       triangular heatmap per model, redundant upper half masked out since
       the matrix is exactly symmetric by construction; see
       analysis/coding_angle.py's combined_coding_angle_summary). Cropped to
       [0, 1] (all-positive in practice) with a sequential white->red
       colormap, not a diverging one -- see draw_coding_angle_heatmap_
       combined.
    2. Fraction of units responsive to each stimulus vs trials -- a second,
       simpler representational-stability readout (unit COUNT, not
       direction) alongside the coding-angle row.
    3. Mean population hidden activity per stimulus vs trials -- third
       stability readout, sensitive to activity MAGNITUDE.
    4. Weight-change RATE through learning, Frobenius-norm layer aggregate,
       now a real CONTINUOUS trajectory across the WHOLE run (pre-reversal
       AND reversal phase both dense-checkpointed) -- see
       checkpoint_weights.full_weight_change_trajectory.
    5. Individual input/output weight-change RATE, one line per individual
       INPUT unit (the 3 stim features) and per individual OUTPUT unit
       (actor, critic, and -- where present -- the 3 stim outputs), plus
       ONE aggregate line for the whole recurrent (h2h) block -- higher-
       granularity companion to row 4, same full pre+reversal trajectory.
       Each line is the Frobenius norm of one input/output unit's WHOLE
       weight vector (all hidden units), not a per-hidden-unit value, so it
       IS seed-averaged like every other row -- see weight_change.
       draw_individual_head_rates's docstring. rl_only has no stim_head, so
       it shows 6 lines instead of 9 (no stim outputs).
    6. Gradient-norm magnitude vs trials, full pre+reversal trajectory --
       total grad_norm plus each module's own norm (backbone, vigour_head,
       value_head, stim_head where present). NEW: enabled by
       track_gradients in the instrumented rerun (see gradient_loss_panels.
       draw_gradient_norms); rl_only shows 4 lines instead of 5.
    7. Individual loss-term trajectories vs trials, full pre+reversal --
       policy_loss, value_loss, activity_loss (regularizer), and aux_loss
       (SSL objective, SSL models only). NEW: same instrumented rerun (see
       gradient_loss_panels.draw_loss_terms).
    8. Reward-prediction-error vs trials -- the REAL directly-PROBED RPE
       (probe_rpe in history.json, from the rerun's infer_rpe probe), not a
       reconstruction. This REPLACES the earlier cost-corrected/uncorrected
       RPE-PROXY rows (built from probe_value + probe_vigour + the known
       reward/cost structure, panels/rpe_proxy.py) now that the real probed
       quantity exists -- see vigour_value.draw_metric_vs_trials(key="rpe").
    9. Critic value estimate vs trials.
    10. Vigour vs trials.
    11. Effective circuit diagram, pre-reversal (coarse/broad functional
        groups only -- fine/granular is too visually busy at composite
        scale, still available via circuit_diagram.build_all()).
    12. Effective circuit diagram, post-reversal -- both circuit rows show
        each specific OHE input dimension actually shown (both the constant
        "context" input AND the response-window cue are omitted entirely
        for visual clarity, not just relabelled -- see
        panels/circuit_diagram.py's module docstring for why, and for where
        the full picture is still available) and each specific readout head
        (actor/critic/stim) as its own circle node.

    DATA SOURCE: every panel in this composite now reads from
    model_runs_instrumented / reversal_5000_instrumented (repointed via the
    combined/transfer/model_runs and model_runs_reversal_5k symlinks) rather
    than the older model_runs_ckpt / reversal_5000_ckpt -- a full rerun with
    (a) dense checkpoints through the ENTIRE run including the reversal
    phase (previously reversal only saved a single before/after snapshot),
    and (b) per-update gradient/loss logging. Recovered-seed counts (30 /
    23 / 11 for rl_only / classif_rl / classif_rl_readout_only) are
    unchanged from the old dataset, consistent with this being the same
    underlying training run with additional logging rather than a
    materially different one -- but every row was regenerated from this
    dataset for full internal consistency rather than mixing sources.

    Every panel in this composite uses the SAME recovered-only seed set per
    model_type by default (see vigour_value.recovered_seeds and each
    module's _default_seeds), so a single failed-to-recover seed can't
    quietly distort one row's average while being correctly excluded from
    another's.

    Y-axis labels are shown once per row (leftmost column only) rather than
    repeated 3x -- see _dedupe_row_ylabels. The coding-angle heatmap row
    additionally has its per-panel titles dropped (_clear_heatmap_titles)
    since the shared "cosine similarity" y-label on the first column already
    says what the matrix is.

    A separate, richer weight-matrix figure (full recurrent + input/output
    weight matrices, unit-sorted by functional subgroup, pre AND post
    reversal) is available via weight_matrix_panel.build_all() -- it doesn't
    fit this composite's one-Axes-per-cell row format (each cell there is
    itself a 3-panel layout), so it's a standalone figure set, not a row
    here.

    A related but separate comparison -- how well does each model type's own
    combined coding-angle matrix (a representational similarity matrix,
    RSM) match the REAL experimental one? -- is in
    cross_model_vs_experiment/coding_angle_comparison.py
    (second_order_similarity() / draw_comparison_bar()): one bar per model
    type, the Pearson correlation between that model's 15 unique
    condition-pair cosine similarities and the experimental RSM's same 15
    values (aligned by condition identity, not position -- the two sides
    list the stimulus transitions in different orders). Also not part of
    this composite (a single 3-bar summary figure, not a 3-column row).

    Requires REV_TAG=_5k -- the coding-angle row and this composite's naming
    convention are pinned to the 5000-trial reversal horizon dataset.
    """
    rows = [
        [(f"MECH.ANGLE.combined.{mt}", (lambda ax, mt=mt: CANG.draw_coding_angle_heatmap_combined(mt, ax=ax)))
         for mt in MODEL_TYPES],
        [(f"MECH.FRACRESP.{mt}", (lambda ax, mt=mt: VV.draw_frac_responsive_vs_trials(mt, ax=ax)))
         for mt in MODEL_TYPES],
        [(f"MECH.POPACT.{mt}", (lambda ax, mt=mt: VV.draw_pop_activity_vs_trials(mt, ax=ax)))
         for mt in MODEL_TYPES],
        [(f"MECH.WC.{mt}", (lambda ax, mt=mt: WCH.draw_weight_change_full(mt, ax=ax)))
         for mt in MODEL_TYPES],
        [(f"MECH.WIND.{mt}", (lambda ax, mt=mt: WCH.draw_individual_head_rates(mt, ax=ax)))
         for mt in MODEL_TYPES],
        [(f"MECH.WCREAD.{mt}", (lambda ax, mt=mt: WCH.draw_weight_change_readout_split(mt, ax=ax)))
         for mt in MODEL_TYPES],
        [(f"MECH.GRAD.{mt}", (lambda ax, mt=mt: GLP.draw_gradient_norms(mt, ax=ax)))
         for mt in MODEL_TYPES],
        [(f"MECH.LOSS.{mt}", (lambda ax, mt=mt: GLP.draw_loss_terms(mt, ax=ax)))
         for mt in MODEL_TYPES],
        [(f"MECH.RPE.{mt}", (lambda ax, mt=mt: VV.draw_metric_vs_trials(mt, key="rpe", ax=ax)))
         for mt in MODEL_TYPES],
        [(f"MECH.VALUE.{mt}", (lambda ax, mt=mt: VV.draw_metric_vs_trials(mt, key="value", ax=ax)))
         for mt in MODEL_TYPES],
        [(f"MECH.VIGOUR.{mt}", (lambda ax, mt=mt: VV.draw_metric_vs_trials(mt, key="vigour", ax=ax)))
         for mt in MODEL_TYPES],
        [(f"MECH.CIRCUIT.broad.pre.{mt}", (lambda ax, mt=mt: CDG.draw_circuit_diagram(mt, granularity="broad", phase="pre", ax=ax)))
         for mt in MODEL_TYPES],
        [(f"MECH.CIRCUIT.broad.post.{mt}", (lambda ax, mt=mt: CDG.draw_circuit_diagram(mt, granularity="broad", phase="post", ax=ax)))
         for mt in MODEL_TYPES],
    ]
    # Only the broad (coarse) granularity is shown here -- the fine-grained (8-group)
    # circuit diagram is visually busy enough to be confusing at composite scale; it's
    # still available as a standalone panel via circuit_diagram.build_all().
    n_rows = len(rows)
    row_heights = [1.0] * n_rows
    row_heights[-2] = 1.8  # circuit diagrams need much more vertical room (near-square, tall with input/readout nodes)
    row_heights[-1] = 1.8
    # Two more rows appended AFTER the circuit-diagram height overrides above,
    # so row_heights[-2]/[-1] indexing there still lands on the circuit rows
    # and not on these newly-appended weight-matrix rows.
    row_heights = row_heights + [1.6, 1.6]  # weight-matrix rows: pre-reversal, post-reversal
    total_rows = n_rows + 2
    fig = plt.figure(figsize=(7.6 * 3, 5.8 * sum(row_heights)))
    gs = gridspec.GridSpec(total_rows, 3, figure=fig, hspace=0.75, wspace=0.4, top=0.95,
                           height_ratios=row_heights)
    _build_grid(fig, gs, rows, letters)
    # Rows 1..9 (0-indexed) are ordinary trials-vs-metric line plots where the
    # SAME quantity is plotted per column (model type) -- harmonize their
    # y-limits across the row so a scale difference between model types
    # reads as a real scale difference, not a plotting artifact. Row 0
    # (coding-angle heatmap, fixed [0,1] by construction) and rows -2/-1
    # (circuit diagrams, node-layout axes with no comparable y-quantity) are
    # excluded. Note: these helpers are all called with the ORIGINAL n_rows
    # (not total_rows) since they index fig.axes assuming exactly one Axes
    # per cell -- the weight-matrix rows added below break that assumption
    # (3 Axes per cell via a nested subgridspec), so they must be appended
    # strictly AFTER every n_rows-scoped helper below has already run.
    _harmonize_row_ylim(fig, range(1, n_rows - 2), n_cols=3)
    _dedupe_row_ylabels(fig, n_rows, n_cols=3)
    _clear_heatmap_titles(fig)  # coding-angle heatmap row: title dropped, "cosine similarity"
                                # ylabel (set once, first column, via _dedupe_row_ylabels above)
                                # carries the meaning instead
    _column_headers(fig, MODEL_TYPES, n_cols=3)
    _row_side_label(fig, fig.axes[(n_rows - 2) * 3], "pre-reversal")
    _row_side_label(fig, fig.axes[(n_rows - 1) * 3], "post-reversal")
    # NEW rows: pre/post-reversal weight-matrix heatmaps (weight_matrix_panel.
    # draw_weight_matrix_by_subgroup), appended after all n_rows-scoped helpers
    # above so their extra per-cell Axes never desync that indexing.
    _add_weight_matrix_rows(fig, gs, n_rows, n_rows + 1, MODEL_TYPES,
                            letters=letters, start_idx=n_rows * 3)
    # Same image-axes-title caveat as MECHFIG2 -- see _add_weight_matrix_rows'
    # own docstring: its per-cell title survives force_remove_titles=True
    # (it's on an image axes) unless cleared explicitly, and only works once
    # those axes actually exist.
    _clear_heatmap_titles(fig)
    suffix = REV_TAG + ("_lettered" if letters else "") + ("_nolegend" if not legend else "")
    if legend:
        # Per-row legends (each row has its own distinct legend vocabulary --
        # stim colors, layer colors, input/output/recurrent line styles,
        # module colors, loss-term colors -- so one shared/deduplicated
        # figure-wide legend was becoming unreadable). Placed in the right
        # margin next to each row rather than combined. The weight-matrix
        # rows (n_rows, n_rows+1) have no line-plot legend of their own, so
        # they're excluded from this range same as the circuit-diagram rows.
        _per_row_legends(fig, range(1, n_rows - 2), n_cols=3)
    _save(fig, "MECHFIG1", "mechanistic_overview" + suffix, legend=False)


def figure_mechanistic_postreversal_summary(letters=False, legend=True, window_trials=1500):
    """Focused 6-row companion to MECHFIG1 (figure_mechanistic_overview),
    pulling together the rows most relevant to "what changed about the
    circuit across the reversal, and how" into one standalone figure, on
    request. One column per model type, rows top to bottom:

    1. Seed-wise vigour post-reversal -- per-SEED (not seed-averaged) raw
       probe_vigour trajectories zoomed on the first `window_trials` trials
       after the reversal, both conditions overlaid per seed -- same panel
       as the standalone VIGDIAG figure (vigour_value.draw_vigour_
       reversal_asymmetry_diagnostic), reused here as a row.
    2. Gradient-norm magnitude vs trials, full pre+reversal trajectory --
       same panel as MECHFIG1 row 6 (gradient_loss_panels.draw_gradient_
       norms).
    3. Effective circuit diagram, pre-reversal (broad/coarse functional
       groups) -- same panel as MECHFIG1's second-to-last row.
    4. Effective circuit diagram, post-reversal -- same panel as MECHFIG1's
       last row.
    5. Weight matrices, pre-reversal (weight_matrix_panel.draw_weight_
       matrix_by_subgroup) -- same nested-subgrid row as MECHFIG1's
       second-to-last weight-matrix row.
    6. Weight matrices, post-reversal -- same, post-reversal.

    Every panel here is reused verbatim from MECHFIG1/VIGDIAG's own drawing
    functions (no new analysis) -- this figure exists purely to present
    that specific 6-row subset on its own, without the other 9 rows of the
    full mechanistic overview.

    Row spacing, on request: the whitespace between rows 3-4 (circuit-pre ->
    circuit-post) and rows 4-5 (circuit-post -> weight-pre) is tightened,
    while every other inter-row gap (1-2, 2-3, 5-6) is left exactly as it
    was, and the inter-column whitespace is widened. A single GridSpec's
    ``hspace`` can only set ONE uniform gap for the whole grid, so this is
    built as 6 real content rows interleaved with 5 explicit invisible
    "spacer" rows (one per gap) inside ONE GridSpec with hspace=0 -- each
    spacer's own height_ratio IS that gap, in the same inches-per-unit scale
    as the content rows (ROW_H below), so every gap is set independently and
    exactly, with no cross-talk between them. Content row heights (and so
    subplot sizes) are unchanged from before; only the gaps and the overall
    canvas move."""
    rows = [
        [(f"VIGDIAG.{mt}", (lambda ax, mt=mt: VV.draw_vigour_reversal_asymmetry_diagnostic(
            mt, ax=ax, window_trials=window_trials))) for mt in MODEL_TYPES],
        [(f"MECH.GRAD.{mt}", (lambda ax, mt=mt: GLP.draw_gradient_norms(mt, ax=ax)))
         for mt in MODEL_TYPES],
        [(f"MECH.CIRCUIT.broad.pre.{mt}", (lambda ax, mt=mt: CDG.draw_circuit_diagram(mt, granularity="broad", phase="pre", ax=ax)))
         for mt in MODEL_TYPES],
        [(f"MECH.CIRCUIT.broad.post.{mt}", (lambda ax, mt=mt: CDG.draw_circuit_diagram(mt, granularity="broad", phase="post", ax=ax)))
         for mt in MODEL_TYPES],
    ]
    n_rows = len(rows)
    # Heights are literal INCHES here (ROW_H=1 below), matched to each row's
    # actual rendered size in the ORIGINAL figure (measured directly), not
    # the old nominal height_ratio*4.15 units -- those units only produced
    # those actual sizes because the original single hspace=0.5 quietly
    # shrank every row well below its nominal allocation; this rebuild uses
    # explicit inch-exact spacer rows instead (see docstring), so a row's
    # height_ratio now maps 1:1 to its rendered inches and must equal the
    # size we actually want, not the old (mostly-fictional) nominal unit.
    # circ-pre/circ-post are additionally kept <=~4.87in (the max height the
    # aspect-locked circuit-diagram axes can use at this column width before
    # WIDTH becomes binding) so they stay HEIGHT-bound -- i.e. fill their
    # row exactly with zero extra vertical buffer -- rather than being
    # centred with ~0.8in of invisible buffer above/below regardless of the
    # gap spacer, which is what happens once a row is taller than that.
    content_heights = [2.505, 2.504, 3.882, 3.882, 3.381, 3.381]  # vigour, grad, circ-pre, circ-post, wmat-pre, wmat-post
    GAP_NORMAL = 1.628    # matches the original uniform hspace=0.5 gap, in inches
    GAP_TIGHT = 0.7       # rows 3-4 and 4-5 -- reduced on request
    TOP_MARGIN = 0.9
    BOTTOM_MARGIN = 0.9
    gaps = [GAP_NORMAL, GAP_NORMAL, GAP_TIGHT, GAP_TIGHT, GAP_NORMAL]  # gaps 1-2,2-3,3-4,4-5,5-6

    # interleave: [TOP_MARGIN, row0, gap0, row1, gap1, row2, gap2, row3, gap3, row4, gap4, row5, BOTTOM_MARGIN]
    height_ratios = [TOP_MARGIN]
    content_row_idx = []   # gridspec row index of each of the 6 content rows
    for i, h in enumerate(content_heights):
        height_ratios.append(h)
        content_row_idx.append(len(height_ratios) - 1)
        if i < len(gaps):
            height_ratios.append(gaps[i])
    height_ratios.append(BOTTOM_MARGIN)
    total_rows = len(height_ratios)

    ROW_H = 1.0   # height_ratios above are already literal inches
    fig = plt.figure(figsize=(7.2 * 3, ROW_H * sum(height_ratios)))
    gs = gridspec.GridSpec(total_rows, 3, figure=fig, hspace=0, wspace=0.5, top=1.0, bottom=0.0,
                           height_ratios=height_ratios)

    r0, r1, r2, r3, r4, r5 = content_row_idx
    for logical_r, row in enumerate(rows):
        gr = content_row_idx[logical_r]
        for c, (tag, fn) in enumerate(row):
            idx = logical_r * 3 + c
            _panel(fig, gs, gr, c, tag, fn,
                  letter=_panel_letter(idx) if letters else None)
    _harmonize_row_ylim(fig, range(0, 2), n_cols=3)
    _dedupe_row_ylabels(fig, n_rows, n_cols=3)
    _column_headers(fig, MODEL_TYPES, n_cols=3)
    # Big vertical "pre-reversal"/"post-reversal" row labels for the two
    # circuit-diagram rows -- matching the weight-matrix rows' own labels
    # below (_add_weight_matrix_rows) and _column_headers' font size, on
    # request, rather than leaving these two rows unlabeled in the margin.
    _row_side_label(fig, fig.axes[2 * 3], "pre-reversal")
    _row_side_label(fig, fig.axes[3 * 3], "post-reversal")
    _add_weight_matrix_rows(fig, gs, r4, r5, MODEL_TYPES,
                            letters=letters, start_idx=n_rows * 3)
    # Weight-matrix cells' own per-panel title lives on an IMAGE axes
    # (ax_in), so _save()'s force_remove_titles=True alone doesn't drop it
    # -- must be cleared explicitly, same as every other heatmap-bearing
    # composite row, and only AFTER those axes exist.
    _clear_heatmap_titles(fig)
    suffix = REV_TAG + ("_lettered" if letters else "") + ("_nolegend" if not legend else "")
    if legend:
        _per_row_legends(fig, range(0, 2), n_cols=3)
    _save(fig, "MECHFIG2", "mechanistic_postreversal_summary" + suffix, legend=False)



def figure_noise_robustness(letters=False, legend=True):
    """New analysis, on request: does noise added to TEST-set hidden
    activations degrade stimulus decoding accuracy, and does it do so in a
    stimulus-specific way (some stimuli mistaken for particular others more
    than for others)? Probes whether a model's learned stimulus
    representation geometry makes it more or less robust to activation
    noise -- see analysis/run_noise_robustness.py's module docstring for
    the full methodology (train a linear decoder on CLEAN pre-reversal
    "expert" activations, corrupt only the held-out test split with
    per-unit-std-scaled iid Gaussian noise at a sweep of magnitudes,
    per-seed-then-averaged aggregation, same conventions as every other
    per-seed analysis in this codebase).

    6 panels: top row -- decoding accuracy vs. noise magnitude, one line
    per stimulus condition (plus overall, black) per model type. Bottom
    row -- one confusion matrix per model type, at ONE shared
    representative noise magnitude (dashed vertical line in the top row;
    chosen post-hoc as a level with visible but non-saturating degradation
    for every model type -- showing confusion matrices at every sweep
    level would be impractical, per request)."""
    if not NR.NOISE_JSON.exists():
        print(f"  (skip NOISE: {NR.NOISE_JSON} not found -- "
              f"run analysis/run_noise_robustness.py in cxval env)")
        return
    rows = [
        [(f"NOISE.acc.{mt}", (lambda ax, mt=mt: NR.draw_noise_accuracy(mt, ax=ax)))
         for mt in MODEL_TYPES],
        [(f"NOISE.conf.{mt}", (lambda ax, mt=mt: NR.draw_noise_confusion(mt, ax=ax)))
         for mt in MODEL_TYPES],
    ]
    fig = plt.figure(figsize=(6.2 * 3, 5.2 * 2))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4, top=0.92, right=0.90)
    _build_grid(fig, gs, rows, letters)
    _column_headers(fig, MODEL_TYPES, n_cols=3)
    _clear_heatmap_titles(fig)
    conf_axes = fig.axes[3:6]
    conf_images = [im for ax in conf_axes for im in ax.get_images()]
    if conf_images:
        fig.colorbar(conf_images[-1], ax=conf_axes, fraction=0.025, pad=0.02,
                     label="fraction of trials")
    suffix = ("_lettered" if letters else "") + ("_nolegend" if not legend else "")
    if legend:
        _per_row_legends(fig, [0], n_cols=3)
    _save(fig, "NOISEFIG", "noise_robustness" + suffix, legend=False)


def _build_3d_phasespace_row(fig, gs, row, draw_fn, model_types, letters=False, start_idx=0):
    """3D analogue of _build_grid for a single row: draw_pca_phase_space_3d /
    draw_tdr_phase_space_3d need a genuine 3D axes (projection="3d"), which
    _panel()/_build_grid's plain fig.add_subplot(gs[r, c]) doesn't create --
    so this bypasses _build_grid the same way _add_weight_matrix_rows does
    for its own nested-subgrid cells."""
    for c, mt in enumerate(model_types):
        ax = fig.add_subplot(gs[row, c], projection="3d")
        draw_fn(mt, ax=ax)
        if letters:
            ax.text2D(-0.05, 1.1, _panel_letter(start_idx + c), transform=ax.transAxes,
                      fontsize=30, fontweight="bold", va="top", ha="left")
        elif _tags.SHOW_TAG:
            ax.text2D(0.015, 0.985, f"3D.{mt}", transform=ax.transAxes, fontsize=6,
                      va="top", ha="left", color="0.35", family="monospace", zorder=1000)


def figure_pca_phasespace_3d(letters=False, legend=True):
    """3D (PC1-PC2-PC3) population-activity phase-space trajectories, one
    panel per model type -- population_pca_panels.draw_pca_phase_space_3d,
    added to the special subset on request (added earlier this session,
    now promoted into its own composite so it can be lettered/synced the
    same way as everything else in figs/special_subset/)."""
    fig = plt.figure(figsize=(7.2 * 3, 6.4))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.5, top=0.88, bottom=0.08)
    _build_3d_phasespace_row(fig, gs, 0, population_pca_panels.draw_pca_phase_space_3d,
                             MODEL_TYPES, letters=letters, start_idx=0)
    _column_headers(fig, MODEL_TYPES, n_cols=3)
    # Each panel's own ax.legend() uses synthetic (unattached) proxy Line2D
    # handles for the stim-colour / pre-post-reversal / marker-meaning key,
    # so _combine_legends' artist-introspection (ax.get_legend_handles_
    # labels()) can't recover them -- build the shared legend directly from
    # the same handles instead (mirrors population_pca_panels.
    # draw_pca_phase_space_3d's own per-panel legend, on request so the
    # composite's legend explains marker shapes and stim colours too, not
    # just the pre/post-reversal linestyle).
    _strip_all_legends(fig)
    if legend:
        stim_handles = [plt.Line2D([0], [0], color=population_pca_panels.S.STIM_COLOURS[s], lw=2.2,
                                   label=f"{population_pca_panels.STIM_PRETTY[s]} stim")
                        for s in population_pca_panels.STIM_ORDER]
        style_handles = [plt.Line2D([0], [0], color="0.2", ls="-", lw=1.8, label="pre-reversal"),
                         plt.Line2D([0], [0], color="0.2", ls="--", lw=1.8, label="post-reversal")]
        handles = stim_handles + style_handles + population_pca_panels._traj_marker_handles()
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.99, 0.5),
                  frameon=False, fontsize=12, labelspacing=0.5)
    suffix = REV_TAG + ("_lettered" if letters else "") + ("_nolegend" if not legend else "")
    _save(fig, "PCAFIG3D", "pca_phase_space_3d" + suffix, legend=False)


def figure_tdr_phasespace_3d(letters=False, legend=True):
    """3D (stim/context/value TDR axes) population-activity phase-space
    trajectories, one panel per model type -- population_tdr_panels.
    draw_tdr_phase_space_3d, same treatment as figure_pca_phasespace_3d
    above."""
    fig = plt.figure(figsize=(7.2 * 3, 6.4))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.5, top=0.88, bottom=0.08)
    _build_3d_phasespace_row(fig, gs, 0, population_tdr_panels.draw_tdr_phase_space_3d,
                             MODEL_TYPES, letters=letters, start_idx=0)
    _column_headers(fig, MODEL_TYPES, n_cols=3)
    # Same treatment as figure_pca_phasespace_3d above -- build the shared
    # legend from the same stim-colour/style/marker handles the per-panel
    # draw_tdr_phase_space_3d legend uses, since _strip_all_legends can't
    # recover its synthetic proxy handles by introspection.
    _strip_all_legends(fig)
    if legend:
        stim_handles = [plt.Line2D([0], [0], color=population_tdr_panels.S.STIM_COLOURS[s], lw=2.2,
                                   label=f"{population_tdr_panels.STIM_PRETTY[s]} stim")
                        for s in population_tdr_panels.STIM_ORDER]
        style_handles = [plt.Line2D([0], [0], color="0.2", ls="-", lw=1.8, label="pre-reversal"),
                         plt.Line2D([0], [0], color="0.2", ls="--", lw=1.8, label="post-reversal")]
        handles = stim_handles + style_handles + population_tdr_panels._traj_marker_handles()
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.99, 0.5),
                  frameon=False, fontsize=12, labelspacing=0.5)
    suffix = REV_TAG + ("_lettered" if letters else "") + ("_nolegend" if not legend else "")
    _save(fig, "TDRFIG3D", "tdr_phase_space_3d" + suffix, legend=False)


def figure_pca_individual_seeds(model_type, n_seeds=10, rng_seed=0):
    """Individual-seed record for one model type: a random (reproducible)
    subset of n_seeds seeds' own per-seed PC1-PC2 phase-space trajectories,
    one small panel per seed (title = seed number) -- on request, as a
    quick-look complement to figure_pca_phasespace_3d()'s cross-seed MEAN
    trajectory (see that function's own docstring / population_pca.py's
    module docstring for exactly how the mean is built: independent per-
    seed PCA fit, sign-aligned, then averaged across seeds -- this figure
    instead shows the un-averaged individual fits themselves, so seed-to-
    seed variability in the mean trajectory can be sanity-checked directly).
    Thin wrapper around population_pca_panels.draw_pca_phase_space_grid's
    existing every-seed grid, just capped to a random subset."""
    fig = population_pca_panels.draw_pca_phase_space_grid(model_type, n_seeds=n_seeds, rng_seed=rng_seed)
    out_dir = OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"pca_individual_seeds_{model_type}{REV_TAG}"
    path_ = out_dir / f"RNN_PCA_SEEDS_{model_type}__{name}.png"
    fig.savefig(path_, dpi=150, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(path_.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  RNN_PCA_SEEDS_{model_type} -> composites/{path_.name} (+ .pdf)")


def figure_tdr_individual_seeds(model_type, n_seeds=10, rng_seed=0):
    """TDR analogue of figure_pca_individual_seeds() above -- same random-
    subset-of-seeds quick-look, in the stim/context/value TDR axes instead
    of PCA components."""
    fig = population_tdr_panels.draw_tdr_phase_space_grid(model_type=model_type, n_seeds=n_seeds, rng_seed=rng_seed)
    out_dir = OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"tdr_individual_seeds_{model_type}{REV_TAG}"
    path_ = out_dir / f"RNN_TDR_SEEDS_{model_type}__{name}.png"
    fig.savefig(path_, dpi=150, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(path_.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  RNN_TDR_SEEDS_{model_type} -> composites/{path_.name} (+ .pdf)")


def figure_pca_individual_seeds_stacked(n_seeds=10, rng_seed=0):
    """One combined figure with all 3 model types' individual-seed PCA
    grids stacked vertically (on request, in place of 3 separate per-model
    files) -- population_pca_panels.draw_pca_phase_space_grid_stacked."""
    fig = population_pca_panels.draw_pca_phase_space_grid_stacked(n_seeds=n_seeds, rng_seed=rng_seed, ncols=5)
    out_dir = OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"pca_individual_seeds_all_models{REV_TAG}"
    path_ = out_dir / f"RNN_PCA_SEEDS_ALL__{name}.png"
    fig.savefig(path_, dpi=150, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(path_.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  RNN_PCA_SEEDS_ALL -> composites/{path_.name} (+ .pdf)")


def figure_tdr_individual_seeds_stacked(n_seeds=10, rng_seed=0):
    """TDR analogue of figure_pca_individual_seeds_stacked() above."""
    fig = population_tdr_panels.draw_tdr_phase_space_grid_stacked(n_seeds=n_seeds, rng_seed=rng_seed, ncols=5)
    out_dir = OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"tdr_individual_seeds_all_models{REV_TAG}"
    path_ = out_dir / f"RNN_TDR_SEEDS_ALL__{name}.png"
    fig.savefig(path_, dpi=150, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(path_.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  RNN_TDR_SEEDS_ALL -> composites/{path_.name} (+ .pdf)")


def figure_pca_individual_seeds_stacked_3d(n_seeds=10, rng_seed=0):
    """3D analogue of figure_pca_individual_seeds_stacked() -- same stacked
    row-band layout, same random seed subset per model, but each panel is a
    PC1-PC2-PC3 3D axes -- population_pca_panels.draw_pca_phase_space_grid_
    stacked_3d."""
    fig = population_pca_panels.draw_pca_phase_space_grid_stacked_3d(n_seeds=n_seeds, rng_seed=rng_seed, ncols=5)
    out_dir = OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"pca_individual_seeds_all_models_3d{REV_TAG}"
    path_ = out_dir / f"RNN_PCA_SEEDS_ALL_3D__{name}.png"
    fig.savefig(path_, dpi=150, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(path_.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  RNN_PCA_SEEDS_ALL_3D -> composites/{path_.name} (+ .pdf)")


def figure_tdr_individual_seeds_stacked_3d(n_seeds=10, rng_seed=0):
    """TDR analogue of figure_pca_individual_seeds_stacked_3d() above."""
    fig = population_tdr_panels.draw_tdr_phase_space_grid_stacked_3d(n_seeds=n_seeds, rng_seed=rng_seed, ncols=5)
    out_dir = OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"tdr_individual_seeds_all_models_3d{REV_TAG}"
    path_ = out_dir / f"RNN_TDR_SEEDS_ALL_3D__{name}.png"
    fig.savefig(path_, dpi=150, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(path_.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  RNN_TDR_SEEDS_ALL_3D -> composites/{path_.name} (+ .pdf)")


def main():
    for legend in (True, False):
        for method in ("time_averaged", "temporal"):
            try:
                figure1(letters=False, legend=legend, method=method)
                figure1(letters=True, legend=legend, method=method)
                figure1(letters=False, include_nonresp=False, legend=legend, method=method)
                figure1(letters=True, include_nonresp=False, legend=legend, method=method)
                figure1_top(letters=False, group_mode="broad", legend=legend, method=method)
                figure1_top(letters=True, group_mode="broad", legend=legend, method=method)
                figure1_top(letters=False, group_mode="fine", legend=legend, method=method)
                figure1_top(letters=True, group_mode="fine", legend=legend, method=method)
                figure1_bottom(letters=False, group_mode="broad", legend=legend, method=method)
                figure1_bottom(letters=True, group_mode="broad", legend=legend, method=method)
                figure1_bottom(letters=False, group_mode="fine", legend=legend, method=method)
                figure1_bottom(letters=True, group_mode="fine", legend=legend, method=method)
                figure1_bottom_gradient(letters=False, group_mode="broad", legend=legend, method=method)
                figure1_bottom_gradient(letters=True, group_mode="broad", legend=legend, method=method)
                figure1_bottom_gradient(letters=False, group_mode="fine", legend=legend, method=method)
                figure1_bottom_gradient(letters=True, group_mode="fine", legend=legend, method=method)
                figure2(letters=False, legend=legend, method=method)
                figure2(letters=True, legend=legend, method=method)
                for mt in MODEL_TYPES:
                    figure1_exact(mt, letters=False, group_mode="broad", legend=legend, method=method)
                    figure1_exact(mt, letters=True, group_mode="broad", legend=legend, method=method)
                    figure1_exact(mt, letters=False, group_mode="fine", legend=legend, method=method)
                    figure1_exact(mt, letters=True, group_mode="fine", legend=legend, method=method)
                    figure1_exact(mt, letters=False, group_mode="broad", include_nonresp=False, legend=legend, method=method)
                    figure1_exact(mt, letters=False, group_mode="fine", include_nonresp=False, legend=legend, method=method)
            except FileNotFoundError as e:
                # e.g. REV_TAG=_5k + method="temporal": raw per-trial time_resolved/*.npz
                # was only ever generated for the 2500-trial horizon (run
                # scripts/16_06_26_run_inference.py against the 5k model_runs to fill
                # this in) -- don't let one missing (REV_TAG, method) combo take out
                # every other combo's composites.
                print(f"  (skip legend={legend} method={method}: {e})")
        # method-independent (pure decoding, no responder-group dependency) -- built once per legend
        for mt in MODEL_TYPES:
            figure2_per_model(mt, letters=False, legend=legend)
            figure2_per_model(mt, letters=True, legend=legend)
        figure2_top_by_model(letters=False, legend=legend)
        figure2_top_by_model(letters=True, legend=legend)
        figure2_bottom_by_model(letters=False, legend=legend)
        figure2_bottom_by_model(letters=True, legend=legend)
        for window in (100, 200):
            figure_gradient_all_sources(letters=False, legend=legend, window=window)
            figure_gradient_all_sources(letters=True, legend=legend, window=window)
    figure_group_sizes_grid(letters=False)
    figure_group_sizes_grid(letters=True)
    figure_group_sizes_by_model(letters=False)
    figure_group_sizes_by_model(letters=True)
    figure_pairwise_chi2_grid(letters=False)
    figure_pairwise_chi2_grid(letters=True)
    figure_chi2_bars_grid(letters=False)
    figure_chi2_bars_grid(letters=True)
    figure_chi2_diagnostics("temporal", "broad", letters=False)
    figure_chi2_diagnostics("temporal", "broad", letters=True)
    figure_chi2_diagnostics("temporal", "fine", letters=False)
    figure_chi2_diagnostics("temporal", "fine", letters=True)
    figure_chi2_diagnostics("time_averaged", "broad", letters=False)
    figure_chi2_diagnostics("time_averaged", "broad", letters=True)
    figure_chi2_diagnostics("time_averaged", "fine", letters=False)
    figure_chi2_diagnostics("time_averaged", "fine", letters=True)
    figure_gradient_fit_diagnostic()
    figure_dimensionality_seed_grids()
    for legend in (True, False):
        figure_noise_robustness(letters=False, legend=legend)
        figure_noise_robustness(letters=True, legend=legend)
    if REV_TAG == "_5k":
        for legend in (True, False):
            figure_mechanistic_overview(letters=False, legend=legend)
            figure_mechanistic_overview(letters=True, legend=legend)
            figure_mechanistic_postreversal_summary(letters=False, legend=legend)
            figure_mechanistic_postreversal_summary(letters=True, legend=legend)
    else:
        print("  (skip mechanistic overview: needs REV_TAG=_5k -- terminal RPE data "
              "only exists for the 5000-trial reversal horizon)")
    for legend in (True, False):
        figure_pca_phasespace_3d(letters=False, legend=legend)
        figure_pca_phasespace_3d(letters=True, legend=legend)
        figure_tdr_phasespace_3d(letters=False, legend=legend)
        figure_tdr_phasespace_3d(letters=True, legend=legend)
    figure_pca_individual_seeds_stacked(n_seeds=10)
    figure_tdr_individual_seeds_stacked(n_seeds=10)
    figure_pca_individual_seeds_stacked_3d(n_seeds=10)
    figure_tdr_individual_seeds_stacked_3d(n_seeds=10)
    print("Composites ->", OUT)


if __name__ == "__main__":
    main()
