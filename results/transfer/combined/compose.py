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
import decoding as DEC        # noqa: E402
import decoding_timeresolved as DTR  # noqa: E402
import population_timeresolved as PT  # noqa: E402
import vigour_value as VV     # noqa: E402
import figures as F           # noqa: E402

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


def _build_grid(fig, gs, rows, letters):
    idx = 0
    for r, row in enumerate(rows):
        for c, (tag, fn) in enumerate(row):
            _panel(fig, gs, r, c, tag, fn,
                   letter=string.ascii_lowercase[idx] if letters else None)
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


def _save(fig, tag, name):
    _combine_legends(fig)
    # Real repo's own _save() always force-removes titles from every
    # composite (relies on panel tags/letters + axis labels + one shared
    # legend instead) -- matched here on request.
    finalize_axes(fig, force_remove_titles=True)
    path = OUT / f"{tag}__{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  {tag:12} -> composites/{path.name}")


def _group_bar_and_sankey_fns(model_type_placeholder, D, Dpost, group_mode, include_nonresp):
    """Shared group_bar_fn/sankey_fn selection (fine vs. broad responder
    categories), used by figure1/figure1_exact/figure1_top/figure1_bottom so
    the group-mode switch stays in one place. Returns
    (group_bar_fn(ax, mt, D_), sankey_fn(ax, mt)) -- both take the model
    type explicitly since these composites vary it per-column/row."""
    if group_mode == "fine":
        group_bar_fn = (lambda ax, mt, D_: T.draw_responder_group_bar(
            mt, D=D_, include_nonresp=include_nonresp, ax=ax))
        sankey_fn = (lambda ax, mt: RF.draw_fine_sankey(mt, Dpre=D, Dpost=Dpost, ax=ax))
    else:
        group_bar_fn = (lambda ax, mt, D_: T.draw_responder_group_bar_broad(
            mt, D=D_, include_nonresp=include_nonresp, ax=ax))
        sankey_fn = (lambda ax, mt: RB.draw_broad_sankey(mt, Dpre=D, Dpost=Dpost, ax=ax))
    return group_bar_fn, sankey_fn


def figure1(letters=False, group_mode="broad", include_nonresp=True):
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
    D = T._load_D()
    Dpost = F.load(str(_HERE / "transfer" / f"figure_data_reversal{REV_TAG}"))
    group_bar_fn, sankey_fn = _group_bar_and_sankey_fns(None, D, Dpost, group_mode, include_nonresp)
    rows = [
        [(f"TRANSFER.popact.{mt}", (lambda ax, mt=mt: T.draw_population_activity_bar(mt, D=D, ax=ax)))
         for mt in MODEL_TYPES],
        [(f"TRANSFER.respgroups.{mt}",
          (lambda ax, mt=mt: group_bar_fn(ax, mt, D)))
         for mt in MODEL_TYPES],
        [(f"SUB.heatmappool.{mt}",
          (lambda ax, mt=mt: PT.draw_pooled_tuning_heatmap(mt, D, ax=ax, phase_label="pre")))
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
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.55, wspace=0.4, top=0.94)
    _build_grid(fig, gs, rows, letters)
    _column_headers(fig, MODEL_TYPES, n_cols=3)
    suffix = (REV_TAG + ("_lettered" if letters else "") + ("_fine" if group_mode == "fine" else "")
              + ("_no_nonresp" if not include_nonresp else ""))
    tag = ("MODELFIG1" + ("_fine" if group_mode == "fine" else "")
           + ("_EXCL_NONRESP" if not include_nonresp else ""))
    _save(fig, tag, "model_overview" + suffix)


def figure1_top(letters=False, group_mode="broad"):
    """Top 3 rows of figure1() as their own composite -- population
    activity / responder-group sizes (non-responsive category dropped,
    unlike figure1()'s own group-size row) / pooled tuning heatmap (no
    per-panel title -- the column header above already names the model) --
    the expert/pre-reversal summary: everything in figure1() that describes
    the model's steady-state pre-reversal representation, without the
    reversal-dynamics rows below it."""
    D = T._load_D()
    Dpost = F.load(str(_HERE / "transfer" / f"figure_data_reversal{REV_TAG}"))
    group_bar_fn, _ = _group_bar_and_sankey_fns(None, D, Dpost, group_mode, False)
    rows = [
        [(f"TRANSFER.popact.{mt}", (lambda ax, mt=mt: T.draw_population_activity_bar(mt, D=D, ax=ax)))
         for mt in MODEL_TYPES],
        [(f"TRANSFER.respgroups.{mt}",
          (lambda ax, mt=mt: group_bar_fn(ax, mt, D)))
         for mt in MODEL_TYPES],
        [(f"SUB.heatmappool.{mt}",
          (lambda ax, mt=mt: PT.draw_pooled_tuning_heatmap(mt, D, ax=ax, phase_label="pre")))
         for mt in MODEL_TYPES],
    ]
    n_rows = len(rows)
    fig = plt.figure(figsize=(7.6 * 3, 5.8 * n_rows))
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.55, wspace=0.4, top=0.94)
    _build_grid(fig, gs, rows, letters)
    _clear_heatmap_titles(fig)   # heatmaps: drop their own title, column header names the model
    _column_headers(fig, MODEL_TYPES, n_cols=3)
    suffix = REV_TAG + ("_lettered" if letters else "") + ("_fine" if group_mode == "fine" else "")
    tag = "MODELFIG1_EXPERT" + ("_fine" if group_mode == "fine" else "")
    _save(fig, tag, "model_expert_summary" + suffix)


def figure1_bottom(letters=False, group_mode="broad"):
    """Bottom 4 rows of figure1() as their own composite -- vigour vs.
    trials / value vs. trials / Sankey (pre->post transitions) / pre->post
    (cross-context) stimulus decode accuracy vs. trials since reversal --
    the post-reversal dynamics: everything in figure1() that tracks the
    model as it adapts through the reversal."""
    D = T._load_D()
    Dpost = F.load(str(_HERE / "transfer" / f"figure_data_reversal{REV_TAG}"))
    _, sankey_fn = _group_bar_and_sankey_fns(None, D, Dpost, group_mode, True)
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
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.55, wspace=0.4, top=0.94)
    _build_grid(fig, gs, rows, letters)
    _column_headers(fig, MODEL_TYPES, n_cols=3)
    suffix = REV_TAG + ("_lettered" if letters else "") + ("_fine" if group_mode == "fine" else "")
    tag = "MODELFIG1_POSTREV" + ("_fine" if group_mode == "fine" else "")
    _save(fig, tag, "model_postreversal_dynamics" + suffix)


def figure1_exact(model_type, letters=False, group_mode="broad", include_nonresp=True):
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
    D = T._load_D()
    Dpost = F.load(str(_HERE / "transfer" / f"figure_data_reversal{REV_TAG}"))
    have_dec = DEC.DECODE_JSON.exists()
    dec_results = DEC._load() if have_dec else None

    if group_mode == "fine":
        group_bar_fn = (lambda ax, D_: T.draw_responder_group_bar(
            model_type, D=D_, include_nonresp=include_nonresp, ax=ax))
        sankey_fn = (lambda ax: RF.draw_fine_sankey(model_type, Dpre=D, Dpost=Dpost, ax=ax))
    else:
        group_bar_fn = (lambda ax, D_: T.draw_responder_group_bar_broad(
            model_type, D=D_, include_nonresp=include_nonresp, ax=ax))
        sankey_fn = (lambda ax: RB.draw_broad_sankey(model_type, Dpre=D, Dpost=Dpost, ax=ax))

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
            model_type, D, ax=ax, phase_label="pre"))),
    ]
    bottom = [
        ("BEH.vigourbar.post", (lambda ax: PT.draw_metric_bar(
            model_type, "vigour", Dpost, ax=ax, ylabel="vigour", phase_label="post"))),
        ("POP.mean_tr.post", (lambda ax: PT.draw_population_mean_trace(
            model_type, Dpost, ax=ax, phase_label="post"))),
        ("TRANSFER.respgroups.post", (lambda ax: group_bar_fn(ax, Dpost))),
        ("TRANSFER.sankey", sankey_fn),
        ("SUB.heatmappool.post", (lambda ax: PT.draw_pooled_tuning_heatmap(
            model_type, Dpost, ax=ax, phase_label="post"))),
    ]
    fig = plt.figure(figsize=(6.2 * 5, 5.6 * 2))
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.6, wspace=0.42, top=0.94)
    _build_grid(fig, gs, [top, bottom], letters)
    suffix = (REV_TAG + ("_lettered" if letters else "")
              + ("_no_nonresp" if not include_nonresp else ""))
    tag = (f"MODELFIG1EQ_{'fine_' if group_mode == 'fine' else ''}{model_type}{REV_TAG}"
           + ("_EXCL_NONRESP" if not include_nonresp else ""))
    name = "expert_reversal_overview" + ("_fine" if group_mode == "fine" else "") + suffix
    _save(fig, tag, name)


def figure2(letters=False):
    """Cross-model vs. experiment summary: chi2 fit (pop. activity, fine
    groups, broad groups, vigour correlation) plus cross-context stimulus
    decoding -- the model-side analogue of FIG2 (reversal decoding)."""
    if not DEC.DECODE_JSON.exists():
        print(f"  (skip MODELFIG2{REV_TAG}: {DEC.DECODE_JSON} not found "
              f"-- run analysis/run_decoding.py in cxval env)")
        return
    D = T._load_D()
    row = [
        ("CHI2.pop_activity", (lambda ax: C2.draw_pop_activity_chi2(D=D, ax=ax))),
        ("CHI2.groups_fine", (lambda ax: C2.draw_fine_groups_chi2(D=D, ax=ax))),
        ("CHI2.groups_broad", (lambda ax: C2.draw_broad_groups_chi2(D=D, ax=ax))),
        ("CHI2.vigour_corr", (lambda ax: C2.draw_vigour_correlation(D=D, ax=ax))),
        ("DEC.crosscontext_bar", DEC.draw_crosscontext_bar),
    ]
    fig = plt.figure(figsize=(6.4 * 5, 7.2))
    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.45, top=0.88)
    _build_grid(fig, gs, [row], letters)
    suffix = REV_TAG + ("_lettered" if letters else "")
    _save(fig, "MODELFIG2", "crossmodel_summary" + suffix)


def figure2_per_model(model_type, letters=False):
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
    suffix = ("_lettered" if letters else "")
    _save(fig, f"FIG2EQ_{model_type}{REV_TAG}", "reversal_decoding_timepooled" + suffix)

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
    _save(fig2, f"FIG2EQ_full_{model_type}{REV_TAG}", "reversal_decoding_full" + suffix)


def figure2_top_by_model(letters=False):
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
    suffix = REV_TAG + ("_lettered" if letters else "")
    _save(fig, "FIG2EQ_top_by_model", "reversal_decoding_timepooled_by_model" + suffix)


def figure2_bottom_by_model(letters=False):
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
    suffix = REV_TAG + ("_lettered" if letters else "")
    _save(fig, "FIG2EQ_bottom_by_model", "reversal_decoding_timeresolved_by_model" + suffix)


def main():
    figure1(letters=False)
    figure1(letters=True)
    figure1(letters=False, include_nonresp=False)
    figure1(letters=True, include_nonresp=False)
    figure1_top(letters=False, group_mode="broad")
    figure1_top(letters=True, group_mode="broad")
    figure1_top(letters=False, group_mode="fine")
    figure1_top(letters=True, group_mode="fine")
    figure1_bottom(letters=False, group_mode="broad")
    figure1_bottom(letters=True, group_mode="broad")
    figure1_bottom(letters=False, group_mode="fine")
    figure1_bottom(letters=True, group_mode="fine")
    figure2(letters=False)
    figure2(letters=True)
    for mt in MODEL_TYPES:
        figure1_exact(mt, letters=False, group_mode="broad")
        figure1_exact(mt, letters=True, group_mode="broad")
        figure1_exact(mt, letters=False, group_mode="fine")
        figure1_exact(mt, letters=True, group_mode="fine")
        figure1_exact(mt, letters=False, group_mode="broad", include_nonresp=False)
        figure1_exact(mt, letters=False, group_mode="fine", include_nonresp=False)
        figure2_per_model(mt, letters=False)
        figure2_per_model(mt, letters=True)
    figure2_top_by_model(letters=False)
    figure2_top_by_model(letters=True)
    figure2_bottom_by_model(letters=False)
    figure2_bottom_by_model(letters=True)
    print("Composites ->", OUT)


if __name__ == "__main__":
    main()
