"""Group 5 (decoding): the model-side FIG2 (reversal_decoding) equivalent,
time-pooled row -- pre- vs post-reversal stimulus-identity decoding, plus
context/value/stim-identity decoders, all from analysis/run_decoding.py's
output JSON (run in the user's cxval env -- see chat). No live decoding
here, just visualizing the saved JSON (no scipy/sklearn needed to import
this module).

  draw_generalisation_matrix / draw_crosscontext_bar   within vs cross-
      context stimulus-pair decode (existing).
  draw_stimpair_bar        the real repo's draw_reversal_stimpair_bar
      analogue: within-context accuracy for each of the 3 stimulus pairs,
      pre (solid) vs post (faded), from within_mean_by_context.
  draw_context_bar          draw_reversal_context_bar analogue: per
      (identity-anchored) stimulus, can pre vs post trials be told apart?
  draw_value_bar             draw_reversal_value_bar ('value_xor') analogue:
      single bar, 0%-cue vs 100%-cue decoded by CURRENT reward value.
  draw_stimidentity_bar      draw_reversal_stimidentity_bar analogue: single
      bar, 0%-cue vs 100%-cue decoded by IDENTITY, pre+post pooled together.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "style"))
sys.path.insert(0, str(_HERE.parent / "transfer" / "code"))

from _tags import new_panel, save_panel  # noqa: E402
import figures as F  # noqa: E402
import style as S  # noqa: E402

MODEL_TYPES = ["rl_only", "classif_rl", "classif_rl_readout_only"]
# "" = decode against the 2500-trial reversal run's figure_data_reversal,
# "_5k" = the longer run -- must match run_decoding.py's --out for this tag
# (e.g. `REV_TAG=_5k python3 analysis/run_decoding.py --post .../figure_data_reversal_5k
#  --out unified_figures/output/decoding_5k`), and this module reads it back
# from output/decoding{REV_TAG}/ to match.
REV_TAG = os.environ.get("REV_TAG", "")
HORIZON_LABEL = f" [{REV_TAG.lstrip('_')} horizon]" if REV_TAG else ""
DECODE_JSON = _HERE.parent / "output" / f"decoding{REV_TAG}" / "crosscontext_decode_stim_average.json"


def _load():
    return json.load(open(DECODE_JSON))


def draw_generalisation_matrix(model_type, results=None, ax=None):
    fig, ax, _ = new_panel(ax, figsize=(4.6, 4.2))
    results = results if results is not None else _load()
    gm = np.array(results[model_type]["generalisation_matrix"])   # (2,2): rows=train, cols=test
    im = ax.imshow(gm, cmap="magma", vmin=0.4, vmax=1.0)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pre", "post"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["pre", "post"])
    ax.set_xlabel("test context"); ax.set_ylabel("train context")
    for a in range(2):
        for b in range(2):
            ax.text(b, a, f"{gm[a, b]:.2f}", ha="center", va="center",
                     color="w" if gm[a, b] < 0.75 else "k", fontsize=12)
    ax.set_title(f"{F.MODELS[model_type]['label']}\nstimulus-identity decode "
                 f"(n={results[model_type]['n_seeds']} seeds){HORIZON_LABEL}")
    fig.colorbar(im, ax=ax, shrink=0.8, label="decode accuracy")
    return fig


def draw_crosscontext_bar(results=None, ax=None):
    """Mean of the two off-diagonal (cross-context) entries per model --
    chance = 0.5 (binary pairwise decode). Low = representation remaps
    across reversal (doesn't just carry the old stimulus code over), high =
    stimulus code stays fixed regardless of which reward value it now
    carries."""
    fig, ax, _ = new_panel(ax, figsize=(5, 4.5))
    results = results if results is not None else _load()
    vals = {}
    for mt in MODEL_TYPES:
        gm = np.array(results[mt]["generalisation_matrix"])
        vals[mt] = float((gm[0, 1] + gm[1, 0]) / 2.0)
    x = np.arange(len(MODEL_TYPES))
    ax.bar(x, [vals[m] for m in MODEL_TYPES], 0.6,
           color=[F.MODELS[m]["color"] for m in MODEL_TYPES])
    ax.axhline(0.5, color="0.3", linestyle="--", lw=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels([F.MODELS[m]["label"] for m in MODEL_TYPES], rotation=20, ha="right")
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("accuracy")
    ax.set_title("Does the stimulus code survive reversal?\n(mean of pre→post & post→pre)")
    ax.legend(frameon=False)
    return fig


PAIR_LABELS = {(0, 1): "0% vs 50%", (0, 2): "0% vs 100%", (1, 2): "50% vs 100%"}
# Matches the real repo's PAIR_COL (decoding_pooled.py): "0v1"=0% vs 100%,
# "0v50"=0% vs 50%, "1v50"=50% vs 100% -- same hex values, re-keyed by our
# (i, j) stim-index tuples.
PAIR_COLOURS = {(0, 1): "#bebada", (0, 2): "#fb8072", (1, 2): "#8dd3c7"}


def draw_stimpair_bar(model_type, results=None, ax=None):
    """Within-context stimulus-pair decode accuracy, pre (solid) vs post
    (faded) -- the model-side analogue of decoding_pooled.py's
    draw_reversal_stimpair_bar."""
    fig, ax, _ = new_panel(ax, figsize=(5.2, 4.5))
    results = results if results is not None else _load()
    r = results[model_type]
    wmc = np.array(r["within_mean_by_context"])   # (2, n_stim, n_stim)
    pairs = [(0, 1), (0, 2), (1, 2)]
    x = np.arange(len(pairs))
    pre_vals = [wmc[0, i, j] for i, j in pairs]
    post_vals = [wmc[1, i, j] for i, j in pairs]
    ax.bar(x - 0.18, pre_vals, 0.34, color=F.MODELS[model_type]["color"], label="pre")
    ax.bar(x + 0.18, post_vals, 0.34, color=F.MODELS[model_type]["color"], alpha=0.4, label="post")
    ax.axhline(0.5, color="0.3", linestyle="--", lw=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels([PAIR_LABELS[p] for p in pairs], rotation=20, ha="right")
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("accuracy")
    ax.set_title(f"{F.MODELS[model_type]['label']}\nstimulus-pair decoding (n={r['n_seeds']} seeds){HORIZON_LABEL}")
    ax.legend(frameon=False)
    return fig


def draw_stim_scatter(model_type, results=None, ax=None):
    """Pre vs post within-context stimulus-pair decode, connected by a line
    per pair -- the model-side analogue of decoding_pooled.py's
    draw_reversal_stim_scatter (real repo's FIG2 top-row panel b)."""
    fig, ax, _ = new_panel(ax, figsize=(5, 4.5))
    results = results if results is not None else _load()
    r = results[model_type]
    wmc = np.array(r["within_mean_by_context"])   # (2, n_stim, n_stim)
    wsc = np.array(r.get("within_sem_by_context", np.zeros_like(wmc)))
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        y = [wmc[0, i, j], wmc[1, i, j]]
        e = [wsc[0, i, j], wsc[1, i, j]]
        ax.errorbar([0, 1], y, yerr=e, color=PAIR_COLOURS[(i, j)], marker="o", lw=2,
                    capsize=4, label=PAIR_LABELS[(i, j)])
    ax.axhline(0.5, color="0.3", linestyle="--", lw=1, label="chance")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pre", "post"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("accuracy")
    ax.set_title(f"{F.MODELS[model_type]['label']}\nstimulus-pair decoding: pre vs post "
                 f"(n={r['n_seeds']} seeds){HORIZON_LABEL}", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    return fig


def draw_context_bar(model_type, results=None, ax=None):
    """Per-stimulus pre-vs-post (context) decode accuracy -- the model-side
    analogue of draw_reversal_context_bar."""
    fig, ax, _ = new_panel(ax, figsize=(5.0, 4.5))
    results = results if results is not None else _load()
    r = results[model_type]
    cd = r.get("context_decode")
    if cd is None:
        ax.text(0.5, 0.5, "no context_decode\n(rerun run_decoding.py)", ha="center",
                va="center", transform=ax.transAxes)
        return fig
    labels = r.get("stim_labels", ["0%", "50%", "100%"])
    x = np.arange(len(labels))
    ax.bar(x, cd["mean"], 0.6, yerr=cd["sem"], capsize=4,
           color=[S_COLOUR(s) for s in labels])
    ax.axhline(0.5, color="0.3", linestyle="--", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels([f"{s} cue" for s in labels], rotation=20, ha="right")
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("accuracy")
    bl = r.get("context_decode_baseline_control", {}).get("mean")
    bl_txt = f"\n(baseline-window control: {bl:.2f} -- close to these = global drift, not cue-specific)" if bl is not None else ""
    ax.set_title(f"{F.MODELS[model_type]['label']}\nDoes context leak into a fixed cue's activity?{bl_txt}")
    ax.legend(frameon=False)
    return fig


def draw_value_bar(results=None, ax=None):
    """Single bar per model: 0%-cue vs 100%-cue decoded by CURRENT reward
    value ('value_xor' analogue), averaged pre+post."""
    fig, ax, _ = new_panel(ax, figsize=(5, 4.5))
    results = results if results is not None else _load()
    means = [results[m]["value_decode"]["mean"] for m in MODEL_TYPES]
    sems = [results[m]["value_decode"]["sem"] for m in MODEL_TYPES]
    x = np.arange(len(MODEL_TYPES))
    ax.bar(x, means, 0.6, yerr=sems, capsize=4, color=[F.MODELS[m]["color"] for m in MODEL_TYPES])
    ax.axhline(0.5, color="0.3", linestyle="--", lw=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels([F.MODELS[m]["label"] for m in MODEL_TYPES], rotation=20, ha="right")
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("accuracy")
    ax.set_title("Value decode (0% vs 100% cue, by current reward)")
    ax.legend(frameon=False)
    return fig


def draw_stimidentity_bar(results=None, ax=None):
    """Single bar per model: 0%-cue vs 100%-cue decoded by IDENTITY, pre+post
    pooled -- the model-side analogue of draw_reversal_stimidentity_bar."""
    fig, ax, _ = new_panel(ax, figsize=(5, 4.5))
    results = results if results is not None else _load()
    means = [results[m]["stimidentity_decode"]["mean"] for m in MODEL_TYPES]
    sems = [results[m]["stimidentity_decode"]["sem"] for m in MODEL_TYPES]
    x = np.arange(len(MODEL_TYPES))
    ax.bar(x, means, 0.6, yerr=sems, capsize=4, color=[F.MODELS[m]["color"] for m in MODEL_TYPES])
    ax.axhline(0.5, color="0.3", linestyle="--", lw=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels([F.MODELS[m]["label"] for m in MODEL_TYPES], rotation=20, ha="right")
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("accuracy")
    ax.set_title("Stimulus-identity decode (0% vs 100% cue,\npre+post pooled, by identity)")
    ax.legend(frameon=False)
    return fig


def draw_value_bar_single(model_type, results=None, ax=None):
    """Single-model version of draw_value_bar -- one bar, one model, the
    direct per-model analogue of the real repo's single value_xor panel."""
    fig, ax, _ = new_panel(ax, figsize=(3.2, 4.5))
    results = results if results is not None else _load()
    d = results[model_type]["value_decode"]
    ax.bar([0], [d["mean"]], 0.5, yerr=[d["sem"]], capsize=4, color=F.MODELS[model_type]["color"])
    ax.axhline(0.5, color="0.3", linestyle="--", lw=1, label="chance")
    ax.set_xticks([0]); ax.set_xticklabels(["value\n(0% vs 100%)"])
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("accuracy")
    ax.set_title("Value decode")
    ax.legend(frameon=False)
    return fig


def draw_stimidentity_bar_single(model_type, results=None, ax=None):
    """Single-model version of draw_stimidentity_bar."""
    fig, ax, _ = new_panel(ax, figsize=(3.2, 4.5))
    results = results if results is not None else _load()
    d = results[model_type]["stimidentity_decode"]
    ax.bar([0], [d["mean"]], 0.5, yerr=[d["sem"]], capsize=4, color=F.MODELS[model_type]["color"])
    ax.axhline(0.5, color="0.3", linestyle="--", lw=1, label="chance")
    ax.set_xticks([0]); ax.set_xticklabels(["identity\n(0% vs 100% cue)"])
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("accuracy")
    ax.set_title("Stimulus-identity decode\n(pre+post pooled)")
    ax.legend(frameon=False)
    return fig


def S_COLOUR(stim_label):
    key = stim_label.rstrip("%")
    return S.STIM_COLOURS.get(key, "0.4")


def build_all(show_tag=None):
    if not DECODE_JSON.exists():
        print(f"  (skip decoding panels: {DECODE_JSON} not found -- run analysis/run_decoding.py in cxval env)")
        return
    results = _load()
    for mt in MODEL_TYPES:
        try:
            fig = draw_generalisation_matrix(mt, results=results)
        except Exception as e:
            print(f"  (skip generalisation matrix for {mt}: {e})")
            continue
        save_panel(fig, "CrossModel/Decoding", f"DEC.genmat{REV_TAG}.{mt}",
                   f"{mt}_generalisation_matrix{REV_TAG}", show_tag)
    try:
        fig = draw_crosscontext_bar(results=results)
        save_panel(fig, "CrossModel/Decoding", f"DEC.crosscontext_bar{REV_TAG}",
                   f"crosscontext_decode_bar{REV_TAG}", show_tag)
    except Exception as e:
        print(f"  (skip crosscontext bar: {e})")

    for mt in MODEL_TYPES:
        try:
            fig = draw_stimpair_bar(mt, results=results)
        except Exception as e:
            print(f"  (skip stimpair bar for {mt}: {e})")
            continue
        save_panel(fig, "CrossModel/Decoding", f"DEC.stimpair{REV_TAG}.{mt}",
                   f"{mt}_stimpair_bar{REV_TAG}", show_tag)

    for mt in MODEL_TYPES:
        try:
            fig = draw_stim_scatter(mt, results=results)
        except Exception as e:
            print(f"  (skip stim scatter for {mt}: {e})")
            continue
        save_panel(fig, "CrossModel/Decoding", f"DEC.stimscatter{REV_TAG}.{mt}",
                   f"{mt}_stim_scatter{REV_TAG}", show_tag)

    for mt in MODEL_TYPES:
        try:
            fig = draw_context_bar(mt, results=results)
        except Exception as e:
            print(f"  (skip context bar for {mt}: {e})")
            continue
        save_panel(fig, "CrossModel/Decoding", f"DEC.context{REV_TAG}.{mt}",
                   f"{mt}_context_bar{REV_TAG}", show_tag)

    try:
        fig = draw_value_bar(results=results)
        save_panel(fig, "CrossModel/Decoding", f"DEC.value_bar{REV_TAG}",
                   f"value_decode_bar{REV_TAG}", show_tag)
    except Exception as e:
        print(f"  (skip value bar: {e})")

    try:
        fig = draw_stimidentity_bar(results=results)
        save_panel(fig, "CrossModel/Decoding", f"DEC.stimidentity_bar{REV_TAG}",
                   f"stimidentity_decode_bar{REV_TAG}", show_tag)
    except Exception as e:
        print(f"  (skip stimidentity bar: {e})")


if __name__ == "__main__":
    build_all()
