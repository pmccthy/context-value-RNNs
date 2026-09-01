"""Time-resolved decoding panels -- the model-side analogue of the
time-resolved half of neuronal-representations' panels/decoding.py (the
bottom row of FIG2), reading the JSON produced by
analysis/run_time_resolved_decoding.py (run separately, in the cxval env --
needs sklearn). No live decoding here, just visualizing the saved JSON.

  draw_stimpair_tr_pre / draw_stimpair_tr_post   3 stimulus-pair curves over
      time, one phase each -- the analogue of
      decoding.draw_reversal_stimpair_tr_pre/_post.
  draw_context_tr    3 per-cue context (pre vs post) curves over time -- the
      analogue of decoding.draw_reversal_context_tr.

The real FIG2's bottom row 4th slot reuses the TIME-POOLED stim_identity bar
(decoding_pooled.draw_reversal_stimidentity_bar) rather than a new
time-resolved decoder -- see run_time_resolved_decoding.py's docstring for
why a time-resolved value/stim-identity decoder isn't computed here either.
compose.py's figure2_per_model() fills that slot with decoding.py's existing
draw_stimidentity_bar_single, matching the real figure2() layout exactly.
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
# Must match run_time_resolved_decoding.py's --out for this REV_TAG (e.g.
# REV_TAG=_5k python3 analysis/run_time_resolved_decoding.py --post
# .../figure_data_reversal_5k --out unified_figures/output/decoding_tr_5k).
REV_TAG = os.environ.get("REV_TAG", "")
HORIZON_LABEL = f" [{REV_TAG.lstrip('_')} horizon]" if REV_TAG else ""
TR_JSON = _HERE.parent / "output" / f"decoding_tr{REV_TAG}" / "time_resolved_decode.json"

PAIR_LABELS = {"0-1": "0% vs 50%", "0-2": "0% vs 100%", "1-2": "50% vs 100%"}
PAIR_COLOURS = {"0-1": "#bebada", "0-2": "#fb8072", "1-2": "#8dd3c7"}
STIM_LABELS = ["0%", "50%", "100%"]


def _load():
    return json.load(open(TR_JSON))


def _segment_lines(ax, bounds):
    """bounds = (n_iti, stim_ts, rew_ts); t is assumed already shifted so 0 =
    stimulus onset (see draw_stimpair_tr/draw_context_tr)."""
    n_iti, stim_ts, rew_ts = bounds
    ax.axvline(-0.5, color="0.6", ls=":", lw=1)
    ax.axvline(stim_ts - 0.5, color="0.6", ls=":", lw=1)


def draw_stimpair_tr(model_type, phase, results=None, ax=None):
    fig, ax, _ = new_panel(ax, figsize=(6, 4.5))
    results = results if results is not None else _load()
    r = results[model_type]
    key = "stim_pair_pre" if phase == "pre" else "stim_pair_post"
    d = r[key]
    t = np.arange(r["n_time"]) - r["bounds"][0]  # 0 = stimulus onset
    for pk, lab in PAIR_LABELS.items():
        c = d[pk]
        mean = np.array(c["mean"]); sem = np.array(c["sem"])
        ax.plot(t, mean, color=PAIR_COLOURS[pk], lw=2, label=lab)
        ax.fill_between(t, mean - sem, mean + sem, color=PAIR_COLOURS[pk], alpha=0.2, lw=0)
    ax.axhline(0.5, color="0.3", ls="--", lw=1, label="chance")
    _segment_lines(ax, r["bounds"])
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("Time from stim onset")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{F.MODELS[model_type]['label']}\nstimulus-pair decoding, {phase} "
                 f"(n={r['n_seeds']} seeds){HORIZON_LABEL}", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    return fig


def draw_stimpair_tr_pre(model_type, results=None, ax=None):
    return draw_stimpair_tr(model_type, "pre", results=results, ax=ax)


def draw_stimpair_tr_post(model_type, results=None, ax=None):
    return draw_stimpair_tr(model_type, "post", results=results, ax=ax)


def draw_context_tr(model_type, results=None, ax=None):
    fig, ax, _ = new_panel(ax, figsize=(6, 4.5))
    results = results if results is not None else _load()
    r = results[model_type]
    d = r["context"]
    t = np.arange(r["n_time"]) - r["bounds"][0]  # 0 = stimulus onset
    for s, lab in enumerate(STIM_LABELS):
        c = d[str(s)]
        mean = np.array(c["mean"]); sem = np.array(c["sem"])
        color = S.STIM_COLOURS[lab.rstrip("%")]
        ax.plot(t, mean, color=color, lw=2, label=f"{lab} cue")
        ax.fill_between(t, mean - sem, mean + sem, color=color, alpha=0.2, lw=0)
    ax.axhline(0.5, color="0.3", ls="--", lw=1, label="chance")
    _segment_lines(ax, r["bounds"])
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("Time from stim onset")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{F.MODELS[model_type]['label']}\ncontext (pre vs post) decoding, per cue "
                 f"(n={r['n_seeds']} seeds){HORIZON_LABEL}\n"
                 f"(near-ceiling even during ITI baseline = general pre/post model drift, not "
                 f"cue-specific coding -- see decoding.py's context_decode_baseline_control)",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    return fig


def build_all(show_tag=None):
    if not TR_JSON.exists():
        print(f"  (skip time-resolved decoding panels: {TR_JSON} not found -- "
              f"run analysis/run_time_resolved_decoding.py in cxval env)")
        return
    results = _load()
    for mt in MODEL_TYPES:
        if mt not in results:
            continue
        try:
            save_panel(draw_stimpair_tr_pre(mt, results=results), "CrossModel/Decoding",
                       f"DEC.stimpair_tr_pre{REV_TAG}.{mt}", f"{mt}_stimpair_tr_pre{REV_TAG}", show_tag)
            save_panel(draw_stimpair_tr_post(mt, results=results), "CrossModel/Decoding",
                       f"DEC.stimpair_tr_post{REV_TAG}.{mt}", f"{mt}_stimpair_tr_post{REV_TAG}", show_tag)
            save_panel(draw_context_tr(mt, results=results), "CrossModel/Decoding",
                       f"DEC.context_tr{REV_TAG}.{mt}", f"{mt}_context_tr{REV_TAG}", show_tag)
        except Exception as e:
            print(f"  (skip time-resolved decoding panels for {mt}: {e})")


if __name__ == "__main__":
    build_all()
