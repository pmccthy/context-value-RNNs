"""Draws the noise-robustness-of-stimulus-decoding panels -- RNN models.

Reads the JSON cache written by analysis/run_noise_robustness.py (per
model_type, per seed: overall/per-stimulus decode accuracy across a sweep
of test-set activation noise magnitudes, plus a row-normalized confusion
matrix at one shared representative noise magnitude) -- see that script's
module docstring for the full methodology (train on clean activations,
corrupt only the held-out test split, per-unit-std-scaled iid Gaussian
noise, per-seed-then-averaged aggregation).

  draw_noise_accuracy(model_type, ax=None)
      Decoding accuracy vs. noise magnitude, one line per stimulus (plus a
      black overall-accuracy line), mean +/- SEM across seeds.

  draw_noise_confusion(model_type, ax=None)
      Row-normalized confusion matrix at the cache's shared
      CONFUSION_LEVEL (chosen post-hoc in run_noise_robustness.py as a
      level with visible but non-saturating degradation for every model
      type), mean across seeds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "style"))
sys.path.insert(0, str(_HERE.parent / "transfer" / "code"))

from _tags import new_panel, save_panel  # noqa: E402
import style as S  # noqa: E402
import figure_config as FC  # noqa: E402

MODEL_TYPES = ["rl_only", "classif_rl", "classif_rl_readout_only"]
NOISE_JSON = _HERE.parent / "figs" / "noise_robustness" / "noise_robustness_summary.json"

_CACHE: dict = {}


def _load():
    if "d" not in _CACHE:
        if not NOISE_JSON.exists():
            raise FileNotFoundError(
                f"{NOISE_JSON} not found -- run analysis/run_noise_robustness.py in cxval env")
        with open(NOISE_JSON) as f:
            _CACHE["d"] = json.load(f)
    return _CACHE["d"]


def _stim_colour(stim_label: str) -> str:
    """stim_label is like '0%'/'50%'/'100%' -- style.STIM_COLOURS keys on
    the bare number ('0'/'50'/'100')."""
    return S.STIM_COLOURS[stim_label.rstrip("%")]


def draw_noise_accuracy(model_type, ax=None):
    fig, ax, _ = new_panel(ax, figsize=(5.2, 4.4))
    d = _load()
    levels = np.asarray(d["noise_levels"], dtype=float)
    stim_labels = d["stim_labels"]
    seeds = d["models"][model_type]
    n_seeds = len(seeds)
    n_stim = len(stim_labels)
    chance = 1.0 / n_stim

    overall = np.array([seeds[s]["overall_acc"] for s in seeds])   # (n_seeds, n_levels)
    om, oe = overall.mean(axis=0), overall.std(axis=0, ddof=1) / np.sqrt(n_seeds)
    ax.plot(levels, om, color="0.15", lw=2.2, label="overall", zorder=3)
    ax.fill_between(levels, om - oe, om + oe, color="0.15", alpha=0.15, lw=0, zorder=3)

    for si, stim_label in enumerate(stim_labels):
        vals = np.array([seeds[s]["per_stim_acc"][str(si)] for s in seeds])   # (n_seeds, n_levels)
        m, e = vals.mean(axis=0), vals.std(axis=0, ddof=1) / np.sqrt(n_seeds)
        colour = _stim_colour(stim_label)
        ax.plot(levels, m, color=colour, lw=1.8, label=f"stim {stim_label}")
        ax.fill_between(levels, m - e, m + e, color=colour, alpha=0.18, lw=0)

    ax.axhline(chance, color="0.5", lw=0.9, ls=":", zorder=1)
    ax.axvline(d.get("confusion_level"), color="0.6", lw=0.9, ls="--", zorder=1)

    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("noise SD")
    ax.set_ylabel("decoding accuracy")
    ax.set_ylim(0.0, 1.03)
    ax.set_xlim(levels[0], levels[-1])
    ax.spines[["top", "right"]].set_visible(False)
    label = FC.MODELS.get(model_type, {}).get("label", model_type)
    ax.set_title(f"{label}\n(n={n_seeds} seeds, {seeds[list(seeds)[0]]['n_test']} test trials/seed)",
                 fontsize=10)
    return fig


def draw_noise_confusion(model_type, ax=None, colorbar=None):
    fig, ax, owns = new_panel(ax, figsize=(4.6, 4.2))
    if colorbar is None:
        colorbar = owns
    d = _load()
    level = d["confusion_level"]
    stim_labels = d["stim_labels"]
    n_stim = len(stim_labels)
    seeds = d["models"][model_type]

    mats = np.array([seeds[s]["confusion"][str(level)] for s in seeds])   # (n_seeds, n_stim, n_stim)
    mean = mats.mean(axis=0)

    im = ax.imshow(mean, vmin=0, vmax=1, cmap="Blues")
    for i in range(n_stim):
        for j in range(n_stim):
            ax.text(j, i, f"{mean[i, j]:.2f}", ha="center", va="center", fontsize=9,
                     color="white" if mean[i, j] > 0.5 else "black")
    ax.set_xticks(range(n_stim)); ax.set_xticklabels(stim_labels)
    ax.set_yticks(range(n_stim)); ax.set_yticklabels(stim_labels)
    ax.set_xlabel("Predicted stimulus")
    ax.set_ylabel("True stimulus")
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of trials")
    label = FC.MODELS.get(model_type, {}).get("label", model_type)
    ax.set_title(f"{label}: confusion @ noise={level:g}\n(n={len(seeds)} seeds, mean)", fontsize=10)
    return fig


def build_all(show_tag=None):
    for mt in MODEL_TYPES:
        try:
            save_panel(draw_noise_accuracy(mt), "Mechanistic/NoiseRobustness",
                       f"NOISE.acc.{mt}", f"{mt}_noise_decode_accuracy", show_tag)
        except FileNotFoundError as e:
            print(f"  (skip noise accuracy for {mt}: {e})")
            return
        try:
            save_panel(draw_noise_confusion(mt), "Mechanistic/NoiseRobustness",
                       f"NOISE.conf.{mt}", f"{mt}_noise_confusion", show_tag)
        except FileNotFoundError as e:
            print(f"  (skip noise confusion for {mt}: {e})")


if __name__ == "__main__":
    build_all()
