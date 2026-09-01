"""Modular figure-panel framework, ported from
~/Documents/neuronal-representations/results/transfer/figures/panels/_tags.py
so model figures use the same draw_*(ax=...) / save_panel / compose pattern
as the experimental figures.

Each individual figure is a small self-contained draw_*(ax=...) function (see
panels/*.py, to be filled in per the figure list) so it can be saved standalone
OR dropped into a shared gridspec axes for a multi-panel composite later (see
../compose.py) -- exactly like the experimental repo's panels/behaviour.py,
panels/population.py, etc.

Every panel has a short tag (e.g. "POP.TRANSFER.mean") that is (1) put in the
output filename and (2) stamped on the figure itself (toggle with SHOW_TAG).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

STYLE_DIR = Path(__file__).resolve().parent.parent / "style"
sys.path.insert(0, str(STYLE_DIR))
plt.style.use(str(STYLE_DIR / "model_figure_style.mplstyle"))

# make unified_figures/style (condition_style, STIM_COLOURS, ...) importable
# from panel modules, same way the experimental repo's panels import _common.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIG_ROOT = Path(__file__).resolve().parent.parent / "output"   # .../unified_figures/output

SHOW_TAG = True   # global default; individual saves can override


def _capitalise(label):
    return (label[0].upper() + label[1:]) if label else label


def finalize_axes(fig, force_remove_titles=False):
    """Shared conventions for every axes in a figure: capitalise x/y labels,
    drop titles on single-panel figures and composites (but keep them on
    heatmap axes with an image, and on multi-panel per-seed grids), and hide
    the top/right spines -- matching the real repo's own convention, applied
    on essentially every panel there (ax.spines[["top","right"]].set_visible
    (False)) but never actually ported here until now."""
    single = len(fig.axes) <= 2
    for ax in fig.axes:
        has_image = bool(ax.get_images())
        remove = (force_remove_titles or single) and not has_image
        if remove:
            ax.set_title("")
        ax.set_xlabel(_capitalise(ax.get_xlabel()))
        ax.set_ylabel(_capitalise(ax.get_ylabel()))
        if not has_image:
            ax.spines[["top", "right"]].set_visible(False)


def new_panel(ax=None, figsize=(6.0, 4.0)):
    """Return (fig, ax, owns_fig). If ax is None a new fig/ax is created."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax, True
    return ax.figure, ax, False


def stamp_tag(fig, tag, show=None):
    show = SHOW_TAG if show is None else show
    if show and tag:
        fig.text(0.006, 0.994, tag, ha="left", va="top", fontsize=7,
                 color="0.4", family="monospace", zorder=1000)


def _norm_folder(folder):
    parts = folder.split("/")
    return "/".join(p.lower().replace(" ", "_").replace("-", "_") for p in parts)


def _norm_tag(tag):
    return tag.replace(".", "_")


def save_panel(fig, folder, tag, name, show_tag=None):
    """Save a figure into output/<folder>/<tag>__<name>.png with its tag."""
    folder = _norm_folder(folder)
    tag = _norm_tag(tag)
    out_dir = FIG_ROOT / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    finalize_axes(fig)
    stamp_tag(fig, tag, show_tag)
    path = out_dir / f"{tag}__{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {tag:24} -> {folder}/{path.name}")
    return path
