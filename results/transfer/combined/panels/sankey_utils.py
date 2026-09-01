"""Hand-rolled Sankey diagram primitives, ported VERBATIM from
~/Documents/neuronal-representations/results/transfer/figures/panels/subgroups.py
(_bezier_band, _draw_sankey) -- no external Sankey library, just matplotlib
Rectangle nodes + cubic-Bezier PathPatch ribbons sized by flow count.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
import matplotlib.patches as mpatches


def _bezier_band(ax, x0, y0a, y0b, x1, y1a, y1b, color, alpha=0.45):
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a),
             (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(mpatches.PathPatch(MPath(verts, codes), facecolor=color,
                                    edgecolor="none", alpha=alpha))


def draw_sankey(ax, M, cats, title, col_map, pretty_map):
    """M: (len(cats), len(cats)) flow-count matrix, M[i,j] = # units in
    pre-category i that end up in post-category j. cats: category keys in
    row/col order. col_map/pretty_map: {cat: colour/display-label}."""
    total = M.sum()
    gap = 0.02 * total if total else 1
    node_w = 0.13

    def extents(sizes):
        ys = {}
        y = 0.0
        for i, s in enumerate(sizes):
            ys[i] = (y, y + s)
            y += s + gap
        return ys, y

    left_sizes = M.sum(axis=1); right_sizes = M.sum(axis=0)
    Ly, Ltop = extents(left_sizes); Ry, Rtop = extents(right_sizes)
    top = max(Ltop, Rtop)
    for i, c in enumerate(cats):
        y0, y1 = Ly[i]
        ax.add_patch(plt.Rectangle((0, top - y1), node_w, y1 - y0, color=col_map[c]))
        ax.text(-0.02, top - (y0 + y1) / 2, pretty_map[c], ha="right", va="center", fontsize=10)
        y0, y1 = Ry[i]
        ax.add_patch(plt.Rectangle((1 - node_w, top - y1), node_w, y1 - y0, color=col_map[c]))
        ax.text(1.02, top - (y0 + y1) / 2, pretty_map[c], ha="left", va="center", fontsize=10)
    lcur = {i: Ly[i][0] for i in range(len(cats))}
    rcur = {j: Ry[j][0] for j in range(len(cats))}
    for i in range(len(cats)):
        for j in range(len(cats)):
            f = M[i, j]
            if f <= 0:
                continue
            y0a = top - lcur[i]; y0b = top - (lcur[i] + f); lcur[i] += f
            y1a = top - rcur[j]; y1b = top - (rcur[j] + f); rcur[j] += f
            _bezier_band(ax, node_w, y0a, y0b, 1 - node_w, y1a, y1b, col_map[cats[i]])
    ax.set_xlim(-0.28, 1.28); ax.set_ylim(-0.16 * top, top * 1.05)
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    ax.text(node_w / 2, -0.09 * top, "pre", fontsize=16, ha="center", va="center")
    ax.text(1 - node_w / 2, -0.09 * top, "post", fontsize=16, ha="center", va="center")
