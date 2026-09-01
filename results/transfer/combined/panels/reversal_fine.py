"""Fine-grained (powerset) responder-group Sankey diagrams, pre -> post
reversal, ported from neuronal-representations' draw_sankey_fine_combined.

Uses the model's EXISTING window-mean paired-t-test responder definition
(figure_data.pkl's "responsive" field) -- NOT the new temporal-cluster port in
model_responders.py, whose calibration against the model's much larger
per-stimulus trial count is still an open question (see chat). Swap the
`_pre_post_group_labels` data source once that's settled -- everything else
here (the fine taxonomy, the Sankey drawing) is independent of which
responder test feeds it.
"""
from __future__ import annotations

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
from sankey_utils import draw_sankey  # noqa: E402

REV_TAG = os.environ.get("REV_TAG", "")  # "" = 2500-trial reversal run, "_5k" = the longer one
HORIZON_LABEL = f" [{REV_TAG.lstrip('_')} horizon]" if REV_TAG else ""

GROUP_LABELS = list(S.GROUP_COLOURS.keys())   # 7 groups + "non-responsive", in order
_GROUP_ORDER = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
_GROUP_CODES = [sum(1 << i for i in g) for g in _GROUP_ORDER]
_BITS = np.array([1, 2, 4])
PRETTY = {
    "0%-only": "0%", "50%-only": "50%", "100%-only": "100%",
    "0% & 50%": "0%&50%", "0% & 100%": "0%&100%", "50% & 100%": "50%&100%",
    "all three": "all 3", "non-responsive": "non-resp",
}


def _classify(code):
    for label, c in zip(GROUP_LABELS[:-1], _GROUP_CODES):
        if code == c:
            return label
    return "non-responsive"


def _pre_post_group_labels(Dpre, Dpost, model_type):
    """Per-unit fine-group label, pre and post, pooled over all present seeds.
    Returns (labels_pre, labels_post), parallel arrays over (seed, unit)."""
    ti_pre = Dpre["model_types"].index(model_type)
    ti_post = Dpost["model_types"].index(model_type)
    seeds = [s for s in Dpre["seeds_present"][model_type] if s in Dpost["seeds_present"][model_type]]
    seed_idx_pre = [Dpre["seeds"].index(s) for s in seeds]
    seed_idx_post = [Dpost["seeds"].index(s) for s in seeds]
    resp_pre = Dpre["responsive"]["data"][ti_pre][seed_idx_pre]     # (n_seed, unit, stim)
    resp_post = Dpost["responsive"]["data"][ti_post][seed_idx_post]
    codes_pre = resp_pre.astype(int) @ _BITS      # (n_seed, unit)
    codes_post = resp_post.astype(int) @ _BITS
    labels_pre = np.vectorize(_classify)(codes_pre)
    labels_post = np.vectorize(_classify)(codes_post)
    return labels_pre.ravel(), labels_post.ravel()


def draw_fine_sankey(model_type, Dpre=None, Dpost=None, ax=None):
    fig, ax, _ = new_panel(ax, figsize=(7.5, 6.5))
    Dpre = Dpre if Dpre is not None else F.load(str(_HERE.parent / "transfer" / "figure_data"))
    Dpost = Dpost if Dpost is not None else F.load(str(_HERE.parent / "transfer" / f"figure_data_reversal{REV_TAG}"))
    labels_pre, labels_post = _pre_post_group_labels(Dpre, Dpost, model_type)
    n = len(GROUP_LABELS)
    idx = {label: i for i, label in enumerate(GROUP_LABELS)}
    M = np.zeros((n, n))
    for lp, lq in zip(labels_pre, labels_post):
        M[idx[lp], idx[lq]] += 1
    draw_sankey(ax, M, GROUP_LABELS, f"{F.MODELS[model_type]['label']}: pre -> post reversal "
                f"(fine groups, n={len(labels_pre)} units){HORIZON_LABEL}",
                col_map=S.GROUP_COLOURS, pretty_map=PRETTY)
    return fig


def build_all(show_tag=None):
    Dpre = F.load(str(_HERE.parent / "transfer" / "figure_data"))
    Dpost = F.load(str(_HERE.parent / "transfer" / f"figure_data_reversal{REV_TAG}"))
    for mt in ["rl_only", "classif_rl", "classif_rl_readout_only"]:
        try:
            fig = draw_fine_sankey(mt, Dpre=Dpre, Dpost=Dpost)
        except Exception as e:
            print(f"  (skip fine sankey for {mt}: {e})")
            continue
        save_panel(fig, "Transfer/Subgroups", f"TRANSFER.finesankey{REV_TAG}.{mt}",
                   f"{mt}_fine_sankey_pre_post{REV_TAG}", show_tag)


if __name__ == "__main__":
    build_all()
