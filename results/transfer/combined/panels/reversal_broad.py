"""BROAD (3-category, winner-take-all preferred-stimulus) pre->post reversal
Sankey, matching neuronal-representations' draw_sankey_combined -- companion
to reversal_fine.py's FINE (mixed-selectivity powerset) Sankey. Uses the same
existing window-mean responder test (figure_data.pkl's "responsive"/"tuning"
fields), via analysis/model_group_categories.py's transition_matrix_broad.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "style"))
sys.path.insert(0, str(_HERE.parent / "transfer" / "code"))
sys.path.insert(0, str(_HERE.parent / "analysis"))

from _tags import new_panel, save_panel  # noqa: E402
import style as S  # noqa: E402
import figures as F  # noqa: E402
import model_group_categories as MG  # noqa: E402
from sankey_utils import draw_sankey  # noqa: E402

REV_TAG = os.environ.get("REV_TAG", "")  # "" = 2500-trial reversal run, "_5k" = the longer one
HORIZON_LABEL = f" [{REV_TAG.lstrip('_')} horizon]" if REV_TAG else ""

BROAD_LABELS = ["non-responsive", "0%", "50%", "100%"]
BROAD_COLOURS = {"non-responsive": "0.7", "0%": S.STIM_COLOURS["0"],
                  "50%": S.STIM_COLOURS["50"], "100%": S.STIM_COLOURS["100"]}
PRETTY = {c: c for c in BROAD_LABELS}


def draw_broad_sankey(model_type, Dpre=None, Dpost=None, ax=None):
    fig, ax, _ = new_panel(ax, figsize=(6, 5.5))
    Dpre = Dpre if Dpre is not None else F.load(str(_HERE.parent / "transfer" / "figure_data"))
    Dpost = Dpost if Dpost is not None else F.load(str(_HERE.parent / "transfer" / f"figure_data_reversal{REV_TAG}"))
    M = MG.transition_matrix_broad(Dpre, Dpost, model_type)   # (4,4), silent-first
    n = int(M.sum())
    draw_sankey(ax, M, BROAD_LABELS,
                f"{F.MODELS[model_type]['label']}: pre -> post reversal "
                f"(broad groups, n={n} units){HORIZON_LABEL}",
                col_map=BROAD_COLOURS, pretty_map=PRETTY)
    return fig


def build_all(show_tag=None):
    Dpre = F.load(str(_HERE.parent / "transfer" / "figure_data"))
    Dpost = F.load(str(_HERE.parent / "transfer" / f"figure_data_reversal{REV_TAG}"))
    for mt in ["rl_only", "classif_rl", "classif_rl_readout_only"]:
        try:
            fig = draw_broad_sankey(mt, Dpre=Dpre, Dpost=Dpost)
        except Exception as e:
            print(f"  (skip broad sankey for {mt}: {e})")
            continue
        save_panel(fig, "Transfer/Subgroups", f"TRANSFER.broadsankey{REV_TAG}.{mt}",
                   f"{mt}_broad_sankey_pre_post{REV_TAG}", show_tag)


if __name__ == "__main__":
    build_all()
