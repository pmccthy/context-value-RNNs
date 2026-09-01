# reversal/reproducibility

`train_model.py`, `train_reversal.py`, `run_inference.py` here are copies
of `scripts/16_06_26_*.py` (repo root) -- confirmed (via diff) DIFFERENT
from the same-named scripts in `../../reproducibility/`. That sibling copy
is the baseline pre/post-reversal pipeline; this copy is what actually
produced every run under `../` (reversal_2500, reversal_5000, and all the
intervention studies: action_std_0p15_full, min_vigour_0p1_full,
squash_1p3_full, reward_scale_intervention, terminal_rpe,
terminal_value_minvig, and their pilots) -- it carries the extra CLI flags
those manipulations need. The originals stay in scripts/ too (untouched).

`../code/` also depends on `cxval/` (see `../../reproducibility/cxval/` --
kept once at the top level rather than duplicated here).
