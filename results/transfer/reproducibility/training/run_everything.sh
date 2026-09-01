#!/usr/bin/env bash
# One-shot driver for everything pending from this session:
#
#   1. Real-data responsiveness re-extraction (neuronal_representations env)
#      -- FDR fix only; the paired-test change was investigated and
#      reverted (matches the original notebook's ttest_ind).
#   2. Pre-reversal checkpointed retrain (cxval env) -> model_runs_ckpt/,
#      a SEPARATE directory from your existing model_runs/ (so nothing
#      currently built from model_runs/ is touched or invalidated).
#   3. BOTH post-reversal horizons, warm-started from THOSE checkpointed
#      pre-reversal weights specifically (model_runs_ckpt/.../model.pt,
#      not model_runs/) -> model_runs_reversal_ckpt(_5k)/.
#
# Step 3 is NOT optional for a valid combined figure: the "pre" and "post"
# segments of a single trajectory have to come from the SAME model. Step 2
# alone would give you a pre-reversal run that's only "the same seed" as
# your existing model_runs -- not necessarily the same weights, since
# training is seeded but not guaranteed bit-reproducible on CPU (BLAS
# threading can change floating-point reduction order run to run). Step 3
# sidesteps that entirely by warm-starting from the literal checkpointed
# model.pt file rather than re-deriving it, so the lineage is exact by
# construction, not by hoping reproducibility holds.
#
# This produces a full PARALLEL lineage (model_runs_ckpt +
# model_runs_reversal_ckpt(_5k)) alongside your existing
# model_runs/model_runs_reversal(_5k) -- nothing existing is overwritten.
# Cost: step 3 is two full reversal-training passes (2500-trial +
# 5000-trial) over all 90 seed x model-type runs, on top of step 2's full
# pre-reversal retrain -- this is the expensive option, not the cheap one.
# Nothing currently rebuilds figure_data/unified_figures from this new
# lineage -- come back once training finishes and we'll wire that up.
#
# Each environment's own python binary is invoked DIRECTLY (no `conda
# activate` / `conda run` / `mamba run`) -- macOS's stock bash (3.2) chokes
# on the wrapper script `mamba run --no-capture-output` generates, so this
# sidesteps that entirely: every script here already accepts a PY=
# override (or, for the real-data script, was just updated to). You still
# don't need to activate anything yourself first.
#
# Usage:
#   bash run_everything.sh                    # all of 1 + 2 + 3
#   SKIP_REAL_DATA=1 bash run_everything.sh    # skip step 1
#   SKIP_PRETRAIN_CKPT=1 bash run_everything.sh   # skip steps 2 + 3 (real-data only)
#   SKIP_REV_2500=1 bash run_everything.sh     # skip just the 2500-trial horizon in step 3
#   SKIP_REV_5K=1 bash run_everything.sh       # skip just the 5000-trial horizon in step 3
#
# Override paths if your checkout layout differs from the defaults below:
#   NEURONAL_REPO, MODEL_REPO, MAMBA_ENVS_ROOT, CHECKPOINT_EVERY, SEEDS,
#   PRE_OUT, REV_OUT_2500, REV_OUT_5000

set -euo pipefail

NEURONAL_REPO="${NEURONAL_REPO:-$HOME/Documents/neuronal-representations}"
MODEL_REPO="${MODEL_REPO:-$HOME/Documents/context-value-RNNs}"
TRAINING_DIR="$MODEL_REPO/results/transfer/reproducibility/training"

# Where your conda/mamba envs actually live -- matches the path seen in
# your earlier error (~/.local/share/mamba/envs/cxval/...). Override if
# yours is elsewhere (e.g. ~/miniconda3/envs or ~/mambaforge/envs).
MAMBA_ENVS_ROOT="${MAMBA_ENVS_ROOT:-$HOME/.local/share/mamba/envs}"
CXVAL_PY="${CXVAL_PY:-$MAMBA_ENVS_ROOT/cxval/bin/python}"
NEURONAL_PY="${NEURONAL_PY:-$MAMBA_ENVS_ROOT/neuronal_representations/bin/python}"

CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-20}"
SEEDS="${SEEDS:-$(seq 42 71)}"
SKIP_REAL_DATA="${SKIP_REAL_DATA:-0}"
SKIP_PRETRAIN_CKPT="${SKIP_PRETRAIN_CKPT:-0}"
SKIP_REV_2500="${SKIP_REV_2500:-0}"
SKIP_REV_5K="${SKIP_REV_5K:-0}"
PRE_OUT="${PRE_OUT:-$MODEL_REPO/results/transfer/model_runs_ckpt}"
REV_OUT_2500="${REV_OUT_2500:-$MODEL_REPO/results/transfer/reversal/reversal_2500_ckpt}"
REV_OUT_5000="${REV_OUT_5000:-$MODEL_REPO/results/transfer/reversal/reversal_5000_ckpt}"

require_py() {
  local py="$1" label="$2"
  if [ ! -x "$py" ]; then
    echo "ERROR: $label python not found/executable at: $py" >&2
    echo "  Set MAMBA_ENVS_ROOT (or ${label}_PY directly) to the right path." >&2
    echo "  e.g. find it with:  find ~ -maxdepth 6 -type d -name envs 2>/dev/null" >&2
    exit 1
  fi
}

if [ "$SKIP_REAL_DATA" != "1" ]; then
  require_py "$NEURONAL_PY" NEURONAL
  echo "############################################################"
  echo "# 1: real-data responsiveness re-extraction (neuronal_representations env)"
  echo "############################################################"
  ( cd "$NEURONAL_REPO" && PY="$NEURONAL_PY" bash rerun_responsiveness_real_data.sh )
else
  echo ">> Skipping step 1 (SKIP_REAL_DATA=1)"
fi

if [ "$SKIP_PRETRAIN_CKPT" != "1" ]; then
  require_py "$CXVAL_PY" CXVAL
  echo
  echo "############################################################"
  echo "# 2: pre-reversal checkpointed retrain (cxval env)"
  echo "#    -> $PRE_OUT"
  echo "############################################################"
  ( cd "$TRAINING_DIR" \
    && PY="$CXVAL_PY" SEEDS="$SEEDS" OUT="$PRE_OUT" CHECKPOINT_EVERY="$CHECKPOINT_EVERY" \
       bash run_all.sh )

  echo
  echo "--- post-hoc pre-reversal cross-context convergence curve ---"
  ( cd "$TRAINING_DIR" \
    && "$CXVAL_PY" compute_pretrain_crosscontext_decode.py --runs-root "$PRE_OUT" )

  echo
  echo "############################################################"
  echo "# 3: post-reversal continuation, BOTH horizons, warm-started from"
  echo "#    THESE checkpointed pre-reversal weights (cxval env)"
  echo "############################################################"
  if [ "$SKIP_REV_2500" != "1" ]; then
    echo "--- 2500-trial reversal continuation -> $REV_OUT_2500 ---"
    ( cd "$TRAINING_DIR" \
      && PY="$CXVAL_PY" SEEDS="$SEEDS" RUNS="$PRE_OUT" OUT="$REV_OUT_2500" N_TRIALS=2500 \
         bash run_reversal_all.sh )
  else
    echo ">> Skipping 2500-trial horizon (SKIP_REV_2500=1)"
  fi

  if [ "$SKIP_REV_5K" != "1" ]; then
    echo "--- 5000-trial reversal continuation -> $REV_OUT_5000 ---"
    ( cd "$TRAINING_DIR" \
      && PY="$CXVAL_PY" SEEDS="$SEEDS" RUNS="$PRE_OUT" OUT="$REV_OUT_5000" N_TRIALS=5000 \
         bash run_reversal_all.sh )
  else
    echo ">> Skipping 5000-trial horizon (SKIP_REV_5K=1)"
  fi

  echo
  echo "New checkpointed lineage trained under:"
  echo "  $PRE_OUT"
  [ "$SKIP_REV_2500" != "1" ] && echo "  $REV_OUT_2500"
  [ "$SKIP_REV_5K" != "1" ] && echo "  $REV_OUT_5000"
  echo "Nothing rebuilds figure_data/unified_figures from this lineage yet --"
  echo "come back once this finishes and we'll wire that up (inference ->"
  echo "figure_data rebuild -> the actual combined pre+post panel)."
else
  echo ">> Skipping steps 2 + 3 (SKIP_PRETRAIN_CKPT=1)"
fi

echo
echo "############################################################"
echo "All requested steps complete."
echo "############################################################"
