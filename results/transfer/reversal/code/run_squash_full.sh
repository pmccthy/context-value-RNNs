#!/usr/bin/env bash
# Full 30-seed smooth soft-clamp squash reversal study (squash_width=1.3, no min_vigour
# floor, default action_std=0.05 -- isolates the squash intervention from the earlier
# floor/action_std ones). Original training -> inference -> reversal continue-training
# -> inference -> all the usual bundle plots.
#
# squash_width replaces the SCORE policy's hard clamp with a smooth soft-clamp sigmoid
# (v = sigmoid(a/width - 3)) applied to the sampled action -- the score-function/
# advantage/critic/TD-bootstrapping objective is completely untouched, only how a raw
# sample maps to an executed vigour changes. See VigourActorCritic.squash_width's
# docstring in cxval/vigour.py for the full design rationale and the width-vs-gradient-
# reach tradeoff. train_reversal.py inherits squash_width automatically from the
# original run's config.json -- no separate flag needed for the reversal phase.
#
# Uses the SAME seeds (42-71) as the baseline/min_vigour/action_std studies, so all four
# are directly seed-for-seed comparable -- only squash_width (None -> 1.3) differs from
# baseline. Reversal budget matches baseline's reversal_5000 EXACTLY (5000 trials) --
# an earlier 8-seed/2500-trial pilot showed the smooth squash needs the full budget to
# fairly compare against the hard clamp (its in-range gradient is genuinely gentler, so
# even easy seeds need more updates to fully converge).
#
# Runtime: same scale as the other full studies (90 original + 90 reversal training
# calls) -- HOURS on CPU. Run this in the background and check on it periodically --
# every step skips work it already did, so it's safe to stop and restart.
#
# Usage:
#   cd results/transfer/reversal
#   nohup bash code/run_squash_full.sh > /tmp/squash_full.log 2>&1 &
#   tail -f /tmp/squash_full.log                    # watch progress
#   find results/squash_1p3_full -name model.pt | wc -l   # count completed runs (max 180)
set -euo pipefail
cd "$(dirname "$0")/.."                 # -> reversal_study/
ROOT="$(cd ../../.. && pwd)"            # repo root, for scripts/16_06_26_*.py (now 3 levels down: results/transfer/reversal/)

OUT=results/squash_1p3_full
SQUASH_WIDTH=1.3
N_TRIALS_ORIG=2500
N_TRIALS_REV=5000
SEEDS=$(seq 42 71)
TYPES="classif_rl rl_only classif_rl_readout_only"

mkdir -p "$OUT"

echo "== original training (squash_width=$SQUASH_WIDTH) =="
for MT in $TYPES; do
  for S in $SEEDS; do
    DIR="$OUT/model_runs/$MT/seed$S"
    if [ -f "$DIR/model.pt" ]; then echo "  skip $MT seed$S (original, already done)"; continue; fi
    echo "  -- $MT seed$S --"
    python3 "$ROOT/scripts/16_06_26_train_model.py" --model-type "$MT" --seed "$S" \
        --n-trials "$N_TRIALS_ORIG" --squash-width "$SQUASH_WIDTH" --probe-every 20 --out "$DIR"
  done
done

echo "== inference: pre-reversal models =="
python3 "$ROOT/scripts/16_06_26_run_inference.py" --runs "$OUT/model_runs" \
    --out "$OUT/figure_data"

echo "== reversal continue-training (squash_width inherited from config.json) =="
for MT in $TYPES; do
  for S in $SEEDS; do
    SRC="$OUT/model_runs/$MT/seed$S"
    DIR="$OUT/model_runs_reversal/$MT/seed$S"
    if [ -f "$DIR/model.pt" ]; then echo "  skip $MT seed$S (reversal, already done)"; continue; fi
    echo "  -- $MT seed$S --"
    python3 "$ROOT/scripts/16_06_26_train_reversal.py" --run "$SRC" --out "$DIR" \
        --n-trials "$N_TRIALS_REV" --probe-every 20
  done
done

echo "== inference: post-reversal models =="
python3 "$ROOT/scripts/16_06_26_run_inference.py" --runs "$OUT/model_runs_reversal" \
    --out "$OUT/figure_data_reversal"

echo "== usual bundle plots =="
python3 code/reversal_analysis.py \
    --pre "$OUT/figure_data" --post "$OUT/figure_data_reversal" \
    --reversal-runs "$OUT/model_runs_reversal" --pre-runs "$OUT/model_runs" \
    --out "$OUT/figures_reversal"
python3 code/population_similarity.py \
    --pre "$OUT/figure_data" --post "$OUT/figure_data_reversal" \
    --out "$OUT/figures_population_similarity"
python3 code/rsa.py --data "$OUT/figure_data_reversal" --out "$OUT/figures_rsa"
python3 code/seed_groups.py \
    --pre "$OUT/figure_data" --post "$OUT/figure_data_reversal" \
    --reversal-runs "$OUT/model_runs_reversal" --pre-runs "$OUT/model_runs" \
    --out "$OUT/figures_seed_groups"
python3 code/recovery_time.py \
    --pre-runs "$OUT/model_runs" --post-runs "$OUT/model_runs_reversal" \
    --out "$OUT/figures_seed_groups"

echo "DONE. Compare results/squash_1p3_full/figures_seed_groups/seed_recovery_strip.png"
echo "(and seed_vigour_curves.png for the readout-only column specifically) against"
echo "results/action_std_0p15_full and results/reversal_5000 to see whether the smooth"
echo "squash actually improves genuine (non-trivial) recovery once given a matched"
echo "5000-trial reversal budget."
