#!/usr/bin/env bash
# Full 30-seed min_vigour=0.1 reversal study: original training -> inference ->
# reversal continue-training -> inference -> all the usual bundle plots (incl. a
# GENUINE training-time RPE curve, unlike the terminal_rpe.py workaround used for the
# already-completed baseline runs, since min_vigour-instrumented training now logs
# probe_rpe as it goes -- see cxval/vigour.py's infer_rpe + train_model.py/
# train_reversal.py's make_probe).
#
# Uses the SAME seeds (42-71) as the existing baseline (min_vigour=0, "without the
# clamping") 30-seed study, so results are directly seed-for-seed comparable -- only
# min_vigour differs. Reversal budget matches the study's current live budget
# (results/reversal_5000, 5000 trials); change N_TRIALS_REV below for 2500 instead.
#
# Runtime: 90 training runs (30 seeds x 3 model classes), original (2500 trials) +
# reversal (5000 trials) each -- this is the same scale as the existing baseline study
# and will take HOURS on CPU, not minutes. The 5-seed pilot (15 runs) already took
# multiple long training calls; scale accordingly. Run this in the background and
# check on it periodically -- it's safe to re-run (each step skips work it already did).
#
# Usage:
#   cd results/transfer/reversal
#   nohup bash code/run_min_vigour_full.sh > /tmp/min_vigour_full.log 2>&1 &
#   tail -f /tmp/min_vigour_full.log                 # watch progress
#   find results/min_vigour_0p1_full -name model.pt | wc -l   # count completed runs (max 180)
set -euo pipefail
cd "$(dirname "$0")/.."                 # -> reversal_study/
ROOT="$(cd ../../.. && pwd)"            # repo root, for scripts/16_06_26_*.py (now 3 levels down: results/transfer/reversal/)

OUT=results/min_vigour_0p1_full
MIN_VIGOUR=0.1
N_TRIALS_ORIG=2500
N_TRIALS_REV=5000
SEEDS=$(seq 42 71)
TYPES="classif_rl rl_only classif_rl_readout_only"

mkdir -p "$OUT"

echo "== original training (min_vigour=$MIN_VIGOUR) =="
for MT in $TYPES; do
  for S in $SEEDS; do
    DIR="$OUT/model_runs/$MT/seed$S"
    if [ -f "$DIR/model.pt" ]; then echo "  skip $MT seed$S (original, already done)"; continue; fi
    echo "  -- $MT seed$S --"
    python3 "$ROOT/scripts/16_06_26_train_model.py" --model-type "$MT" --seed "$S" \
        --n-trials "$N_TRIALS_ORIG" --min-vigour "$MIN_VIGOUR" --probe-every 20 --out "$DIR"
  done
done

echo "== inference: pre-reversal models =="
python3 "$ROOT/scripts/16_06_26_run_inference.py" --runs "$OUT/model_runs" \
    --out "$OUT/figure_data"

echo "== reversal continue-training =="
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

echo "DONE. Figures (incl. seed_rpe_curves — a REAL training-time RPE trajectory this"
echo "time, not the terminal-model workaround) are in $OUT/figures_*"
