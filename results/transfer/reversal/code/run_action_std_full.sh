#!/usr/bin/env bash
# Full 30-seed action_std=0.15 reversal study (no min_vigour floor -- isolates the
# exploration-noise intervention from the earlier floor one): original training ->
# inference -> reversal continue-training -> inference -> all the usual bundle plots.
#
# infer_value/infer_rpe are now BOTH wired into make_probe (cxval/vigour.py,
# train_model.py, train_reversal.py), so this run logs GENUINE training-time
# probe_vigour, probe_rpe, AND probe_value -- unlike the min_vigour=0.1 job, no
# terminal-model workaround is needed here for anything; seed_groups.py's main()
# already calls fig_seed_curves/_avg for all three (vigour, rpe, value) automatically.
#
# Uses the SAME seeds (42-71) as the baseline and min_vigour=0.1 studies, so all three
# are directly seed-for-seed comparable -- only action_std (0.05 -> 0.15) and
# min_vigour (0.0 throughout) differ from baseline.
#
# Runtime: same scale as the min_vigour_0p1_full run (90 original + 90 reversal
# training calls) -- HOURS on CPU. The 5-seed pilot already took many sequential long
# training calls even at the smaller scale. Run this in the background and check on it
# periodically -- every step skips work it already did, so it's safe to stop and
# restart.
#
# Usage:
#   cd results/transfer/reversal
#   nohup bash code/run_action_std_full.sh > /tmp/action_std_full.log 2>&1 &
#   tail -f /tmp/action_std_full.log                    # watch progress
#   find results/action_std_0p15_full -name model.pt | wc -l   # count completed runs (max 180)
set -euo pipefail
cd "$(dirname "$0")/.."                 # -> reversal_study/
ROOT="$(cd ../../.. && pwd)"            # repo root, for scripts/16_06_26_*.py (now 3 levels down: results/transfer/reversal/)

OUT=results/action_std_0p15_full
ACTION_STD=0.15
N_TRIALS_ORIG=2500
N_TRIALS_REV=5000
SEEDS=$(seq 42 71)
TYPES="classif_rl rl_only classif_rl_readout_only"

mkdir -p "$OUT"

echo "== original training (action_std=$ACTION_STD) =="
for MT in $TYPES; do
  for S in $SEEDS; do
    DIR="$OUT/model_runs/$MT/seed$S"
    if [ -f "$DIR/model.pt" ]; then echo "  skip $MT seed$S (original, already done)"; continue; fi
    echo "  -- $MT seed$S --"
    python3 "$ROOT/scripts/16_06_26_train_model.py" --model-type "$MT" --seed "$S" \
        --n-trials "$N_TRIALS_ORIG" --action-std "$ACTION_STD" --probe-every 20 --out "$DIR"
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
python3 code/recovery_time.py \
    --pre-runs "$OUT/model_runs" --post-runs "$OUT/model_runs_reversal" \
    --out "$OUT/figures_seed_groups"

echo "DONE. Figures in $OUT/figures_* include seed_vigour_curves, seed_rpe_curves,"
echo "AND seed_value_curves (+ _avg versions) -- all genuine training-time trajectories,"
echo "no terminal-model workaround needed for this run."
