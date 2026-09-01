#!/usr/bin/env bash
# CAUSAL INTERVENTION on the 0%->100% vs 100%->0% relearning-speed asymmetry: re-run
# ONLY the reversal phase (warm-started from the already-trained action_std_0p15_full
# original models -- no need to retrain those) under three conditions that directly
# manipulate the realized-reward term feeding each stimulus's RPE:
#
#   control    reward_scale = (1, 1, 1)   -- unmodified, for a fair same-seed baseline
#              (NOT the same run as action_std_0p15_full's own reversal -- this uses a
#              fresh optimizer/RNG draw too, so compare against THIS control, not the
#              original study, to isolate the effect of the scaling itself)
#   boost_up   reward_scale = (2, 1, 1)   -- DOUBLE the realized reward (and hence RPE)
#              for the 0%->100% stimulus specifically. Prediction if RPE size is truly
#              driving the slow 0->100 relearning: this should speed it up.
#   damp_down  reward_scale = (1, 1, 0.3) -- CUT the realized reward for the 100%->0%
#              stimulus to 30%. Prediction: this should slow down the (normally fast)
#              100->0 relearning.
#
# Order is [0%-stim's scale, 50%-stim's, 100%-stim's] in the REVERSED task's own value_
# matrix, i.e. index 0 = the originally-0%, now-100%-valued ("newly valuable") stimulus,
# index 2 = the originally-100%, now-0%-valued ("newly worthless") stimulus -- see
# --reward-scale's help text in train_reversal.py for the full explanation.
#
# --track-gradients is on for all three conditions, so you also get, per update:
# per-parameter-group gradient norms (backbone/vigour_head/value_head), the raw
# policy_loss/value_loss scalars, and an analytic per-stimulus REINFORCE-score magnitude
# -- all in history.json, no extra plotting pipeline needed to just look at the numbers.
#
# Runtime: 3 conditions x 3 model classes x SEEDS reversal-only training calls (each the
# same cost as one action_std_0p15_full reversal run) -- much cheaper than a full study
# since original training is skipped entirely. Still likely HOURS for the full 30 seeds;
# SEEDS below defaults to a 8-seed pilot -- widen it once you've checked the effect is
# there and worth the full seed count.
#
# Usage:
#   cd results/transfer/reversal
#   nohup bash code/run_reward_scale_intervention.sh > /tmp/reward_scale_intervention.log 2>&1 &
#   tail -f /tmp/reward_scale_intervention.log
set -euo pipefail
cd "$(dirname "$0")/.."                 # -> reversal_study/
ROOT="$(cd ../../.. && pwd)"            # repo root, for scripts/16_06_26_*.py (now 3 levels down: results/transfer/reversal/)

SRC=results/action_std_0p15_full/model_runs     # already-trained ORIGINAL models (unchanged)
OUT_ROOT=results/reward_scale_intervention
N_TRIALS_REV=5000
SEEDS=$(seq 42 49)                      # 8-seed pilot; widen to `seq 42 71` for the full 30
TYPES="classif_rl rl_only classif_rl_readout_only"

CONDITION_NAMES="control boost_up damp_down"

for COND in $CONDITION_NAMES; do
  # macOS ships bash 3.2 (no associative arrays), so the condition -> scale mapping
  # is a plain case statement instead of `declare -A` -- portable everywhere.
  case "$COND" in
    control)   SCALE="1.0 1.0 1.0" ;;
    boost_up)  SCALE="2.0 1.0 1.0" ;;
    damp_down) SCALE="1.0 1.0 0.3" ;;
  esac
  OUT="$OUT_ROOT/$COND"
  echo "== condition: $COND  (reward_scale = $SCALE) =="
  for MT in $TYPES; do
    for S in $SEEDS; do
      SRC_DIR="$SRC/$MT/seed$S"
      DIR="$OUT/model_runs_reversal/$MT/seed$S"
      if [ -f "$DIR/model.pt" ]; then echo "  skip $MT seed$S ($COND, already done)"; continue; fi
      if [ ! -f "$SRC_DIR/model.pt" ]; then echo "  ! missing source model $SRC_DIR, skipping"; continue; fi
      echo "  -- $MT seed$S --"
      python3 "$ROOT/scripts/16_06_26_train_reversal.py" --run "$SRC_DIR" --out "$DIR" \
          --n-trials "$N_TRIALS_REV" --probe-every 20 --track-gradients --reward-scale $SCALE
    done
  done
  echo "  -- inference + onset probe + usual diagnostics ($COND) --"
  python3 "$ROOT/scripts/16_06_26_run_inference.py" --runs "$OUT/model_runs_reversal" \
      --out "$OUT/figure_data_reversal"
  python3 code/reversal_onset_probe.py --post-runs "$OUT/model_runs_reversal"
  python3 code/seed_groups.py \
      --pre results/action_std_0p15_full/figure_data --post "$OUT/figure_data_reversal" \
      --reversal-runs "$OUT/model_runs_reversal" --pre-runs "$SRC" \
      --out "$OUT/figures_seed_groups"
  python3 code/recovery_time.py \
      --pre-runs "$SRC" --post-runs "$OUT/model_runs_reversal" \
      --out "$OUT/figures_seed_groups"
done

echo "DONE. Compare recovered_fraction / vigour-target-match TTR for the 0%->100% vs"
echo "100%->0% stimulus across results/reward_scale_intervention/{control,boost_up,damp_down}/"
echo "-- e.g. does boost_up show a shorter 0->100 TTR than control (from"
echo "time_to_recovery_vigour_target_per_stim, stim=0), and does damp_down show a LONGER"
echo "100->0 TTR (stim=2) than control? Grad/loss traces are in each condition's"
echo "model_runs_reversal/<model_type>/seed<N>/history.json (grad_norm_*, policy_loss,"
echo "value_loss, policy_grad_by_stim)."
