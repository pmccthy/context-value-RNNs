#!/usr/bin/env bash
# Dose-response follow-up to run_reward_scale_intervention.sh, now that damp_down (0.3x)
# showed a large, universal effect (every seed, every model class, slower to relearn the
# 100%->0% stimulus): does the slowdown scale smoothly with HOW MUCH the punishment is
# softened, or is there a sharper threshold?
#
# Scales swept (all applied to stim index 2 -- the 100%->0%, "newly worthless" stimulus
# -- stim 0/1 stay at 1.0, exactly as in damp_down): 0.1, 0.3, 0.5, 0.7, 1.0(=control).
# 1.0 and 0.3 REUSE the already-completed control/ and damp_down/ runs from
# run_reward_scale_intervention.sh (same SEEDS below -> same seeds/RNG -> directly
# paired) -- only 0.1/0.5/0.7 need fresh training.
#
# Usage:
#   cd results/transfer/reversal
#   screen -S dose_response
#   bash code/run_reward_scale_dose_response.sh
#   (Ctrl-a d to detach, screen -r dose_response to reattach)
set -euo pipefail
cd "$(dirname "$0")/.."                 # -> reversal_study/
ROOT="$(cd ../../.. && pwd)"            # repo root, for scripts/16_06_26_*.py (now 3 levels down: results/transfer/reversal/)

SRC=results/action_std_0p15_full/model_runs     # already-trained ORIGINAL models (unchanged)
OUT_ROOT=results/reward_scale_intervention
N_TRIALS_REV=5000
SEEDS=$(seq 42 49)                      # SAME 8 seeds as run_reward_scale_intervention.sh
TYPES="classif_rl rl_only classif_rl_readout_only"
SCALES="1.0 0.7 0.5 0.3 0.1"

for SCALE in $SCALES; do
  # macOS ships bash 3.2 (no associative arrays) -- portable case statement instead.
  case "$SCALE" in
    1.0) COND=control ;;                # reuses the existing control/ run untouched
    0.3) COND=damp_down ;;               # reuses the existing damp_down/ run untouched
    0.7) COND=damp_070 ;;
    0.5) COND=damp_050 ;;
    0.1) COND=damp_010 ;;
  esac
  OUT="$OUT_ROOT/$COND"
  echo "== scale=$SCALE (dir: $COND) =="
  for MT in $TYPES; do
    for S in $SEEDS; do
      SRC_DIR="$SRC/$MT/seed$S"
      DIR="$OUT/model_runs_reversal/$MT/seed$S"
      if [ -f "$DIR/model.pt" ]; then echo "  skip $MT seed$S ($COND, already done)"; continue; fi
      if [ ! -f "$SRC_DIR/model.pt" ]; then echo "  ! missing source model $SRC_DIR, skipping"; continue; fi
      echo "  -- $MT seed$S --"
      python3 "$ROOT/scripts/16_06_26_train_reversal.py" --run "$SRC_DIR" --out "$DIR" \
          --n-trials "$N_TRIALS_REV" --probe-every 20 --track-gradients \
          --reward-scale 1.0 1.0 "$SCALE"
    done
  done
  echo "  -- inference + onset probe ($COND) --"
  python3 "$ROOT/scripts/16_06_26_run_inference.py" --runs "$OUT/model_runs_reversal" \
      --out "$OUT/figure_data_reversal"
  python3 code/reversal_onset_probe.py --post-runs "$OUT/model_runs_reversal"
done

echo "== dose-response figure =="
python3 -c "
import sys; sys.path.insert(0, 'code')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import recovery_time as RT
import seed_groups as SG
F = SG.RA.F
plt.style.use(str(F.DEFAULT_STYLE))

SRC = '$SRC'
OUT_ROOT = '$OUT_ROOT'
out = f'{OUT_ROOT}/figures_comparison'
scale_runs = [
    (1.0, f'{OUT_ROOT}/control/model_runs_reversal'),
    (0.7, f'{OUT_ROOT}/damp_070/model_runs_reversal'),
    (0.5, f'{OUT_ROOT}/damp_050/model_runs_reversal'),
    (0.3, f'{OUT_ROOT}/damp_down/model_runs_reversal'),
    (0.1, f'{OUT_ROOT}/damp_010/model_runs_reversal'),
]
import os
os.makedirs(out, exist_ok=True)
RT.fig_reward_scale_dose_response(SRC, scale_runs, stim=2, out=out,
                                  xlabel='reward scale on 100%->0% stimulus')
print('wrote dose-response figure to', out)
"

echo "DONE. See results/reward_scale_intervention/figures_comparison/reward_scale_dose_response.png"
