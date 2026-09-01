#!/usr/bin/env python3
"""Extract real per-stimulus vigour (lick rate) and population-activity values
from the experimental repo, as the "expected shape" for the generalized
3-category chi-squared comparison in chi2_metrics.py (vigour and pop_activity
are continuous metrics, not counts -- see that module's docstring for exactly
how they're compared).

Sources (~/Documents/neuronal-representations/results/transfer):
  data/lick/lick_rates_per_trial_expert.csv
    columns: session, trial_number, stimulus_label (0/50/100), stimulus_type,
    reward_prob_rev, anticipatory_lick_rate (Hz, 2s post-stim window)
  data/population/population_means_expert.csv
    columns: stimulus (0/50/100), frame, time_s, pop_mean_pooled,
    sem_over_sessions, n_neurons_total, n_sessions

Usage: python3 extract_experiment_metric_values.py
Output: group_counts/vigour_expert.json, group_counts/pop_activity_expert.json
  each {"0": value, "50": value, "100": value}
"""
import json
import os
from pathlib import Path

import pandas as pd

EXPERIMENT_ROOT = Path(
    os.environ.get("NEURONAL_REPO", str(Path.home() / "Documents" / "neuronal-representations"))
) / "results" / "transfer"
LICK_CSV = EXPERIMENT_ROOT / "data" / "lick" / "lick_rates_per_trial_expert.csv"
POP_CSV = EXPERIMENT_ROOT / "data" / "population" / "population_means_expert.csv"
STIM_WINDOW = (0.0, 2.0)   # seconds post stimulus onset, matches the model's stim window


def main():
    out_dir = Path("group_counts")
    out_dir.mkdir(exist_ok=True)

    lick = pd.read_csv(LICK_CSV)
    vigour = lick.groupby("stimulus_label")["anticipatory_lick_rate"].mean()
    vigour = {str(int(k)): float(v) for k, v in vigour.items()}
    (out_dir / "vigour_expert.json").write_text(json.dumps(vigour, indent=2))
    print(f"source: {LICK_CSV}")
    print("real per-stimulus vigour (mean anticipatory lick rate, Hz):", vigour)

    pop = pd.read_csv(POP_CSV)
    win = pop[(pop["time_s"] >= STIM_WINDOW[0]) & (pop["time_s"] <= STIM_WINDOW[1])]
    pop_activity = win.groupby("stimulus")["pop_mean_pooled"].mean()
    pop_activity = {str(int(k)): float(v) for k, v in pop_activity.items()}
    (out_dir / "pop_activity_expert.json").write_text(json.dumps(pop_activity, indent=2))
    print(f"\nsource: {POP_CSV}  (stim window {STIM_WINDOW}s)")
    print("real per-stimulus population activity (mean dF/F):", pop_activity)


if __name__ == "__main__":
    main()
