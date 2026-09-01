#!/usr/bin/env python3
"""Generalized chi-squared shape-fit for CONTINUOUS per-stimulus metrics
(vigour, population activity), extending population_similarity.chi2_fit_to_data
(which only handles the categorical 7-group responder composition) to the
3-stimulus case.

Framing (confirmed with the user): chi-squared goodness-of-fit is built for
categorical counts. To use it on a continuous metric, the 3 stimuli are
treated as categories and each metric's per-stimulus VALUE as a pseudo-
frequency -- i.e. exactly the same "rescale real proportions to the model's
own total, then scipy.stats.chisquare(f_obs=model, f_exp=rescaled-real)" trick
the existing group-level test already uses, just on 3 categories instead of 7.

Caveat (same one that applies to the group version): the absolute chi2 value
here is scale-dependent (it grows with whatever "total" the model's per-stim
values are rescaled to) -- what's meaningful is the RELATIVE ranking across
model types at a FIXED total, which is what this function reports.
"""
import json
from pathlib import Path

import numpy as np

STIM_ORDER = ["0", "50", "100"]


def chi2_shape_fit(obs_values, real_values_dict, total=1000.0):
    """obs_values: array-like of length 3 (model's own 0/50/100 values, any
    consistent unit). real_values_dict: {"0":.., "50":.., "100":..} real data.
    Returns (chi2, obs_rescaled, exp) -- p-value needs scipy.stats.chisquare,
    not computed here (see the note at the bottom of this file for the
    one-line call once scipy is available)."""
    obs = np.asarray([obs_values[i] for i in range(3)], dtype=float)
    real = np.asarray([real_values_dict[s] for s in STIM_ORDER], dtype=float)
    real_p = real / real.sum()
    obs_rescaled = obs / obs.sum() * total
    exp = real_p * total
    chi2 = float(np.sum((obs_rescaled - exp) ** 2 / exp))
    return chi2, obs_rescaled, exp


# once scipy is available (your cxval/neuronal_representations env):
#   from scipy.stats import chisquare
#   chi2, p = chisquare(f_obs=obs_rescaled, f_exp=exp)   # dof = 2
