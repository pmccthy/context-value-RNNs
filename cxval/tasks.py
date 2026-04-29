"""
Code for generating task data.
Author: patrick.mccarthy@dpag.ox.ac.uk
"""
import numpy as np

class ValueMatrix:
    """
    A class for generating a matrix of values and contexts.
    """

    def __init__(self,
                 seed,
                 contexts,
                 stimuli,
                 delta_context,
                 base_lower=0.0,
                 base_upper=1.0,
                 base_values=None,  # if set, overrides base_lower/base_upper in generate_base_values
                 ):
        self.contexts = contexts
        self.stimuli = stimuli
        self.delta_context = delta_context
        self.base_lower = base_lower
        self.base_upper = base_upper
        self.base_values = base_values

    def generate_base_values(self, seed):
        """
        Sample one base value per stimulus from U(base_lower, base_upper) and store on self.base_values.
        Call this once per desired base-value instantiation, then call generate_value_matrix()
        as many times as needed to layer different noise realisations on top.
        """
        rng = np.random.default_rng(seed)
        self.base_values = rng.uniform(self.base_lower, self.base_upper, size=len(self.stimuli))
        return self.base_values

    def generate_value_matrix(self, seed):
        """
        Apply context-specific noise on top of self.base_values.
        Call generate_base_values() (or pass base_values to __init__) before calling this.
        """
        if self.base_values is None:
            raise ValueError("base_values not set — call generate_base_values() first or pass base_values to __init__.")

        rng = np.random.default_rng(seed)
        base = np.asarray(self.base_values)[:, np.newaxis]
        context_noise = rng.uniform(-self.delta_context, self.delta_context, size=(len(self.stimuli), len(self.contexts)))

        return np.clip(base + context_noise, 0, 1)


class StimulusSequence:
    """
    A class for generating a sequence of stimuli from a value matrix.
    """

    def __init__(self,
                 value_matrix,
                 trials_per_phase,
                 phases_per_context,
                 context_order='random',  # 'random', 'sequential', or list of context indices
                 ):
        self.value_matrix = value_matrix  # (n_stim, n_ctx) array
        self.n_stimuli, self.n_contexts = value_matrix.shape
        self.trials_per_phase = trials_per_phase
        self.phases_per_context = phases_per_context
        self.context_order = context_order

    def generate(self, seed):
        """
        Generate trial-level context and stimulus sequences.
        Returns (trial_contexts, trial_stimuli), each of shape (n_trials,).
        Within each phase, stimuli are shuffled so each appears equally often.
        """
        rng = np.random.default_rng(seed)

        if self.context_order == 'random':
            ctx_sequence = rng.permutation(self.n_contexts)
        elif self.context_order == 'sequential':
            ctx_sequence = np.arange(self.n_contexts)
        else:
            ctx_sequence = np.asarray(self.context_order)

        trial_contexts, trial_stimuli = [], []
        for ctx in ctx_sequence:
            for _ in range(self.phases_per_context):
                repeats = int(np.ceil(self.trials_per_phase / self.n_stimuli))
                stim_pool = np.tile(np.arange(self.n_stimuli), repeats)[:self.trials_per_phase]
                rng.shuffle(stim_pool)
                trial_contexts.append(np.full(self.trials_per_phase, ctx))
                trial_stimuli.append(stim_pool)

        self.trial_contexts = np.concatenate(trial_contexts)
        self.trial_stimuli = np.concatenate(trial_stimuli)
        return self.trial_contexts, self.trial_stimuli


class StateSequence:
    """
    A class for generating a sequence of states from a stimulus-reward sequence.

    Each trial is structured as: [ITI | stimulus | reward].
    The state vector at each timestep is [context_onehot | stimulus_onehot | reward],
    where context is on throughout the trial, stimulus only during the stim epoch,
    and reward only during the reward epoch.
    """

    def __init__(self,
                 stimulus_sequence,
                 value_matrix,
                 stim_timesteps,
                 reward_timesteps,
                 iti_timesteps,  # int for fixed duration, or (min, max) tuple to sample uniformly
                 ):
        self.stimulus_sequence = stimulus_sequence
        self.value_matrix = value_matrix  # (n_stim, n_ctx) array of reward probabilities
        self.stim_timesteps = stim_timesteps
        self.reward_timesteps = reward_timesteps
        self.iti_timesteps = iti_timesteps

    def generate_rewards(self):
        """
        Generate binary reward outcomes by sampling from per-stimulus reward probabilities.
        Requires generate() to have been called first to populate trial_contexts/trial_stimuli.
        """
        probs = self.value_matrix[self.trial_stimuli, self.trial_contexts]
        probs = np.clip(probs, 0, 1)
        self.rewards = self._rng.binomial(1, probs).astype(float)
        return self.rewards

    def generate(self, seed):
        """
        Build the full time-series state array of shape (total_timesteps, n_contexts + n_stimuli + 1).
        Returns (states, rewards).
        """
        self._rng = np.random.default_rng(seed)

        self.trial_contexts = self.stimulus_sequence.trial_contexts
        self.trial_stimuli = self.stimulus_sequence.trial_stimuli
        n_trials = len(self.trial_contexts)
        n_stimuli, n_contexts = self.value_matrix.shape
        input_dim = n_contexts + n_stimuli + 1

        # generate reward sequence
        self.generate_rewards()

        # generate ITI durations
        if isinstance(self.iti_timesteps, (tuple, list)):
            self.iti_durations = self._rng.integers(self.iti_timesteps[0], self.iti_timesteps[1] + 1, size=n_trials)
        else:
            self.iti_durations = np.full(n_trials, self.iti_timesteps, dtype=int)

        total_timesteps = int((self.iti_durations + self.stim_timesteps + self.reward_timesteps).sum())
        states = np.zeros((total_timesteps, input_dim))

        t = 0
        for i in range(n_trials):
            ctx, stim, iti_len = self.trial_contexts[i], self.trial_stimuli[i], self.iti_durations[i]
            trial_end = t + iti_len + self.stim_timesteps + self.reward_timesteps

            # context one-hot is on for the entire trial
            states[t:trial_end, ctx] = 1.0

            # stimulus one-hot during stim epoch
            stim_start = t + iti_len
            states[stim_start:stim_start + self.stim_timesteps, n_contexts + stim] = 1.0

            # reward scalar during reward epoch
            rew_start = stim_start + self.stim_timesteps
            states[rew_start:rew_start + self.reward_timesteps, -1] = self.rewards[i]

            t = trial_end

        self.states = states
        return states, self.rewards
