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
                 context_reps=1,          # how many times the context_order sequence is repeated
                 ):
        self.value_matrix = value_matrix  # (n_stim, n_ctx) array
        self.n_stimuli, self.n_contexts = value_matrix.shape
        self.trials_per_phase = trials_per_phase
        self.phases_per_context = phases_per_context
        self.context_order = context_order
        self.context_reps = context_reps

    def generate(self, seed):
        """
        Generate trial-level context and stimulus sequences.
        Returns (trial_contexts, trial_stimuli), each of shape (n_trials,).
        Within each phase, stimuli are shuffled so each appears equally often.
        """
        rng = np.random.default_rng(seed)

        if self.context_order == 'random':
            ctx_sequence = np.concatenate([rng.permutation(self.n_contexts)
                                           for _ in range(self.context_reps)])
        elif self.context_order == 'sequential':
            ctx_sequence = np.tile(np.arange(self.n_contexts), self.context_reps)
        else:
            ctx_sequence = np.tile(np.asarray(self.context_order), self.context_reps)

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

    Each trial is structured as: [ITI | stimulus | reward_window].
    The state vector at each timestep is [context_onehot | stimulus_onehot | reward_window_cue],
    where context is on throughout the trial, stimulus only during the stim epoch,
    and reward_window_cue is 1 during the response window regardless of whether
    reward is actually available (so the agent sees the response opportunity but
    not the outcome).

    Reward availability is returned as a separate (T,) array that is 1 during
    reward-window timesteps for rewarded trials and 0 otherwise.  This is the
    signal used to compute the RL reward after the agent's lick decision.
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
        Build the full time-series arrays for the task sequence.

        Returns:
            states: (total_timesteps, n_contexts + n_stimuli + 1) float32 array.
                Last column is a reward-window cue (1 during all response windows).
            rewards: (n_trials,) binary array of per-trial reward outcomes.
            reward_availability: (total_timesteps,) binary array; 1 during response-
                window timesteps for rewarded trials, 0 otherwise.  Use this to
                compute the RL reward signal given the agent's lick decision.

        Also stores trial_structure on self: list of dicts (one per trial) with
        keys context, stimulus, reward_available, trial_start, trial_end, and
        (start, end) half-open index pairs iti_window, stim_window, reward_window.
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
        reward_availability = np.zeros(total_timesteps)
        trial_structure = []

        t = 0
        for i in range(n_trials):
            ctx, stim, iti_len = self.trial_contexts[i], self.trial_stimuli[i], self.iti_durations[i]
            stim_start = t + iti_len
            rew_start = stim_start + self.stim_timesteps
            rew_end = rew_start + self.reward_timesteps
            trial_end = rew_end

            # context one-hot is on for the entire trial
            states[t:trial_end, ctx] = 1.0

            # stimulus one-hot during stim epoch
            states[stim_start:stim_start + self.stim_timesteps, n_contexts + stim] = 1.0

            # reward window cue during response epoch (always 1, regardless of reward)
            states[rew_start:rew_end, -1] = 1.0

            # reward availability: 1 only if this trial is rewarded
            if self.rewards[i] > 0:
                reward_availability[rew_start:rew_end] = 1.0

            trial_structure.append({
                "trial_idx": i,
                "context": int(ctx),
                "stimulus": int(stim),
                "reward_available": bool(self.rewards[i]),
                "trial_start": t,
                "trial_end": trial_end,
                "iti_window": (t, stim_start),
                "stim_window": (stim_start, rew_start),
                "reward_window": (rew_start, rew_end),
            })

            t = trial_end

        self.states = states
        self.reward_availability = reward_availability
        self.trial_structure = trial_structure
        return states, self.rewards, reward_availability
