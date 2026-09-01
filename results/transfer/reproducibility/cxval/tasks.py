"""
Code for generating task data.
Author: patrick.mccarthy@dpag.ox.ac.uk
"""
import numpy as np


def build_swap_value_matrix(n_swap_low=2, n_swap_high=2, p_low=0.0, p_high=1.0):
    """Build a (n_stimuli, 2) value matrix for a context-swap value task.

    ``n_swap_low`` stimuli are rewarded p_low in context 0 / p_high in
    context 1; ``n_swap_high`` stimuli are rewarded p_high in context 0 /
    p_low in context 1. Reward contingencies fully swap between the two
    contexts. The default (2, 2, p_low=0.0, p_high=1.0) gives the original
    DETERMINISTIC 4-stimulus / 2-context task: two 0%<->100% stimuli and two
    100%<->0% stimuli.

    p_low/p_high: pass e.g. p_low=0.2, p_high=0.8 for a PROBABILISTIC swap
    task instead -- a single trial's outcome is then no longer fully
    diagnostic of the current context (unlike the deterministic 0%/100%
    default, where one surprising outcome perfectly reveals a switch), so
    genuine multi-trial evidence accumulation becomes necessary to infer
    context reliably, rather than the single-trial "probe this reward
    window, then commit" strategy that the deterministic task allows. Every
    downstream consumer (cxval.geometry.build_conditions, reward sampling in
    cxval.tasks.StateSequence.generate_rewards, etc.) already treats
    value_matrix entries as literal Bernoulli reward PROBABILITIES, not
    assumed-binary values -- the only fix needed for non-{0,1} entries was
    build_conditions rounding (not truncating) to a binary "value" label for
    decoding/CCGP purposes, since e.g. int(0.8) would wrongly truncate to 0.

    Returns:
        value_matrix: (n_swap_low + n_swap_high, 2) float32 array.
    """
    rows = [[p_low, p_high]] * n_swap_low + [[p_high, p_low]] * n_swap_high
    return np.array(rows, dtype=np.float32)

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
                 include_context=True,
                 ):
        """
        include_context: if False, the context one-hot block is OMITTED from
            the state vector entirely (input_dim = n_stimuli + 1 instead of
            n_contexts + n_stimuli + 1) -- for the uncued/meta-learning
            variant of this task, where the agent must infer the latent
            context from its own action/reward history instead of being told
            it directly. `trial_structure`'s "context" field still records
            the TRUE context on every trial regardless of this flag, purely
            for downstream analysis/decoding -- only the observation the
            model actually sees is affected. See
            cxval.vigour.BatchedVigourEnv's feedback_action_reward for the
            other half of this (splicing the agent's own previous action and
            reward into its next observation, which becomes the only
            available context signal when include_context=False).
        """
        self.stimulus_sequence = stimulus_sequence
        self.value_matrix = value_matrix  # (n_stim, n_ctx) array of reward probabilities
        self.stim_timesteps = stim_timesteps
        self.reward_timesteps = reward_timesteps
        self.iti_timesteps = iti_timesteps
        self.include_context = include_context

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
            states: (total_timesteps, [n_contexts if include_context else 0] +
                n_stimuli + 1) float32 array.
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
        ctx_dim = n_contexts if self.include_context else 0
        input_dim = ctx_dim + n_stimuli + 1

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

            # context one-hot is on for the entire trial (omitted entirely if
            # include_context=False -- see __init__ docstring)
            if self.include_context:
                states[t:trial_end, ctx] = 1.0

            # stimulus one-hot during stim epoch
            states[stim_start:stim_start + self.stim_timesteps, ctx_dim + stim] = 1.0

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


class InterleavedStimulusSequence:
    """Trial-level interleaved multi-context stimulus sequence.

    Unlike ``StimulusSequence`` with ``context_order='random'`` (which
    shuffles the ORDER of whole context *phases*, each internally a run of
    ``trials_per_phase`` same-context trials), this shuffles individual
    TRIALS: each context is assigned ``trials_per_context`` trials total
    (stimuli balanced within each context, same as StimulusSequence), the two
    context-labelled pools are concatenated and then randomly permuted at the
    single-trial level. This is the "interleaved" counterpart to a
    block-structured run with the same total per-context trial count, so the
    two training regimes are directly comparable.

    Exposes ``trial_contexts`` / ``trial_stimuli`` and a ``generate(seed)``
    method with the same contract as ``StimulusSequence``, so it is a drop-in
    for ``StateSequence(stimulus_sequence=...)``.

    Args:
        value_matrix: (n_stim, n_ctx) array (only used for its shape here;
            reward sampling itself happens downstream in StateSequence).
        trials_per_context: Total trials for each context. Either a single
            int (same for every context) or a list/array of length n_ctx.
    """

    def __init__(self, value_matrix, trials_per_context):
        value_matrix = np.asarray(value_matrix)
        self.value_matrix = value_matrix
        self.n_stimuli, self.n_contexts = value_matrix.shape
        self.trials_per_context = trials_per_context

    def generate(self, seed):
        """Generate trial-level interleaved context and stimulus sequences.

        Returns (trial_contexts, trial_stimuli), each of shape (n_trials,),
        with n_trials = sum(trials_per_context). Within each context's
        allocation, stimuli are balanced (each appears equally often, as in
        StimulusSequence); the concatenated pool is then shuffled at the
        single-trial level so context labels are fully interleaved.
        """
        rng = np.random.default_rng(seed)

        if isinstance(self.trials_per_context, (list, tuple, np.ndarray)):
            tpc = list(self.trials_per_context)
        else:
            tpc = [self.trials_per_context] * self.n_contexts

        ctx_list, stim_list = [], []
        for ci, n in enumerate(tpc):
            n = int(n)
            repeats = int(np.ceil(n / self.n_stimuli))
            stim_pool = np.tile(np.arange(self.n_stimuli), repeats)[:n]
            rng.shuffle(stim_pool)
            ctx_list.append(np.full(n, ci))
            stim_list.append(stim_pool)

        ctx_arr = np.concatenate(ctx_list)
        stim_arr = np.concatenate(stim_list)
        order = rng.permutation(len(ctx_arr))

        self.trial_contexts = ctx_arr[order]
        self.trial_stimuli = stim_arr[order]
        return self.trial_contexts, self.trial_stimuli
