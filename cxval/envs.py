import numpy as np


class TaskEnv:
    """Lick / no-lick environment wrapping a pre-generated state sequence.

    The pre-generated states array encodes context, stimulus, and a reward-
    window cue (last column).  This environment appends two feedback columns
    — rewarded and unrewarded — that are filled in at runtime based on the
    agent's lick decision.  The model therefore sees observations of shape
    (n_contexts + n_stimuli + 3,).

    Actions
    -------
    0 : lick
    1 : no-lick

    Reward structure
    ----------------
    Lick during reward window, reward available     → reward_lick      (default +1)
    Lick during reward window, reward not available → reward_lick_miss  (default -1)
    Any other timestep / no-lick                    → reward_no_lick    (default  0)

    Only the first lick in each reward window is counted; subsequent licks
    within the same window are treated as no-lick.

    Parameters
    ----------
    states : ndarray, shape (T, D)
        Pre-generated state array from StateSequence.generate().
        Last column must be the reward-window cue (1 during response windows).
    reward_availability : ndarray, shape (T,)
        Pre-generated availability array from StateSequence.generate().
        1 during response-window timesteps of rewarded trials, 0 otherwise.
    reward_lick : float
    reward_no_lick : float
    reward_lick_miss : float
    """

    LICK = 0
    NO_LICK = 1

    def __init__(
        self,
        states,
        reward_availability,
        reward_lick=1.0,
        reward_no_lick=0.0,
        reward_lick_miss=-1.0,
    ):
        self._states = np.asarray(states, dtype=np.float32)
        self._reward_availability = np.asarray(reward_availability, dtype=np.float32)
        self.T = len(states)
        self.obs_dim = states.shape[1] + 2  # base + [rewarded, unrewarded]
        self.reward_lick = reward_lick
        self.reward_no_lick = reward_no_lick
        self.reward_lick_miss = reward_lick_miss

        self._t = 0
        self._licked_this_window = False
        self._lick_rewarded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Reset to the start of the sequence.

        Returns
        -------
        obs : ndarray, shape (obs_dim,)
        info : dict
        """
        self._t = 0
        self._licked_this_window = False
        self._lick_rewarded = False
        return self._obs(), {}

    def step(self, action):
        """Advance one timestep.

        Parameters
        ----------
        action : int
            0 for lick, 1 for no-lick.

        Returns
        -------
        obs : ndarray, shape (obs_dim,)
        reward : float
        terminated : bool
        truncated : bool
        info : dict
        """
        t = self._t
        in_window = bool(self._states[t, -1])

        # Clear lick tracking at the start of any non-window timestep so
        # that each reward window gets exactly one lick opportunity.
        if not in_window:
            self._licked_this_window = False
            self._lick_rewarded = False

        licked = (action == self.LICK)
        reward_available = bool(self._reward_availability[t])

        if in_window and licked and not self._licked_this_window:
            self._licked_this_window = True
            self._lick_rewarded = reward_available
            reward = self.reward_lick if reward_available else self.reward_lick_miss
        else:
            reward = self.reward_no_lick

        self._t += 1
        terminated = self._t >= self.T
        obs = np.zeros(self.obs_dim, dtype=np.float32) if terminated else self._obs()

        info = {
            "timestep": t,
            "in_reward_window": in_window,
            "licked": licked,
            "reward_available": reward_available,
        }
        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs(self):
        base = self._states[self._t].copy()
        feedback = np.zeros(2, dtype=np.float32)
        if base[-1] > 0 and self._licked_this_window:
            base[-1] = 0.0  # replace window cue with outcome
            feedback[0 if self._lick_rewarded else 1] = 1.0
        return np.concatenate([base, feedback])
