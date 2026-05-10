import torch
from torch.distributions import Categorical

from cxval.models import ActorCritic


class Agent:
    """Stateful agent that wraps an :class:`ActorCritic` for episode interaction.

    Maintains the RNN hidden state across timesteps within an episode and
    resets it between episodes.  Action selection is done by sampling from the
    policy distribution; the agent runs in inference mode (no gradients).

    Args:
        actor_critic: A constructed ActorCritic instance.
        device: Device to run inference on.  Defaults to CPU.
    """

    def __init__(self, actor_critic: ActorCritic, device=None):
        self.actor_critic = actor_critic
        self.device = device or torch.device("cpu")
        self.hidden: torch.Tensor | None = None

    def reset(self):
        """Reset the hidden state at the start of a new episode."""
        self.hidden = None

    @torch.no_grad()
    def act(self, obs):
        """Select an action given a single observation.

        Advances the hidden state by one step.  Call :meth:`reset` at the
        start of each episode.

        Args:
            obs: Observation as a numpy array or tensor of shape (input_size,).

        Returns:
            action: Sampled action index (int).
            log_prob: Log probability of the sampled action (scalar tensor).
            value: Value estimate (scalar tensor).
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs = obs.unsqueeze(0)  # (1, input_size)

        logits, value, self.hidden = self.actor_critic.step(obs, self.hidden)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value
