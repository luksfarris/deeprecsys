from abc import ABC, abstractmethod
from typing import Any

from gymnasium import Space


class ReinforcementLearning(ABC):
    """Abstract class that encapsulates the behavior of RL models."""

    @abstractmethod
    def action_for_state(self, state: Any) -> Any:
        """Given a state, return the next predicted action."""

    @abstractmethod
    def top_k_actions_for_state(self, state: Any, k: int = 1) -> Any:
        """Retrieve the next K best actions for this state."""

    @abstractmethod
    def store_experience(self, state: Any, action: Any, reward: float, done: bool, new_state: Any) -> None:
        """Store an experience (used in case of experience replay buffers)"""


class RandomAgent(ReinforcementLearning):
    """An agent that randomly samples actions, regardless of the
    environment's state.
    """

    action_space: Space

    def __init__(self, action_space: Space, random_state: Any = 42):
        """Start the agent with the provided action space and seed."""
        self.action_space = action_space
        # we seed the state so actions are reproducible
        self.action_space.seed(random_state)

    def action_for_state(self, state: Any) -> Any:
        """Sample a random action from the action space."""
        return self.action_space.sample()

    def top_k_actions_for_state(self, state: Any, k: int = 1) -> Any:
        """Randomly sample K actions from the action space."""
        return self.action_space.sample()

    def store_experience(self, state: Any, action: Any, reward: float, done: bool, new_state: Any) -> None:
        """Ignore the experience because this agent doesn't have any experience replay."""
        pass
