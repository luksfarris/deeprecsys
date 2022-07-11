from abc import ABC, abstractmethod
from gym import Space
from typing import Any


class ReinforcementLearning(ABC):
    @abstractmethod
    def action_for_state(self, state: Any) -> Any:
        pass

    @abstractmethod
    def top_k_actions_for_state(self, state: Any, k: int = 1) -> Any:
        pass

    @abstractmethod
    def store_experience(
        self, state: Any, action: Any, reward: float, done: bool, new_state: Any
    ):
        pass


class RandomAgent(ReinforcementLearning):
    """An agent that randomly samples actions, regardless of the
    environment's state."""

    action_space: Space

    def __init__(self, action_space: Space, random_state=42):
        self.action_space = action_space
        # we seed the state so actions are reproducible
        self.action_space.seed(random_state)

    def action_for_state(self, state: Any) -> Any:
        return self.action_space.sample()

    def top_k_actions_for_state(self, state: Any, k: int = 1) -> Any:
        return self.action_space.sample()

    def store_experience(
        self, state: Any, action: Any, reward: float, done: bool, new_state: Any
    ):
        pass
