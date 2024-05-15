from abc import ABC, abstractmethod
from typing import Any

from numpy.random import RandomState

from deeprecsys.rl.agents.agent import ReinforcementLearning


class DecayingEpsilonGreedy(ReinforcementLearning, ABC):
    """A reinforcement learning model that reduces exploration over time."""

    def __init__(
        self,
        initial_exploration_probability: float = 0.2,
        decay_rate: float = 1,
        minimum_exploration_probability: float = 0.01,
        random_state: RandomState = None,
    ):
        """To set up a classic e-greedy model, the decay rate needs to be 1."""
        if random_state is None:
            random_state = RandomState()
        self.random_state = random_state
        self.epsilon = initial_exploration_probability
        self.minimum_exploration_probability = minimum_exploration_probability
        self.decay_rate = decay_rate

    def action_for_state(self, state: Any) -> Any:
        """With probability epsilon, we explore by sampling one of the random available actions.
        Otherwise we exploit by chosing the action with the highest Q value.
        """
        if self.random_state.random() < self.epsilon:
            action = self.explore()
        else:
            action = self.exploit(state)
        return action

    def _decay(self) -> None:
        """Slowly decrease the exploration probability."""
        self.epsilon = max(self.epsilon * self.decay_rate, self.minimum_exploration_probability)

    @abstractmethod
    def explore(self) -> Any:
        """Randomly selects an action"""
        pass

    @abstractmethod
    def exploit(self, state: Any) -> Any:
        """Select the best action known for the given state"""
        pass
