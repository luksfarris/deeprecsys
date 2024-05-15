from typing import Any, List, Optional

from numpy import arange
from numpy.random import RandomState

from deeprecsys.neural_networks.deep_q_network import (
    DeepQNetwork,
    sequential_architecture,
)
from deeprecsys.rl.agents.epsilon_greedy import DecayingEpsilonGreedy
from deeprecsys.rl.experience_replay.buffer_parameters import (
    ExperienceReplayBufferParameters,
)
from deeprecsys.rl.experience_replay.experience_buffer import ExperienceReplayBuffer


class DQNAgent(DecayingEpsilonGreedy):
    """TODO: This agent needs to be fixed"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: List,
        network_update_frequency: int = 3,
        initial_exploration_probability: float = 1.0,
        decay_rate: float = 0.99,
        minimum_exploration_probability: float = 0.05,
        buffer_size: int = 10000,
        buffer_burn_in: int = 1000,
        batch_size: int = 32,
        discount_factor: float = 0.99,
        learning_rate: float = 0.99,
        random_state: Optional[RandomState] = None,
    ) -> None:
        """TODO"""
        if random_state is None:
            random_state = RandomState()
        super().__init__(
            initial_exploration_probability,
            decay_rate,
            minimum_exploration_probability,
            random_state,
        )

        architecture = sequential_architecture([input_size] + hidden_layers + [output_size])
        self.network = DeepQNetwork(learning_rate, architecture, discount_factor)
        self.buffer = ExperienceReplayBuffer(
            ExperienceReplayBufferParameters(
                max_experiences=buffer_size,
                minimum_experiences_to_start_predicting=buffer_burn_in,
                batch_size=batch_size,
                random_state=random_state,
            )
        )
        self.step_count = 0
        self.network_update_frequency = network_update_frequency
        self.actions = arange(output_size)

    def _check_update_network(self) -> None:
        if self.buffer.ready_to_predict():
            self.step_count += 1
            if self.step_count == self.network_update_frequency:
                self.step_count = 0
                batch = self.buffer.sample_batch()
                self.network.learn_from(batch)

    def action_for_state(self, state: Any) -> Any:
        """TODO"""
        state_flat = state.flatten()
        if self.buffer.ready_to_predict():
            action = super().action_for_state(state_flat)
        else:
            action = self.explore()
        self._check_update_network()
        return action

    def top_k_actions_for_state(self, state: Any, k: int = 1) -> Any:
        """TODO"""

    def explore(self) -> float:
        """TODO"""
        return self.random_state.choice(self.actions)

    def exploit(self, state: Any) -> float:
        """TODO"""
        return self.network.best_action_for_state(state)

    def store_experience(self, state: Any, action: Any, reward: float, done: bool, new_state: Any) -> None:
        """TODO"""
        if done and self.buffer.ready_to_predict():
            self._decay()
        state_flat = state.flatten()
        new_state_flat = new_state.flatten()
        self.buffer.store_experience(state_flat, action, reward, done, new_state_flat)
