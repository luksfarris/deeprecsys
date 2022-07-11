import numpy as np
from deeprecsys.rl.agents.agent import ReinforcementLearning
from typing import Any, List, Optional
from deeprecsys.rl.experience_replay.experience_buffer import ExperienceReplayBuffer
from deeprecsys.rl.experience_replay.buffer_parameters import (
    ExperienceReplayBufferParameters,
)
from deeprecsys.rl.neural_networks.policy_estimator import PolicyEstimator
from torch import FloatTensor


class ReinforceAgent(ReinforcementLearning):
    """Policy estimator using a value estimator as a baseline.
    It's on-policy, for discrete action spaces, and episodic environments."""

    def __init__(
        self,
        n_actions: int,
        state_size: int,
        hidden_layers: Optional[List[int]] = None,
        discount_factor: int = 0.99,  # a.k.a gamma
        learning_rate=1e-3,
    ):
        self.episode_count = 0
        if not hidden_layers:
            hidden_layers = [state_size * 2, state_size * 2]
        self.policy_estimator = PolicyEstimator(
            state_size,
            hidden_layers,
            n_actions,
            learning_rate=learning_rate,
        )
        self.discount_factor = discount_factor
        # starts the buffer
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = ExperienceReplayBuffer(
            ExperienceReplayBufferParameters(10000, 1, 1)
        )

    def top_k_actions_for_state(self, state: Any, k: int = 1) -> List[int]:
        return self.policy_estimator.predict(state, k=k)

    def action_for_state(self, state: Any) -> int:
        return self.top_k_actions_for_state(state)[0]

    def store_experience(
        self, state: Any, action: Any, reward: float, done: bool, new_state: Any
    ):
        state_flat = state.flatten()
        new_state_flat = new_state.flatten()
        self.buffer.store_experience(state_flat, action, reward, done, new_state_flat)
        # FIXME: should learn after every episode, or after every N experiences?
        if done:  # and self.buffer.ready_to_predict():
            self.learn_from_experiences()
            self.reset_buffer()

    def discounted_rewards(self, rewards: np.array) -> np.array:
        """From a list of rewards obtained in an episode, we calculate
        the return minus the baseline. The baseline is the list of discounted
        rewards minus the mean, divided by the standard deviation."""
        discount_r = np.zeros_like(rewards)
        timesteps = range(len(rewards))
        reward_sum = 0
        for i in reversed(timesteps):
            reward_sum = rewards[i] + self.discount_factor * reward_sum
            discount_r[i] = reward_sum
        return_mean = discount_r.mean()
        return_std = discount_r.std()
        baseline = (discount_r - return_mean) / return_std
        return baseline

    def learn_from_experiences(self):
        experiences = list(self.buffer.experience_queue)
        states, actions, rewards, dones, next_states = zip(*experiences)
        advantages = self.discounted_rewards(rewards)
        advantages_tensor = FloatTensor(advantages).to(
            device=self.policy_estimator.device
        )
        self.policy_estimator.update(states, advantages_tensor, actions)
