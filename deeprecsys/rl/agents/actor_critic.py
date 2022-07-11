from deeprecsys.rl.agents.agent import ReinforcementLearning
from typing import Any, List, Optional
from deeprecsys.rl.experience_replay.experience_buffer import ExperienceReplayBuffer
from deeprecsys.rl.experience_replay.buffer_parameters import (
    ExperienceReplayBufferParameters,
)
from deeprecsys.rl.neural_networks.policy_estimator import PolicyEstimator
from deeprecsys.rl.neural_networks.value_estimator import ValueEstimator


class ActorCriticAgent(ReinforcementLearning):
    """Policy estimator using a value estimator as a baseline.
    It's on-policy, for discrete action spaces, and episodic environments.
    This implementation uses stochastic policies.
    TODO: could be a sub class of reinforce"""

    def __init__(
        self,
        n_actions: int,
        state_size: int,
        discount_factor: int = 0.99,
        actor_hidden_layers: Optional[List[int]] = None,
        critic_hidden_layers: Optional[List[int]] = None,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
    ):
        if not actor_hidden_layers:
            actor_hidden_layers = [state_size * 2, state_size * 2]
        if not critic_hidden_layers:
            critic_hidden_layers = [state_size * 2, int(state_size / 2)]
        self.episode_count = 0
        self.value_estimator = ValueEstimator(
            state_size,
            critic_hidden_layers,
            1,
            learning_rate=critic_learning_rate,
        )
        self.policy_estimator = PolicyEstimator(
            state_size,
            actor_hidden_layers,
            n_actions,
            learning_rate=actor_learning_rate,
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

    def learn_from_experiences(self):
        experiences = list(self.buffer.experience_queue)
        for timestep, experience in enumerate(experiences):
            total_return = 0
            for i, t in enumerate(experiences[timestep:]):
                total_return += (self.discount_factor ** i) * t.reward

            # Calculate baseline/advantage
            baseline_value = self.value_estimator.predict(experience.state).detach()
            advantage = total_return - baseline_value
            # Update our value estimator
            self.value_estimator.update(experience.state, total_return)
            # Update our policy estimator
            self.policy_estimator.update(experience.state, advantage, experience.action)
