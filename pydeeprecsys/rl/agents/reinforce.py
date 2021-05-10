from pydeeprecsys.rl.agents.agent import ReinforcementLearning
from typing import Any
from pydeeprecsys.rl.experience_replay.experience_buffer import ExperienceReplayBuffer
from pydeeprecsys.rl.neural_networks.policy_estimator import PolicyEstimator
from pydeeprecsys.rl.neural_networks.value_estimator import ValueEstimator


class ReinforceAgent(ReinforcementLearning):
    """REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient."""

    def __init__(
        self,
        n_actions: int,
        state_size: int,
        discount_factor: int = 0.99,
        learning_rate=1e-3,
    ):
        self.episode_count = 0
        self.value_estimator = ValueEstimator(
            state_size,
            [state_size * 2, int(state_size / 2)],
            1,
            learning_rate=learning_rate,
        )
        self.policy_estimator = PolicyEstimator(
            state_size,
            [state_size * 2, state_size * 2],
            n_actions,
            learning_rate=learning_rate,
        )
        self.discount_factor = discount_factor
        # starts the buffer
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = ExperienceReplayBuffer(10000, 1, 1)

    def action_for_state(self, state: Any) -> Any:
        return self.policy_estimator.predict(state)

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
            baseline_value = self.value_estimator.predict(experience.state)
            advantage = total_return - baseline_value
            # Update our value estimator
            self.value_estimator.update(experience.state, total_return)
            # Update our policy estimator
            self.policy_estimator.update(experience.state, advantage, experience.action)
