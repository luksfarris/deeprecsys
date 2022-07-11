from copy import deepcopy
import torch
from torch import FloatTensor, BoolTensor
from typing import Any
from gym.spaces import Space
from deeprecsys.rl.neural_networks.gaussian_actor import GaussianActor
from deeprecsys.rl.neural_networks.q_value_estimator import TwinnedQValueEstimator
from deeprecsys.rl.agents.agent import ReinforcementLearning
from deeprecsys.rl.experience_replay.priority_replay_buffer import (
    PrioritizedExperienceReplayBuffer,
)
from deeprecsys.rl.experience_replay.buffer_parameters import (
    ExperienceReplayBufferParameters,
    PERBufferParameters,
)


class SoftActorCritic(ReinforcementLearning):
    """TODO: there's things to fix in this agent. It needs temperature
    optimization, and replace the current q-value estimator with the
    Q-value + value + value_target estimators, like described here
    https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html"""

    def __init__(
        self,
        action_space: Space,
        state_size: int,
        timesteps_to_start_predicting: int = 256,
        learning_rate: float = 0.0001,
        soft_target_update_rate: float = 0.005,
        entropy_coefficient: float = 0.2,
        target_update_interval: int = 2,
        discount_factor: float = 0.99,
        buffer_parameters=ExperienceReplayBufferParameters(),
        per_parameters=PERBufferParameters(),
    ):

        self.action_space = action_space
        n_actions = 1  # TODO: slate size
        self.actor = GaussianActor(
            inputs=state_size,
            outputs=n_actions,
            learning_rate=learning_rate,
            entropy_coefficient=entropy_coefficient,
            discount_factor=discount_factor,
        )
        self.critic = TwinnedQValueEstimator(
            inputs=state_size + 1, learning_rate=learning_rate
        )
        self.target_critic = deepcopy(self.critic)
        self.buffer = PrioritizedExperienceReplayBuffer(
            buffer_parameters=ExperienceReplayBufferParameters(),
            per_parameters=PERBufferParameters(),
        )

        # disable gradient calculations of the target network
        self.target_critic.disable_learning()

        self.timesteps_to_start_predicting = timesteps_to_start_predicting
        self.timesteps = 0
        self.learning_steps = 0  # times the network was trained
        self.tau = soft_target_update_rate
        self.target_update_interval = target_update_interval
        self.gamma = discount_factor

    def should_update_network(self):
        return (
            self.timesteps >= self.timesteps_to_start_predicting
            and self.buffer.ready_to_predict()  # noqa
        )

    def action_for_state(self, state: Any) -> Any:
        if self.timesteps < self.timesteps_to_start_predicting:
            action = self.action_space.sample()
        else:
            action = self.explore(state)
        return int(action)

    def top_k_actions_for_state(self, state, k):
        # TODO:
        pass

    def store_experience(
        self, state: Any, action: Any, reward: float, done: bool, new_state: Any
    ):
        self.timesteps += 1
        state_flat = state.flatten()
        new_state_flat = new_state.flatten()
        self.buffer.store_experience(state_flat, action, reward, done, new_state_flat)
        if self.should_update_network():
            self.learn()

    def explore(self, state: Any) -> Any:
        # act with gaussian randomness
        with torch.no_grad():
            action, _, _ = self.actor.predict(state.reshape(1, -1))
        action_array = action.cpu().numpy().reshape(-1)
        n_actions = self.action_space.n
        return action_array[0].clip(0, n_actions - 1).round()

    def exploit(self, state: Any) -> Any:
        # act without randomness
        with torch.no_grad():
            _, _, action = self.actor.predict(state.reshape(1, -1))
        action_array = action.cpu().numpy().reshape(-1)
        n_actions = self.action_space.n
        return action_array[0].clip(0, n_actions - 1).round()

    def learn(self):
        self.learning_steps += 1

        if self.learning_steps % self.target_update_interval == 0:
            # instead of updating the target network "the hard way", we use a Tau
            # parameter as a weighting factor to update the weights as an
            # exponential moving average. This is analogous to the target net update
            # in the DQN algorithm.
            self.target_critic.soft_parameter_update(self.critic, update_rate=self.tau)

        # batch with indices and priority weights
        batch = self.buffer.sample_batch()
        states, actions, rewards, dones, next_states, weights, samples = [
            i for i in batch
        ]
        # convert to tensors
        device = self.critic.device
        state_tensors = FloatTensor(states).to(device=device)
        next_state_tensors = FloatTensor(next_states).to(device=device)
        reward_tensors = FloatTensor(rewards).to(device=device).reshape(-1, 1)
        action_tensors = FloatTensor(actions).reshape(-1, 1).to(device=device)
        done_tensors = BoolTensor(dones).to(device=device)
        weight_tensors = FloatTensor(weights).to(device=device)

        errors = self.critic.calculate_loss(
            state_tensors,
            action_tensors,
            reward_tensors,
            done_tensors,
            next_state_tensors,
            weight_tensors,
            actor=self.actor,
            target=self.target_critic,
        )
        self.actor.calculate_loss(
            state_tensors,
            action_tensors,
            reward_tensors,
            done_tensors,
            next_state_tensors,
            weight_tensors,
            critic=self.critic,
        )
        # update priority weights
        self.buffer.update_priorities(batch, errors.cpu().numpy())
