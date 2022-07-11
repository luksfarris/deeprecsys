from deeprecsys.rl.neural_networks.base_network import BaseNetwork
from deeprecsys.rl.neural_networks.deep_q_network import sequential_architecture
import torch
from torch.distributions import Normal
import numpy as np
from torch import FloatTensor, Tensor
from torch.optim import Adam
from deeprecsys.rl.neural_networks.q_value_estimator import TwinnedQValueEstimator

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPSILON = 1e-6


class GaussianActor(BaseNetwork):
    def __init__(
        self,
        inputs: int,
        outputs: int,
        learning_rate: float = 1e-3,
        entropy_coefficient: float = 0.2,
        discount_factor: float = 0.99,
    ):
        super().__init__()
        network_output = outputs * 2  # estimation of means and standard deviations
        layers = [inputs] + [inputs * 2, inputs * 2] + [network_output]
        self.model = sequential_architecture(layers)
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        # TODO: implement entropy learning
        self.alpha = torch.tensor(entropy_coefficient).to(self.device)
        self.gamma = discount_factor

    def forward(self, states: FloatTensor):
        mean, log_std = torch.chunk(self.model(states), 2, dim=-1)
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        return mean, log_std

    def predict(self, states: np.array):
        states_tensor = FloatTensor(states).to(device=self.device)
        # calculate Gaussian distribusion of (mean, std)
        means, log_stds = self.forward(states_tensor)
        stds = log_stds.exp()
        normals = Normal(means, stds)
        # sample actions
        xs = normals.rsample()
        actions = torch.tanh(xs)
        # calculate entropies
        log_probs = normals.log_prob(xs) - torch.log(1 - actions.pow(2) + EPSILON)
        entropies = -log_probs.sum(dim=1, keepdim=True)
        return actions, entropies, torch.tanh(means)

    def calculate_loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        dones: Tensor,
        next_states: Tensor,
        weights: Tensor,
        critic: TwinnedQValueEstimator,
    ) -> Tensor:
        """ Calculates the loss, backpropagates, and returns the entropy. """
        # We re-sample actions to calculate expectations of Q.
        sampled_action, entropy, _ = self.predict(states)
        # expectations of Q with clipped double Q technique
        q1, q2 = critic(states, sampled_action)
        q = torch.min(q1, q2)
        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        actor_loss = torch.mean((-q - self.alpha * entropy) * weights)
        self.run_backpropagation(actor_loss)
        return entropy

    def calculate_target_q(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
        target_critic: TwinnedQValueEstimator,
    ) -> Tensor:
        with torch.no_grad():
            # actor samples next actions
            next_actions, next_entropies, _ = self.predict(next_states)
            # cricic estimates q values for next actions
            next_q_critic = target_critic.predict(next_states, next_actions)
            next_q = next_q_critic + self.alpha * next_entropies
        target_q = rewards + self.gamma * next_q
        target_q[dones] = 0
        return target_q
