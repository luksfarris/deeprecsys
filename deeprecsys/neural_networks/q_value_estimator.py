from typing import Tuple

import torch
from torch import FloatTensor, Tensor
from torch.nn import Module
from torch.optim import Adam

from deeprecsys.neural_networks.base_network import BaseNetwork
from deeprecsys.neural_networks.deep_q_network import sequential_architecture


class QValueEstimator(BaseNetwork):
    """Estimate the Q-value (expected return) of each (state,action) pair"""

    def __init__(self, inputs: int, outputs: int, learning_rate: float = 1e-3):
        """Create the network architecture with the provided parameters."""
        super().__init__()
        layers = [inputs] + [inputs * 2, inputs * 2] + [outputs]
        self.model = sequential_architecture(layers)
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        if self.device == "cuda":
            self.model.cuda()

    def predict(self, states: Tensor, actions: Tensor) -> Tensor:
        """Given a state and an action, return the estimated Q-Value"""
        inputs = torch.cat([states, actions.type(FloatTensor)], dim=1).to(device=self.device)
        return self.model(inputs)


class TwinnedQValueEstimator(BaseNetwork):
    """Estimate the Q-value (expected return) of each (state,action) pair,
    using 2 independent estimators, and predicting with the minimum estimated Q-value.
    This is the "critic" part of the Actor-Critic model.
    """

    def __init__(self, inputs: int, outputs: int = 1, learning_rate: float = 1e-3):
        """Create the two estimators with the provided parameters."""
        super().__init__()
        self.Q1 = QValueEstimator(inputs, outputs, learning_rate=learning_rate)
        self.Q2 = QValueEstimator(inputs, outputs, learning_rate=learning_rate)

    def predict(self, states: Tensor, actions: Tensor) -> Tensor:
        """Given a (state, action) pair return the smaller Q-value of the two networks."""
        q1, q2 = self.forward(states, actions)
        return torch.min(q1, q2)

    def forward(self, states: Tensor, actions: Tensor) -> Tuple:
        """Calculate the output weighs for the given (state, action) pair"""
        q1 = self.Q1.predict(states, actions)
        q2 = self.Q2.predict(states, actions)
        return q1, q2

    def calculate_loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        dones: Tensor,
        next_states: Tensor,
        weights: Tensor,
        actor: Module,
        target: "TwinnedQValueEstimator",
    ) -> Tensor:
        """Train the network and return the loss."""
        curr_q1, curr_q2 = self(states, actions)
        target_q = actor.calculate_target_q(
            states,
            actions,
            rewards,
            next_states,
            dones,
            target_critic=target,
        )
        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)
        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)
        self.Q1.run_backpropagation(q1_loss)
        self.Q2.run_backpropagation(q2_loss)
        return errors
