from deeprecsys.rl.neural_networks.base_network import BaseNetwork
from deeprecsys.rl.neural_networks.deep_q_network import sequential_architecture
import torch
from torch import Tensor, FloatTensor
from torch.nn import Module
from torch.optim import Adam


class QValueEstimator(BaseNetwork):
    """ Estimates the Q-value (expected return) of each (state,action) pair """

    def __init__(self, inputs: int, outputs: int, learning_rate: float = 1e-3):
        super().__init__()
        layers = [inputs] + [inputs * 2, inputs * 2] + [outputs]
        self.model = sequential_architecture(layers)
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        if self.device == "cuda":
            self.model.cuda()

    def predict(self, states: Tensor, actions: Tensor):
        inputs = torch.cat([states, actions.type(FloatTensor)], dim=1).to(
            device=self.device
        )
        return self.model(inputs)


class TwinnedQValueEstimator(BaseNetwork):
    """Estimates the Q-value (expected return) of each (state,action) pair,
    using 2 independent estimators, and predicting with the minimum estimated Q-value"""

    def __init__(self, inputs: int, outputs: int = 1, learning_rate: float = 1e-3):
        super().__init__()
        self.Q1 = QValueEstimator(inputs, outputs, learning_rate=learning_rate)
        self.Q2 = QValueEstimator(inputs, outputs, learning_rate=learning_rate)

    def predict(self, states: Tensor, actions: Tensor):
        q1, q2 = self.forward(states, actions)
        return torch.min(q1, q2)

    def forward(self, states: Tensor, actions: Tensor):
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
