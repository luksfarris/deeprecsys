from typing import List
from pydeeprecsys.rl.neural_networks.deep_q_network import sequential_architecture
from torch.optim import Adam
from torch.nn import MSELoss, Module
from torch import FloatTensor
import numpy as np


class ValueEstimator(Module):
    """Estimates the value function: the expected return of being in a
    particular state"""

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        learning_rate=0.1,
        device: str = "cpu",
    ):
        super().__init__()
        self.model = sequential_architecture(
            [input_size] + hidden_layers + [output_size]
        )
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        self.device = device
        if self.device == "cuda":
            self.model.cuda()
        self.loss_function = MSELoss()

    def predict(self, state: np.array) -> float:
        state_tensor = FloatTensor(state).to(device=self.device)
        return self.model(state_tensor)

    def update(self, state: np.array, reward: float):
        expected_reward = FloatTensor(np.array([reward])).to(device=self.device)
        predicted_reward = self.predict(state)
        self.optimizer.zero_grad()
        loss = self.loss_function(predicted_reward, expected_reward)
        loss.backward()
        self.optimizer.step()
