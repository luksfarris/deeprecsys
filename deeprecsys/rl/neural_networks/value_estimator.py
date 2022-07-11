from typing import List
from deeprecsys.rl.neural_networks.deep_q_network import sequential_architecture
from torch.optim import Adam
from torch.nn import MSELoss
from torch import FloatTensor
import numpy as np
from deeprecsys.rl.neural_networks.base_network import BaseNetwork


class ValueEstimator(BaseNetwork):
    """Estimates the value function: the expected return of being in a
    particular state"""

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        learning_rate=0.1,
    ):
        super().__init__()
        self.model = sequential_architecture(
            [input_size] + hidden_layers + [output_size]
        )
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        if self.device == "cuda":
            self.model.cuda()
        self.loss_function = MSELoss()

    def predict(self, state: np.array) -> float:
        state_tensor = FloatTensor(state).to(device=self.device)
        return self.model(state_tensor)

    def update(self, state: np.array, return_value: float):
        expected_return = FloatTensor(np.array([return_value])).to(device=self.device)
        predicted_return = self.predict(state)
        self.optimizer.zero_grad()
        loss = self.loss_function(predicted_return, expected_return)
        loss.backward()
        self.optimizer.step()
