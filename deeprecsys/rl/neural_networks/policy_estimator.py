from typing import Any, List
from torch.optim import Adam
from torch.nn import Sequential, Softmax, Linear, Tanh
from torch import FloatTensor, multinomial, Tensor
from torch import sum as torch_sum
from torch.distributions import Categorical
import numpy as np
from deeprecsys.rl.neural_networks.base_network import BaseNetwork


class PolicyEstimator(BaseNetwork):
    """Estimates the policy function: the probability of each action being the
    best decision in a particular state."""

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        learning_rate=1e-2,
    ):
        super().__init__()
        layers = [input_size] + hidden_layers + [output_size]
        architecture = []
        for i in range(len(layers) - 2):
            architecture.append(Linear(layers[i], layers[i + 1]))
            architecture.append(Tanh())
        architecture.append(Linear(layers[-2], layers[-1]))
        architecture.append(Softmax(dim=-1))
        self.model = Sequential(*architecture)
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        if self.device == "cuda":
            self.model.cuda()

    def action_probabilities(self, state: Any):
        return self.model(FloatTensor(state))

    def predict(self, state: Any, k: int = 1) -> List[int]:
        probabilities = self.action_probabilities(state)
        prediction = multinomial(probabilities, num_samples=k, replacement=False)
        if self.device == "cuda":
            return prediction.detach().cpu().numpy()
        else:
            return prediction.detach().numpy()

    def update(self, state: np.array, reward_baseline: Tensor, action: np.array):
        state_tensor = FloatTensor(state).to(device=self.device)
        action_tensor = FloatTensor(np.array(action, dtype=np.float32)).to(
            device=self.device
        )
        """ Update logic from the Policy Gradient theorem. """
        action_probabilities = self.model(state_tensor)
        action_distribution = Categorical(action_probabilities)
        selected_log_probabilities = action_distribution.log_prob(action_tensor)
        loss = torch_sum(-selected_log_probabilities * reward_baseline)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.device == "cuda":
            return loss.detach().cpu().numpy()
        else:
            return loss.detach().numpy()
