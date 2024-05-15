from typing import Any, List

import numpy as np
from torch import FloatTensor, Tensor, multinomial
from torch import sum as torch_sum
from torch.distributions import Categorical
from torch.nn import Linear, Sequential, Softmax, Tanh
from torch.optim import Adam

from deeprecsys.neural_networks.base_network import BaseNetwork


class PolicyEstimator(BaseNetwork):
    """Estimates the policy function: the probability of each action being the
    best decision in a particular state.
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        learning_rate: float = 1e-2,
    ):
        """Create the neural network architecture for the policy estimator with the provided values."""
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

    def action_probabilities(self, state: Any) -> Tensor:
        """Return a map of each possible action, and the probability that that's the best action to take at
        this step.
        """
        return self.model(FloatTensor(state))

    def predict(self, state: Any, k: int = 1) -> List[int]:
        """Given a state, uses the network output to choose the `k` best next actions according to the probability
        distribution trained so far.
        """
        probabilities = self.action_probabilities(state)
        prediction = multinomial(probabilities, num_samples=k, replacement=False)
        if self.device == "cuda":
            return prediction.detach().cpu().numpy()
        else:
            return prediction.detach().numpy()

    def update(self, state: np.array, reward_baseline: Tensor, action: np.array) -> np.ndarray:
        """Update the network with the given state, reward, and action taken."""
        state_tensor = FloatTensor(state).to(device=self.device)
        action_tensor = FloatTensor(np.array(action, dtype=np.float32)).to(device=self.device)
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
