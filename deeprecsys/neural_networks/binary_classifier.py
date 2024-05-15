from torch import Tensor
from torch.nn import BCELoss, Linear, ReLU, Sequential, Sigmoid
from torch.optim import Adam

from deeprecsys.neural_networks.base_network import BaseNetwork


class BinaryClassifier(BaseNetwork):
    def __init__(self, input_shape: int, learning_rate: float = 0.0025):
        super().__init__()
        layers = [
            Linear(in_features=input_shape, out_features=256, bias=True),
            ReLU(),
            Linear(in_features=256, out_features=64, bias=True),
            ReLU(),
            Linear(in_features=64, out_features=16, bias=True),
            ReLU(),
            Linear(in_features=16, out_features=1, bias=True),
            Sigmoid(),
        ]
        self.model = Sequential(*layers)
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        self.loss_function = BCELoss()
        if self.device == "cuda":
            self.model.cuda()

    def update(self, features_batch: Tensor, targets_batch: Tensor) -> float:
        predicted_selection = self.forward(features_batch)
        loss = self.loss_function(predicted_selection, targets_batch)  # .reshape(-1,1))
        self.run_backpropagation(loss)
        return loss.item()
