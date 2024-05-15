import numpy as np
import torch
from torch import Tensor, device, load, save
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot


class BaseNetwork(Module):
    """Base class representing a neural network, contains utility and common methods"""

    def __init__(self) -> None:
        """Automatically detects if the device is CUDA or CPU"""
        super().__init__()
        self.device = self._auto_detect_device()

    @staticmethod
    def _auto_detect_device() -> device:
        has_cuda = torch.cuda.is_available() and torch.cuda.is_initialized()
        return torch.device("cuda" if has_cuda else "cpu")

    def save(self, path: str) -> None:
        """Write the model's parameters to the given path."""
        save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Read the model's parameters from the given path."""
        self.load_state_dict(load(path))

    def soft_parameter_update(self, source_network: Module, update_rate: float = 0.0) -> None:
        """When using target networks, this method updates the parameters of the current network
        using the parameters of the given source network. The update_rate is a float in
        range (0,1) and controls how the update affects the target (self). update_rate=0
        means a full deep copy, and update_rate=1 means the target does not update
        at all. This parameter is usually called Tau. This method is usually called
        an exponential moving average update.
        """
        for t, s in zip(self.parameters(), source_network.parameters(), strict=False):
            t.data.copy_(t.data * (1.0 - update_rate) + s.data * update_rate)

    def run_backpropagation(self, loss: Tensor) -> None:
        """Run backward on the given loss, and step the optimizer.
        Requires an optimizer property.
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def disable_learning(self) -> None:
        """Turn off the `requires_grad` parameter to stop the learning."""
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, *input: Tensor) -> Module:
        """Return the output of the network for the provided input"""
        return self.model(*input)

    def add_to_tensorboard(self, input_example: np.array) -> None:
        """Add the input to the tensor board and renders the network graph to pdf"""
        writer = SummaryWriter(f"output/writer/{type(self).__name__}")
        tensor = torch.FloatTensor(input_example)
        writer.add_graph(self, tensor, verbose=True)
        writer.close()
        graph = make_dot(
            self.forward(tensor),
            params=dict(self.named_parameters()),
            show_attrs=True,
            show_saved=True,
        )
        graph.format = "pdf"
        graph.render(f"output/graphs/{type(self).__name__}")
