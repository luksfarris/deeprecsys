from torch.nn import Module
from torch import save, load
import torch


class BaseNetwork(Module):
    def __init__(self):
        super().__init__()
        self.device = self._auto_detect_device()

    @staticmethod
    def _auto_detect_device():
        has_cuda = torch.cuda.is_available() and torch.cuda.is_initialized()
        return torch.device("cuda" if has_cuda else "cpu")

    def save(self, path: str):
        """ Writes the model's parameters to the given path. """
        save(self.state_dict(), path)

    def load(self, path: str):
        """ Reads the model's parameters from the given path. """
        self.load_state_dict(load(path))

    def soft_parameter_update(self, source_network: Module, update_rate: float = 0.0):
        """When using target networks, this method updates the parameters of the current network
        using the parameters of the given source network. The update_rate is a float in
        range (0,1) and controls how the update affects the target (self). update_rate=0
        means a full deep copy, and update_rate=1 means the target does not update
        at all."""
        for t, s in zip(self.parameters(), source_network.parameters()):
            t.data.copy_(t.data * (1.0 - update_rate) + s.data * update_rate)

    def run_backpropagation(self, loss):
        """Requires an optimizer property. Runs backward on the given loss, and
        steps the optimizer."""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def disable_learning(self):
        for param in self.parameters():
            param.requires_grad = False
