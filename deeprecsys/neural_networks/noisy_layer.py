import torch
from torch import Tensor
from torch.nn import Linear, Parameter, functional


class NoisyLayer(Linear):
    """Special type of layer that adds random gaussian noise to the signal The gaussian noise parameters are
    registered, and therefore the noise decreases over time. This is a better alternative to e-greedy exploration.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sigma: float = 0.017,
    ) -> None:
        """Create the layer with the given sigma weight. Registers epsilon as a parameter so that the network will
        learn to reduce the noise.
        """
        super().__init__(in_features, out_features, bias=bias)
        self.sigma_weight = Parameter(torch.full((out_features, in_features), sigma))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = Parameter(torch.full((out_features,), sigma))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))

    def forward(self, input: Tensor) -> Tensor:
        """At every forward operation, feeds the weights and biases with normally
        distributed random variables with mean zero and std deviation 1. This means
        the bias and the weights will have a noise of:
        sigma (constant) * epsilon (random in range(-1,1))
        """
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.clone()
        return functional.linear(input, self.weight + self.sigma_weight * self.epsilon_weight.clone(), bias)
