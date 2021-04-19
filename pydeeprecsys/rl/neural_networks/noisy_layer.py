from torch.nn import Linear, Parameter, functional
import torch


class NoisyLayer(Linear):
    def __init__(
        self, in_features: int, out_features: int, bias=True, sigma: float = 0.017
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.sigma_weight = Parameter(torch.full((out_features, in_features), sigma))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = Parameter(torch.full((out_features,), sigma))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))

    def forward(self, input):
        """At every forward operation, feeds the weights and biases with normally
        distributed random variables with mean zero and std deviation 1. This means
        the bias and the weights will have a noise of:
        sigma (constant) * epsilon (random in range(-1,1))"""
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.clone()
        return functional.linear(
            input, self.weight + self.sigma_weight * self.epsilon_weight.clone(), bias
        )
