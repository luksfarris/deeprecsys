from torch import FloatTensor, LongTensor, BoolTensor, gather, Tensor
from numpy import array, ravel
from torch.nn import Module, ReLU, Linear, Sequential, functional
from torch.optim import Adam
from typing import List, Any, Tuple
from pydeeprecsys.rl.neural_networks.noisy_layer import NoisyLayer
from pydeeprecsys.rl.experience_replay.priority_replay_buffer import (
    PrioritizedExperienceReplayBuffer,
)
from pydeeprecsys.rl.learning_statistics import LearningStatistics


class DuelingDDQN(Module):
    """ Dueling DQN with Double DQN and Noisy Networks """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        learning_rate: float,
        noise_sigma: float = 0.17,
        discount_factor: float = 0.99,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.discount_factor = discount_factor
        self._build_network(n_input, n_output, noise_sigma)
        self.optimizer = Adam(self.parameters(), lr=learning_rate)

    def _build_network(self, n_input: int, n_output: int, noise_sigma: float):
        """Builds the dueling network with noisy layers, the value
        subnet and the advantage subnet"""
        self.fully_connected_1 = Linear(n_input, 256, bias=True)
        self.fully_connected_2 = NoisyLayer(256, 256, bias=True, sigma=noise_sigma)
        self.value_subnet = Sequential(
            NoisyLayer(256, 64, bias=True, sigma=noise_sigma),
            ReLU(),
            Linear(64, 1, bias=True),
        )
        self.advantage_subnet = Sequential(
            NoisyLayer(256, 64, bias=True, sigma=noise_sigma),
            ReLU(),
            Linear(64, n_output, bias=True),
        )

    def forward(self, state):
        """Calculates the forward between the layers"""
        layer_1_out = functional.relu(self.fully_connected_1(state))
        layer_2_out = functional.relu(self.fully_connected_2(layer_1_out))
        value_of_state = self.value_subnet(layer_2_out)
        advantage_of_state = self.advantage_subnet(layer_2_out)
        # This is the Dueling DQN part
        # Combines V and A to get Q: Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
        if len(state.shape) == 2:
            q_values = value_of_state + (
                advantage_of_state - advantage_of_state.mean(dim=1, keepdim=True)
            )
        else:
            q_values = value_of_state + (advantage_of_state - advantage_of_state.mean())
        return q_values

    def best_action_for_state(self, state: Any) -> Any:
        if type(state) is tuple:
            state = array([ravel(s) for s in state])
        state_tensor = FloatTensor(state).to(device=self.device)
        q_values = self.forward(state_tensor)
        best_action = q_values.max(-1)[1].item()
        return best_action

    def learn_with(
        self, buffer: PrioritizedExperienceReplayBuffer, target_network: Module
    ):
        experiences = buffer.sample_batch()
        self.optimizer.zero_grad()
        td_error, weights = self._calculate_td_error_and_weigths(
            experiences, target_network
        )
        loss = (td_error.pow(2) * weights).mean().to(self.device)
        loss.backward()
        self.optimizer.step()
        # store loss in statistics
        if self.device == "cuda":
            LearningStatistics.loss.append(loss.detach().cpu().numpy())
        else:
            LearningStatistics.loss.append(loss.detach().numpy())
        # update buffer priorities
        errors_from_batch = td_error.detach().cpu().numpy()
        buffer.update_priorities(experiences, errors_from_batch)

    def _calculate_td_error_and_weigths(
        self, experiences: List[Tuple], target_network: Module
    ) -> Tuple[Tensor, Tensor]:
        states, actions, rewards, dones, next_states, weights, samples = [
            i for i in experiences
        ]
        # convert to tensors
        state_tensors = FloatTensor(states).to(device=self.device)
        next_state_tensors = FloatTensor(next_states).to(device=self.device)
        reward_tensors = FloatTensor(rewards).to(device=self.device).reshape(-1, 1)
        action_tensors = (
            LongTensor(array(actions)).reshape(-1, 1).to(device=self.device)
        )
        done_tensors = BoolTensor(dones).to(device=self.device)
        weight_tensors = FloatTensor(weights).to(device=self.device)
        # the following logic is the DDQN update
        # Then we get the predicted actions for the states that came next
        # (using the main network)
        actions_for_next_states = [
            self.best_action_for_state(s) for s in next_state_tensors
        ]
        actions_for_next_states_tensor = (
            LongTensor(actions_for_next_states).reshape(-1, 1).to(device=self.device)
        )
        # Then we use them to get the estimated Q Values for these next states/actions,
        # according to the target network. Remember that the target network is a copy
        # of this one taken some steps ago
        next_q_values = target_network.forward(next_state_tensors)
        # now we get the q values for the actions that were predicted for the next state
        # we call detach() so no gradient will be backpropagated along this variable
        next_q_values_for_actions = gather(
            next_q_values, 1, actions_for_next_states_tensor
        ).detach()
        # zero value for done timesteps
        next_q_values_for_actions[done_tensors] = 0
        # bellman equation
        expected_q_values = (
            self.discount_factor * next_q_values_for_actions + reward_tensors
        )
        # Then get the Q-Values of the main network for the selected actions
        q_values = gather(self.forward(state_tensors), 1, action_tensors)
        # And compare them (this is the time-difference or TD error)
        td_error = q_values - expected_q_values
        return td_error, weight_tensors.reshape(-1, 1)
