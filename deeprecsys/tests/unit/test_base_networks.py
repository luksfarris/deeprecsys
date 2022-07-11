from deeprecsys.rl.agents.rainbow import RainbowDQNAgent
from deeprecsys.rl.neural_networks.value_estimator import ValueEstimator
import numpy as np
from deeprecsys.movielens_fairness_env import MovieLensFairness  # noqa: F401
from deeprecsys.rl.agents.reinforce import ReinforceAgent
from deeprecsys.rl.agents.actor_critic import ActorCriticAgent


def test_save_load(tmp_file_cleanup):
    # given a neural network
    network = ValueEstimator(4, [4], 1)
    # when we train it a little bit
    inputs = np.array([1, 2, 3, 4])
    output = 1
    for i in range(20):
        network.update(inputs, output)
    # then it can accurately make predictions
    predicted_value = network.predict(inputs).detach().cpu().numpy()[0]
    assert round(predicted_value) == output
    # and when we store params
    network.save(tmp_file_cleanup)
    # and recreate the network
    network = ValueEstimator(4, [4], 1)
    network.load(tmp_file_cleanup)
    # then the prediction is the same
    assert network.predict(inputs).detach().cpu().numpy()[0] == predicted_value


def test_tensorboard_writer_reinforce():
    env = MovieLensFairness(slate_size=1)
    reinforce_agent = ReinforceAgent(
        state_size=env.observation_space.shape[0], n_actions=env.action_space.n
    )
    reinforce_agent.policy_estimator.add_to_tensorboard(env.reset())
    ac_agent = ActorCriticAgent(
        state_size=env.observation_space.shape[0], n_actions=env.action_space.n
    )
    ac_agent.value_estimator.add_to_tensorboard(env.reset())
    dqn_agent = RainbowDQNAgent(env.observation_space.shape[0], env.action_space.n)
    dqn_agent.network.add_to_tensorboard(env.reset())
    # if no errors were raised, we're good
