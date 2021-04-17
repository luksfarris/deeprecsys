from pytest import fixture
import gym
from pydeeprecsys.ddqn import DDQN, experienceReplayBuffer, DDQNAgent
import numpy as np
from pydeeprecsys.random_agent import RandomAgent


def identity(obj, *args, **kwargs):
    return obj


@fixture
def cartpole_env() -> gym.Env:
    return gym.make("CartPole-v0")


@fixture
def cartpole_random_agent(cartpole_env):
    return RandomAgent(cartpole_env, state_encoder=identity)


@fixture
def cartpole_ddqn_agent(cartpole_env):
    n_inputs = 25  # env.observation_space.shape[0]
    n_outputs = cartpole_env.action_space.n  # env.action_space.n
    learning_rate = 25 * (10 ** -5)
    batch_size = 32
    memory_size = 10000
    burn_in = 1000
    start_epsilon = 1
    epsilon_decay = 0.99

    ddqn = DDQN(n_inputs, n_outputs, np.arange(n_outputs), learning_rate)
    erb = experienceReplayBuffer(memory_size, burn_in)
    agent = DDQNAgent(
        cartpole_env,
        ddqn,
        erb,
        start_epsilon,
        epsilon_decay,
        batch_size,
        state_encoder=(lambda s, a: s),
    )
    return agent
