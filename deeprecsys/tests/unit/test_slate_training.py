from deeprecsys.rl.manager import MovieLensFairnessManager
from deeprecsys.rl.agents.agent import RandomAgent
from deeprecsys.rl.agents.reinforce import ReinforceAgent
import pytest


@pytest.mark.parametrize("slate_size", [5, 10, 30])
def test_slate_interaction(slate_size):
    manager = MovieLensFairnessManager(slate_size=slate_size, seed=42)
    env = manager.env
    assert env.reset() is not None
    example_action = env.action_space.sample()
    assert len(example_action) == slate_size
    assert env.step(example_action) is not None


@pytest.mark.parametrize("slate_size", [5, 10, 30])
def test_slate_random_training(slate_size):
    manager = MovieLensFairnessManager(slate_size=slate_size, seed=42)
    agent = RandomAgent(manager.env.action_space)
    manager.train(agent, max_episodes=30)


@pytest.mark.parametrize("slate_size", [5, 10, 30])
def test_slate_reinforce_training(slate_size):
    manager = MovieLensFairnessManager(slate_size=slate_size, seed=42)
    agent = ReinforceAgent(
        int(manager.env.action_space.nvec[0]),
        manager.env.observation_space.shape[0],
        discount_factor=0.95,
        learning_rate=0.0001,
    )
    manager.train(agent, max_episodes=200)
