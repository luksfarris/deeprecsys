from deeprecsys.rl.agents.rainbow import RainbowDQNAgent
from deeprecsys.rl.manager import CartpoleManager
from deeprecsys.rl.learning_statistics import LearningStatistics
from numpy.random import RandomState
from typing import Optional


def _create_agent(random_state: Optional[RandomState]) -> RainbowDQNAgent:
    return RainbowDQNAgent(
        4,
        2,
        network_update_frequency=5,
        network_sync_frequency=20,
        batch_size=16,
        learning_rate=0.001,
        discount_factor=0.95,
        buffer_size=10000,
        buffer_burn_in=32,
        random_state=random_state,
    )


def test_rainbow_init():
    agent = _create_agent(None)
    assert agent is not None


def test_reinforce_interaction():
    manager = CartpoleManager(seed=42)
    agent = _create_agent(manager.random_state)
    learning_statistics = LearningStatistics()
    manager.train(
        agent, statistics=learning_statistics, max_episodes=200, should_print=False
    )
    assert learning_statistics.episode_rewards.tolist()[-1] > 30
