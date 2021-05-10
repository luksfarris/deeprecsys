from pydeeprecsys.rl.agents.rainbow import RainbowDQNAgent
from pydeeprecsys.rl.manager import CartpoleManager
from pydeeprecsys.rl.learning_statistics import LearningStatistics
from numpy.random import RandomState


def _create_agent():
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
        random_state=RandomState(42),
    )


def test_rainbow_init():
    agent = _create_agent()
    assert agent is not None


def test_reinforce_interaction():
    manager = CartpoleManager()
    agent = _create_agent()
    learning_statistics = LearningStatistics()
    manager.train(
        agent, statistics=learning_statistics, max_episodes=200, should_print=False
    )
    assert learning_statistics.episode_rewards.tolist()[-1] > 30