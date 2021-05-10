from pydeeprecsys.rl.agents.reinforce import ReinforceAgent
from pydeeprecsys.rl.manager import CartpoleManager
from pydeeprecsys.rl.learning_statistics import LearningStatistics


def test_reinforce_init():
    agent = ReinforceAgent(n_actions=2, state_size=4, discount_factor=0.95)
    assert agent is not None


def test_reinforce_interaction():
    manager = CartpoleManager()
    agent = ReinforceAgent(n_actions=2, state_size=4, discount_factor=0.95)
    learning_statistics = LearningStatistics()
    manager.train(
        agent, statistics=learning_statistics, max_episodes=200, should_print=False
    )
    assert learning_statistics.episode_rewards.tolist()[-1] > 30
