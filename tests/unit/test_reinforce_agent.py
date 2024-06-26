from deeprecsys.rl.agents.reinforce import ReinforceAgent
from deeprecsys.rl.learning_statistics import LearningStatistics
from deeprecsys.rl.manager import CartpoleManager


def test_reinforce_init() -> None:
    agent = ReinforceAgent(n_actions=2, state_size=4, discount_factor=0.95)
    assert agent is not None


def test_reinforce_interaction() -> None:
    manager = CartpoleManager()
    agent = ReinforceAgent(n_actions=2, state_size=4, discount_factor=0.95, learning_rate=0.001)
    learning_statistics = LearningStatistics()
    manager.train(agent, statistics=learning_statistics, max_episodes=500)
    assert learning_statistics.episode_rewards.tolist()[-1] > 30
