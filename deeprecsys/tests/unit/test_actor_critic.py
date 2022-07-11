from deeprecsys.rl.agents.actor_critic import ActorCriticAgent
from deeprecsys.rl.manager import CartpoleManager
from deeprecsys.rl.learning_statistics import LearningStatistics


def test_reinforce_init():
    agent = ActorCriticAgent(n_actions=2, state_size=4, discount_factor=0.95)
    assert agent is not None


def test_reinforce_interaction():
    manager = CartpoleManager()
    agent = ActorCriticAgent(
        n_actions=2,
        state_size=4,
        discount_factor=0.95,
        actor_learning_rate=0.001,
        critic_learning_rate=0.001,
    )
    learning_statistics = LearningStatistics()
    manager.train(
        agent, statistics=learning_statistics, max_episodes=500, should_print=False
    )
    assert learning_statistics.episode_rewards.tolist()[-1] > 30
