from pydeeprecsys.rl.agents.soft_actor_critic import SoftActorCritic
from pydeeprecsys.rl.manager import CartpoleManager
from pydeeprecsys.rl.learning_statistics import LearningStatistics


def test_reinforce_init():
    # given some environment
    manager = CartpoleManager()
    # and an a SAC agent
    agent = SoftActorCritic(action_space=manager.env.action_space, state_size=4)
    # then the agent is initialized properly
    assert agent is not None


def test_reinforce_interaction():
    # given an environment
    manager = CartpoleManager()
    # and a SAC agent
    agent = SoftActorCritic(
        action_space=manager.env.action_space,
        state_size=4,
        discount_factor=0.95,
        learning_rate=0.001,
        timesteps_to_start_predicting=64,
        target_update_interval=1,
    )
    # when we train the agent
    learning_statistics = LearningStatistics()
    manager.train(
        agent, statistics=learning_statistics, max_episodes=200, should_print=False
    )
    # then it is able to learn
    assert learning_statistics.episode_rewards.tolist()[-1] > 30
    # and it is able to make predictions
    exploit_action = agent.exploit(manager.env.reset())
    assert exploit_action in [0, 1]
