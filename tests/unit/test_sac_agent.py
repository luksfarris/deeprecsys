from deeprecsys.rl.agents.soft_actor_critic import SoftActorCritic
from deeprecsys.rl.learning_statistics import LearningStatistics
from deeprecsys.rl.manager import CartpoleManager


def test_sac_init() -> None:
    # given some environment
    manager = CartpoleManager()
    # and an a SAC agent
    agent = SoftActorCritic(action_space=manager.env.action_space, state_size=4)
    # then the agent is initialized properly
    assert agent is not None


def test_sac_interaction() -> None:
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
    manager.train(agent, statistics=learning_statistics, max_episodes=200)
    # then it is able to learn
    assert learning_statistics.episode_rewards.tolist()[-1] > 30
    # and it is able to make predictions
    state, info = manager.env.reset()
    exploit_action = agent.exploit(state)
    assert exploit_action in [0, 1]
