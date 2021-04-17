from pydeeprecsys.benchmark import run


def test_random_agent(cartpole_env, cartpole_random_agent):
    training_metrics = run(cartpole_random_agent, cartpole_env, episodes=50)
    assert training_metrics is not None
    assert training_metrics.average_episode_reward() > 10.0
    assert training_metrics.average_timestep_reward() == 1.0
