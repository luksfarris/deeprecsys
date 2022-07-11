from deeprecsys.rl.agents.dqn import DQNAgent


def test_decaying_epsilon_greedy(sample_state):
    agent = DQNAgent(
        1,
        1,
        [],
        initial_exploration_probability=1,
        decay_rate=0.5,
        buffer_burn_in=0,
        batch_size=0,
    )
    assert agent.action_for_state(sample_state) is not None
    agent.store_experience(sample_state, sample_state[0], 1, True, sample_state)
    assert agent.epsilon == 0.5
    for i in range(5):
        agent.store_experience(sample_state, sample_state[0], 1, True, sample_state)
    assert agent.epsilon == agent.minimum_exploration_probability
