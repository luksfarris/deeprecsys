from deeprecsys.rl.manager import MovieLensFairnessManager, CartpoleManager
from deeprecsys.rl.experience_replay.experience_buffer import (
    ExperienceReplayBuffer,
    ExperienceReplayBufferParameters,
)
from deeprecsys.rl.agents.rainbow import RainbowDQNAgent

SEED = 42


def test_environment_seed():
    # given an environment
    manager = MovieLensFairnessManager(seed=SEED)
    # and an initial state
    state = manager.env.reset()
    # when we try to restart it several times
    for attempt in range(3):
        # and we recreate the environment
        manager = MovieLensFairnessManager(seed=SEED)
        # then the initial state is always the same
        assert (manager.env.reset() == state).all()


def _create_buffer(random_state):
    buffer = ExperienceReplayBuffer(
        parameters=ExperienceReplayBufferParameters(
            random_state=random_state, batch_size=1, max_experiences=200
        )
    )
    for i in range(200):
        buffer.store_experience(i, 0, 1, False, i + 1)
    return buffer


def test_numpy_seed():
    # given an environment
    manager = CartpoleManager(SEED)
    # and a replay buffer with some experiences
    buffer = _create_buffer(manager.random_state)
    # and an initial sample
    sample = list(buffer.sample_batch())[0]
    # when we try to get samples several times
    for attempt in range(3):
        manager.setup_reproducibility(SEED)
        buffer = _create_buffer(manager.random_state)
        # then the samples are always the same
        new_sample = list(buffer.sample_batch())[0]
        assert new_sample == sample


def test_pytorch_weights():
    # given a manager and an agent
    manager = CartpoleManager(SEED)
    agent = RainbowDQNAgent(4, 2, buffer_burn_in=32, random_state=manager.random_state)
    # when we train once
    manager.train(agent, max_episodes=100)
    model_params = list(agent.network.named_parameters())
    # and train again
    manager = CartpoleManager(SEED)
    agent = RainbowDQNAgent(4, 2, buffer_burn_in=32, random_state=manager.random_state)
    manager.train(agent, max_episodes=100)
    # then all the weights are exactly the same
    second_model_params = list(agent.network.named_parameters())
    for i in range(len(model_params)):
        first_tensor = model_params[i][1].detach().numpy().flatten().tolist()
        second_tensor = second_model_params[i][1].detach().numpy().flatten().tolist()
        assert first_tensor == second_tensor
