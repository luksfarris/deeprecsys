from deeprecsys.rl.agents.agent import RandomAgent
import pytest
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
import numpy as np


@pytest.mark.parametrize(
    "action_space",
    [
        Discrete(10),
        Box(0, 1, (2, 2), dtype=np.float32),
    ],
)
def test_random_agent(action_space, random_seed):
    # given an agent
    agent = RandomAgent(action_space, random_state=random_seed)
    # then it has a valid action space
    assert agent.action_space == action_space
    # when we sample a new action
    action = agent.action_for_state(None)
    # then the action is valid
    assert action is not None
    # when we generate another action from a new agent but the same seed
    new_action = RandomAgent(action_space, random_state=random_seed).action_for_state(
        None
    )
    # then the actions are identical
    if type(action) == int:
        assert action == new_action
    else:
        assert action.flatten().tolist() == new_action.flatten().tolist()
    # when we request an experience to be stored
    # then nothing happens
    assert agent.store_experience(None, None, None, None, None) is None
