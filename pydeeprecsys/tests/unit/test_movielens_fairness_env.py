from gym import make
from pydeeprecsys.movielens_fairness_env import MovieLensFairness  # noqa: F401
import numpy as np


def test_env_generation():
    env = make("MovieLensFairness-v0")
    assert env is not None


def test_reset_state():
    env = make("MovieLensFairness-v0")
    state = env.reset()
    assert type(state) is np.ndarray


def test_step():
    env = make("MovieLensFairness-v0")
    env.reset()
    state, reward, done, info = env.step(123)
    assert type(state) is np.ndarray
    assert len(state) == 25
    assert 0 <= reward <= 1
    assert done in [True, False]
    assert type(info) is dict
