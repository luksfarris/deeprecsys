from pytest import fixture
import numpy as np


@fixture
def random_seed() -> int:
    return 0


@fixture
def sample_state() -> np.array:
    return np.array([1, 2, 3])
