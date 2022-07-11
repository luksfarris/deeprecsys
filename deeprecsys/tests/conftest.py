from pytest import fixture
import numpy as np
import os


@fixture
def random_seed() -> int:
    return 0


@fixture
def sample_state() -> np.array:
    return np.array([1, 2, 3])


@fixture
def tmp_file_cleanup():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_dir, "tmp_test_file")
    # setup test by removing file if it exists
    if os.path.exists(file_path):
        os.remove(file_path)
    yield file_path
    # tear down by removing the file if it exists
    if os.path.exists(file_path):
        os.remove(file_path)
