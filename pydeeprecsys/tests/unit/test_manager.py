from pydeeprecsys.rl.manager import MovieLensFairnessManager


def test_movie_lens_manager():
    manager = MovieLensFairnessManager()
    assert manager.env is not None
