from deeprecsys.rl.manager import MovieLensFairnessManager
from deeprecsys.rl.agents.reinforce import ReinforceAgent


def test_movie_lens_manager():
    manager = MovieLensFairnessManager()
    assert manager.env is not None


def test_hyperparameter_search():
    manager = MovieLensFairnessManager()
    agent = ReinforceAgent

    default_params = {
        "n_actions": manager.env.action_space.n,
        "state_size": manager.env.observation_space.shape[0],
        "hidden_layers": [64, 64],
        "discount_factor": 0.95,
        "learning_rate": 0.0001,
    }

    optimize_params = {
        "hidden_layers": [[64, 64], [128, 128], [256, 256]],
        "discount_factor": [0.9, 0.95, 0.99],
        "learning_rate": [0.00001, 0.0001, 0.001],
    }

    results = manager.hyperparameter_search(
        agent=agent,
        runs_per_combination=2,
        episodes=10,
        params=optimize_params,
        default_params=default_params,
    )

    assert results is not None
    assert len(results.items()) == 9
    assert len(results["discount_factor=0.9"]) == 2
