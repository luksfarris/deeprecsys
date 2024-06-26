from deeprecsys.rl.agents.reinforce import ReinforceAgent
from deeprecsys.rl.manager import CartpoleManager


def test_manager_init() -> None:
    manager = CartpoleManager()
    assert manager.env is not None


def test_hyperparameter_search() -> None:
    manager = CartpoleManager()
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
