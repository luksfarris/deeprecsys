#!/usr/bin/env python


""" [ES] Este proyecto es una implementación de algunos algoritmos de aprendizaje por refuerzo para
la práctica de M2.883 - Aprendizaje por refuerzo, del Máster universitario en Ciencia de datos Data science)
de la UOC - Estudios de Informática, Multimedia y Telecomunicación.

[EN] This project is an implementation of some Reinforcement Learning algorighms for the Deep Reinforcement Learning
subject in the Data Science MSc degree of the UOC, department of Informatics, Multimedia, and Telecom. """

__author__ = "Lucas Farris"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Lucas Farris"
__email__ = "lfarris@uoc.edu"
__status__ = "Development"


from pydeeprecsys.rl.manager import HighwayManager
from numpy.random import RandomState
from pydeeprecsys.rl.agents.rainbow import RainbowDQNAgent


def test_params_search():
    """Ejemplo de busqueda de hiperparametros"""
    manager = HighwayManager(vehicles=10)
    input_size = (
        manager.env.observation_space.shape[0] * manager.env.observation_space.shape[1]
    )
    output_size = manager.env.action_space.n
    random_state = RandomState(42)
    default_params = {"input_size": input_size, "output_size": output_size}
    params = {
        "network_update_frequency": [3, 5, 10],
        "network_sync_frequency": [200, 300],
        "priority_importance": [0.4, 0.6, 0.8],
        "priority_weigth_growth": [0.001, 0.005, 0.01],
        "noise_sigma": [0.01, 0.02, 0.05],
    }
    output = manager.hyperparameter_search(
        RainbowDQNAgent,
        episodes=300,
        default_params=default_params,
        runs_per_combination=3,
        params=params,
    )


def test_run():
    """Ejemplo de como entrenar el agente"""
    manager = HighwayManager(vehicles=5)
    random_state = RandomState(42)
    input_size = (
        manager.env.observation_space.shape[0] * manager.env.observation_space.shape[1]
    )
    output_size = manager.env.action_space.n
    dqn_agent = RainbowDQNAgent(
        input_size,
        output_size,
        network_update_frequency=5,
        network_sync_frequency=250,
        batch_size=32,
        learning_rate=0.00025,
        discount_factor=0.9,
        buffer_size=10000,
        buffer_burn_in=32,
        random_state=random_state,
    )
    manager.train(dqn_agent, max_episodes=500)


if __name__ == "__main__":
    pass
