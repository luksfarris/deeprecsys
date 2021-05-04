from pydeeprecsys.rl.manager import MovieLensFairnessManager
from pydeeprecsys.rl.agents.rainbow import RainbowDQNAgent
from numpy.random import RandomState


def main():
    manager = MovieLensFairnessManager()
    # manager = InterestEvolutionManager()
    random_state = RandomState(42)
    rainbow_agent = RainbowDQNAgent(
        input_size=25,
        output_size=manager.env.action_space.nvec[0],
        network_update_frequency=5,
        network_sync_frequency=250,
        batch_size=32,
        learning_rate=0.00025,
        discount_factor=0.9,
        buffer_size=10000,
        buffer_burn_in=32,
        random_state=random_state,
    )
    manager.train(rainbow_agent, max_episodes=100)


if __name__ == "__main__":
    main()
