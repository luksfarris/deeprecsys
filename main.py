from movielens_fairness_env import prepare_environment
from random_agent import RandomAgent
from benchmark import run
from ddqn import create_ddqn_agent


def main():
    env = prepare_environment()
    agent = RandomAgent(env)
    agent = create_ddqn_agent(env)
    run(agent, env)


if __name__ == "__main__":
    main()
