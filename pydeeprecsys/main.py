from pydeeprecsys.movielens_fairness_env import prepare_environment
from pydeeprecsys.random_agent import RandomAgent
from pydeeprecsys.benchmark import run
from pydeeprecsys.ddqn import create_ddqn_agent


def main():
    env = prepare_environment()
    agent = RandomAgent(env)
    agent = create_ddqn_agent(env)
    run(agent, env)


if __name__ == "__main__":
    main()
