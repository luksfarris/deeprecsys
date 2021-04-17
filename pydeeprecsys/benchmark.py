from pydeeprecsys.random_agent import RandomAgent
from gym import Env
from typing import List, Dict
import pandas
from collections import defaultdict
import numpy as np


class TrainingMetrics:
    def __init__(self, timestep_rewards: Dict[int, List[float]]):
        self.rewards = pandas.DataFrame.from_dict(timestep_rewards, orient="index")

    def average_episode_reward(self):
        return self.rewards.sum(axis=1).mean()

    def average_timestep_reward(self):
        return np.nanmean(self.rewards.values.flatten())


def run(agent: RandomAgent, env: Env, episodes: int = 50) -> TrainingMetrics:
    rewards = defaultdict(list)
    for episode in range(episodes):
        print(f"\r ep {episode}", end="")
        done, state = False, env.reset()
        state = agent.features_from_state(state, [])
        while not done:
            next_action = agent.get_next_action(state)
            state, reward, done, _ = env.step(next_action)
            state = agent.features_from_state(state, next_action)
            rewards[episode].append(reward)
        agent.episode_finished()

    return TrainingMetrics(rewards)
