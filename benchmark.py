import time
from random_agent import RandomAgent
from gym import Env
from typing import List


def run(agent: RandomAgent, env: Env, episodes: int = 50) -> List[float]:
    rewards = []
    total_time = 0
    total_timesteps = 0
    for episode in range(episodes):
        print(f"\r ep {episode}", end="")
        t, done, episode_reward_per_timestep, state = 0, False, 0, env.reset()
        state = agent.features_from_state(state, [])
        while not done:
            start_time = time.time()
            next_action = agent.get_next_action(state)
            state, reward, done, _ = env.step(next_action)
            state = agent.features_from_state(state, next_action)
            total_time += time.time() - start_time
            episode_reward_per_timestep += reward
            t += 1
            total_timesteps += 1
        rewards += [episode_reward_per_timestep / t]
        agent.episode_finished()
    # print(
    #     f"\nAfter running {episodes} episodes, "
    #     f"avg reward was {sum(rewards)/len(rewards)}"
    # )
    # print(f"Seconds per timestep: {(total_time/total_timesteps)*1000}")
    return rewards
