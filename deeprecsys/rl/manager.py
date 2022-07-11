from gym import make, spec, Env
from collections import namedtuple, defaultdict
from typing import Any, List, Optional
import math
from numpy.random import RandomState
import numpy as np
import highway_env  # noqa: F401
import deeprecsys.movielens_fairness_env  # noqa: F401
from deeprecsys.rl.agents.agent import ReinforcementLearning
from deeprecsys.rl.learning_statistics import LearningStatistics
import torch


# An episode output is a data model to represent 3 things: how many timesteps the
# episode took to finish, the total sum of rewards, and the average reward sum of the
# last 100 episodes.
EpisodeOutput = namedtuple("EpisodeOutput", "timesteps,reward_sum")


class Manager(object):
    """ Class for learning from gym environments with some convenience methods. """

    env_name: str
    env: Any

    def __init__(
        self,
        env_name: Optional[str] = None,
        seed: Optional[int] = None,
        env: Optional[Env] = None,
        max_episode_steps: int = math.inf,
        reward_threshold: float = math.inf,
        **kwargs,
    ):
        if any(
            [env_name is None and env is None, env_name is not None and env is not None]
        ):
            raise ValueError("Must specify exactly one of [env_name, env]")
        if env_name is not None:
            self.env_name = env_name
            # extract some parameters from the environment
            self.max_episode_steps = (
                spec(self.env_name).max_episode_steps or max_episode_steps
            )
            self.reward_threshold = (
                spec(self.env_name).reward_threshold or reward_threshold
            )
            # create the environment
            self.env = make(env_name, **kwargs)
            # we seed the environment so that results are reproducible
        else:
            self.env = env
            self.max_episode_steps = max_episode_steps
            self.reward_threshold = reward_threshold

        self.setup_reproducibility(seed)
        self.slate_size: int = kwargs["slate_size"] if "slate_size" in kwargs else 1

    def print_overview(self):
        """ Prints the most important variables of the environment. """
        print("Reward threshold: {} ".format(self.reward_threshold))
        print("Reward signal range: {} ".format(self.env.reward_range))
        print("Maximum episode steps: {} ".format(self.max_episode_steps))
        print("Action apace size: {}".format(self.env.action_space))
        print("Observation space size {} ".format(self.env.observation_space))

    def execute_episodes(
        self,
        rl: ReinforcementLearning,
        n_episodes: int = 1,
        should_render: bool = False,
        should_print: bool = False,
    ) -> List[EpisodeOutput]:
        """Execute any number of episodes with the given agent.
        Returns the number of timesteps and sum of rewards per episode."""
        episode_outputs = []
        for episode in range(n_episodes):
            t, reward_sum, done, state = 0, 0, False, self.env.reset()
            if should_print:
                print(f"Running episode {episode}, starting at state {state}")
            while not done and t < self.max_episode_steps:
                if should_render:
                    self.env.render()
                action = rl.action_for_state(state)
                state, reward, done, _ = self.env.step(action)
                if should_print:
                    print(f"t={t} a={action} r={reward} s={state}")
                reward_sum += reward
                t += 1
            episode_outputs.append(EpisodeOutput(t, reward_sum))
            self.env.close()
        return episode_outputs

    def train(
        self,
        rl: ReinforcementLearning,
        statistics: Optional[LearningStatistics] = None,
        max_episodes=50,
        should_print: bool = True,
    ):
        if should_print is True:
            print("Training...")
        episode_rewards = []
        for episode in range(max_episodes):
            state = self.env.reset()
            rewards = []
            if statistics:
                statistics.episode = episode
                statistics.timestep = 0
            done = False
            while done is False:
                if self.slate_size == 1:
                    action = rl.action_for_state(state)
                else:
                    action = rl.top_k_actions_for_state(state, k=self.slate_size)
                new_state, reward, done, info = self.env.step(action)
                if "chosen_action" in info:
                    action = action[info["chosen_action"]]
                rl.store_experience(
                    state, action, reward, done, new_state
                )  # guardar experiencia en el buffer
                rewards.append(reward)
                state = new_state.copy()
                if statistics:
                    statistics.timestep += 1
            episode_rewards.append(sum(rewards))
            moving_average = np.mean(episode_rewards[-100:])
            if statistics:
                statistics.append_metric("episode_rewards", sum(rewards))
                statistics.append_metric("timestep_rewards", rewards)
                statistics.append_metric("moving_rewards", moving_average)
            if should_print is True:
                print(
                    "\rEpisode {:d} Mean Rewards {:.2f} Last Reward {:.2f}\t\t".format(
                        episode, moving_average, sum(rewards)
                    ),
                    end="",
                )
            if moving_average >= self.reward_threshold:
                if should_print is True:
                    print("Reward threshold reached")
                break

    def hyperparameter_search(
        self,
        agent: type,
        params: dict,
        default_params: dict,
        episodes: int = 100,
        runs_per_combination: int = 3,
        verbose: bool = True,
    ) -> dict:
        """Given an agent class, and a dictionary of hyperparameter names and values,
        will try all combinations, and return the mean reward of each combinatio
        for the given number of episods, and will run the determined number of times."""
        combination_results = defaultdict(lambda: [])
        for (p_name, p_value) in params.items():
            if len(p_value) < 2:
                continue
            for value in p_value:
                rl = agent(**{**default_params, p_name: value})
                learning_statistics = LearningStatistics()
                combination_key = f"{p_name}={value}"
                for run in range(runs_per_combination):
                    self.train(
                        rl=rl,
                        max_episodes=episodes,
                        should_print=False,
                        statistics=learning_statistics,
                    )
                    combination_results[combination_key].append(
                        learning_statistics.moving_rewards.iloc[-1]
                    )
                    if verbose:
                        print(
                            f"\rTested combination {p_name}={value} round {run} "
                            f"result was {learning_statistics.moving_rewards.iloc[-1]}"
                            "\t\t",
                            end="",
                        )

        return combination_results

    def setup_reproducibility(
        self, seed: Optional[int] = None
    ) -> Optional[RandomState]:
        """ Seeds the project's libraries: numpy, torch, gym """
        if seed:
            # seed pytorch
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # seed numpy
            np.random.seed(seed)
            # seed gym
            self.env.seed(seed)
            self.random_state = RandomState(seed)
            return self.random_state


class HighwayManager(Manager):
    def __init__(self, seed: Optional[int] = None, vehicles: int = 50):
        super().__init__(env_name="highway-v0", seed=seed)
        self.env.configure({"vehicles_count": vehicles})
        self.max_episode_steps = self.env.config["duration"]


class CartpoleManager(Manager):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(env_name="CartPole-v0", seed=seed)
        self.reward_threshold = 50


class LunarLanderManager(Manager):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(env_name="LunarLander-v2", seed=seed)


class MovieLensFairnessManager(Manager):
    def __init__(self, seed: Optional[int] = None, slate_size: int = 1):
        super().__init__(
            env_name="MovieLensFairness-v0", seed=seed, slate_size=slate_size
        )
