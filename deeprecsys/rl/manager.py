import math
from collections import defaultdict, namedtuple
from typing import Any, Generator, List, Optional

import numpy as np
import torch
from gymnasium import Env, make, spec
from numpy.core.multiarray import ndarray
from numpy.random import RandomState

from deeprecsys.rl import Logger
from deeprecsys.rl.agents.agent import ReinforcementLearning
from deeprecsys.rl.learning_statistics import LearningStatistics

# An episode output is a data model to represent 3 things: how many timesteps the
# episode took to finish, the total sum of rewards, and the average reward sum of the
# last 100 episodes.
EpisodeOutput = namedtuple("EpisodeOutput", "timesteps,reward_sum")

logger = Logger.create()


class Manager:
    """Class for learning from gym environments with some convenience methods."""

    env_name: str
    env: Any
    seed: int | None = None
    random_state: RandomState | None = None

    def __init__(
        self,
        env_name: Optional[str] = None,
        seed: Optional[int] = None,
        env: Optional[Env] = None,
        max_episode_steps: float = math.inf,
        reward_threshold: float = math.inf,
        **kwargs: Any,
    ) -> None:
        """Start the manager"""
        if any([env_name is None and env is None, env_name is not None and env is not None]):
            raise ValueError("Must specify exactly one of [env_name, env]")
        if env_name is not None:
            self.env_name = env_name
            # extract some parameters from the environment
            self.max_episode_steps = spec(self.env_name).max_episode_steps or max_episode_steps
            self.reward_threshold = spec(self.env_name).reward_threshold or reward_threshold
            # create the environment
            self.env = make(env_name, **kwargs)
            # we seed the environment so that results are reproducible
        else:
            self.env = env
            self.max_episode_steps = max_episode_steps
            self.reward_threshold = reward_threshold

        self.setup_reproducibility(seed)
        self.slate_size: int = kwargs["slate_size"] if "slate_size" in kwargs else 1

    def print_overview(self) -> None:
        """Print the most important variables of the environment."""
        logger.info("Reward threshold: {} ".format(self.reward_threshold))
        logger.info("Reward signal range: {} ".format(self.env.reward_range))
        logger.info("Maximum episode steps: {} ".format(self.max_episode_steps))
        logger.info("Action apace size: {}".format(self.env.action_space))
        logger.info("Observation space size {} ".format(self.env.observation_space))

    def execute_episodes(
        self,
        rl: ReinforcementLearning,
        n_episodes: int = 1,
        should_render: bool = False,
    ) -> List[EpisodeOutput]:
        """Execute any number of episodes with the given agent.
        Returns the number of timesteps and sum of rewards per episode.
        """
        episode_outputs = []
        for episode in range(n_episodes):
            t, reward_sum, done, (state, _) = 0, 0, False, self.env.reset(seed=self.seed)
            logger.info(f"Running episode {episode}, starting at state {state}")
            while not done and t < self.max_episode_steps:
                if should_render:
                    self.env.render()
                action = rl.action_for_state(state)
                state, reward, done, _ = self.env.step(action)
                logger.info(f"t={t} a={action} r={reward} s={state}")
                reward_sum += reward
                t += 1
            episode_outputs.append(EpisodeOutput(t, reward_sum))
            self.env.close()
        return episode_outputs

    @staticmethod
    def _train_start_new_episode(statistics: LearningStatistics, episode: int) -> None:
        if statistics:
            statistics.episode = episode
            statistics.timestep = 0

    @staticmethod
    def _train_update_timestep(statistics: LearningStatistics) -> None:
        if statistics:
            statistics.timestep += 1

    @staticmethod
    def _train_add_statistics(statistics: LearningStatistics, rewards: List, moving_average: ndarray) -> None:
        if statistics:
            statistics.append_metric("episode_rewards", sum(rewards))
            statistics.append_metric("timestep_rewards", rewards)
            statistics.append_metric("moving_rewards", moving_average)

    def _train_get_step_action(self, rl: ReinforcementLearning, state: Any) -> Any:
        if self.slate_size == 1:
            return rl.action_for_state(state)
        else:
            return rl.top_k_actions_for_state(state, k=self.slate_size)

    def train(
        self,
        rl: ReinforcementLearning,
        statistics: Optional[LearningStatistics] = None,
        max_episodes: int = 50,
    ) -> None:
        """Train the agent for the given amount of episodes."""
        logger.info("Training...")
        episode_rewards = []
        for episode in range(max_episodes):
            state, info = self.env.reset(seed=self.seed)
            rewards = []
            self._train_start_new_episode(statistics, episode)
            done = False
            while done is False:
                action = self._train_get_step_action(rl, state)
                new_state, reward, done, _, info = self.env.step(action)
                if "chosen_action" in info:
                    action = action[info["chosen_action"]]
                rl.store_experience(state, action, reward, done, new_state)
                rewards.append(reward)
                state = new_state.copy()
                self._train_update_timestep(statistics)
            episode_rewards.append(sum(rewards))
            moving_average = np.mean(episode_rewards[-100:])
            self._train_add_statistics(statistics, rewards, moving_average)

            logger.print(
                f"\rEpisode {episode:d} Mean Rewards {moving_average:.2f} Last Reward {rewards[-1]:.2f}\t\t",
                end="",
            )
            if moving_average >= self.reward_threshold:
                logger.info("Reward threshold reached")
                break

    def _hyperparameter_search_run_combinations(
        self,
        runs_per_combination: int,
        rl: ReinforcementLearning,
        episodes: int,
        learning_statistics: LearningStatistics,
        parameter_name: str,
        parameter_value: Any,
    ) -> Generator:
        for run in range(runs_per_combination):
            self.train(
                rl=rl,
                max_episodes=episodes,
                statistics=learning_statistics,
            )
            yield learning_statistics.moving_rewards.iloc[-1]
            logger.print(
                f"\rTested combination {parameter_name}={parameter_value} round {run} result was {learning_statistics.moving_rewards.iloc[-1]} \t\t",  # noqa: E501
                end="",
            )

    def hyperparameter_search(
        self,
        agent: type,
        params: dict,
        default_params: dict,
        episodes: int = 100,
        runs_per_combination: int = 3,
    ) -> dict:
        """Given an agent class, and a dictionary of hyperparameter names and values,
        will try all combinations, and return the mean reward of each combination
        for the given number of episodes, and will run the determined number of times.
        """
        combination_results = defaultdict(lambda: [])
        for p_name, p_value in params.items():
            if len(p_value) < 2:
                continue
            for value in p_value:
                rl = agent(**{**default_params, p_name: value})
                learning_statistics = LearningStatistics()
                combination_key = f"{p_name}={value}"
                for result in self._hyperparameter_search_run_combinations(
                    runs_per_combination,
                    rl,
                    episodes,
                    learning_statistics,
                    p_name,
                    value,
                ):
                    combination_results[combination_key].append(result)

        return combination_results

    def setup_reproducibility(self, seed: Optional[int] = None) -> Optional[RandomState]:
        """Seeds the project's libraries: numpy, torch, gym"""
        if seed:
            # seed pytorch
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # seed numpy
            np.random.seed(seed)
            # seed gym
            self.seed = seed
            self.random_state = RandomState(seed)
            return self.random_state
        return None


class HighwayManager(Manager):
    """Manager for the highway environment"""

    def __init__(self, seed: Optional[int] = None, vehicles: int = 50):
        """Start the manager"""
        super().__init__(env_name="highway-v0", seed=seed)
        self.env.configure({"vehicles_count": vehicles})
        self.max_episode_steps = self.env.config["duration"]


class CartpoleManager(Manager):
    """Manager for the cart pole environment"""

    def __init__(self, seed: Optional[int] = None):
        """Start the manager"""
        super().__init__(env_name="CartPole-v1", seed=seed)
        self.reward_threshold = 50


class LunarLanderManager(Manager):
    """Manager for the lunar lander environment"""

    def __init__(self, seed: Optional[int] = None):
        """Start the manager"""
        super().__init__(env_name="LunarLander-v2", seed=seed)


class MovieLensFairnessManager(Manager):
    """Manager for the movie lens environment"""

    def __init__(self, seed: Optional[int] = None, slate_size: int = 1):
        """Start the manager"""
        super().__init__(env_name="MovieLensFairness-v0", seed=seed, slate_size=slate_size)
