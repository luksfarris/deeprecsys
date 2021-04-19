import os
import functools
import attr
from mlfairnessgym.environments.recommenders import movie_lens_utils
from mlfairnessgym.environments.recommenders import recsim_samplers
from mlfairnessgym.environments.recommenders import movie_lens_dynamic as movie_lens
from recsim.simulator import recsim_gym
from gym.envs.registration import register
from gym import Env
from typing import List
import numpy as np
from recsim.environments import interest_evolution

_env_specs = {
    "id": "InterestEvolution-v0",
    "entry_point": "pydeeprecsys.interest_evolution_env:InterestEvolution",
}
register(**_env_specs)


class InterestEvolution(Env):
    """ TODO: implement multiple slate sizes """

    def __init__(self, slate_size: int = 1):
        self.internal_env = self.prepare_environment()
        self.slate_size = slate_size

    def step(self, action):
        """ Normalize reward and flattens/normalizes state """
        state, reward, done, info = self.internal_env.step([action])
        return self.state_encoder(state, [action]), reward, done, info

    def reset(self):
        """ flattens/normalizes state """
        state = self.internal_env.reset()
        return self.state_encoder(state, [])

    def render(self, mode="human", close=False):
        return self.internal_env.render(mode)

    @property
    def action_space(self):
        return self.internal_env.action_space

    @property
    def reward_range(self):
        return self.internal_env.reward_range

    @property
    def observation_space(self):
        return self.internal_env.observation_space

    @staticmethod
    def state_encoder(state: dict, action_slate: List[int]) -> List[int]:
        user_features = state["user"]
        response_features = state["response"]
        # doc_features = [
        #     state["doc"][str(action_slate[i])]["genres"]
        #     for i in range(len(action_slate))
        # ]
        refined_state = {
            "user": user_features,
            "response": response_features,
            # "doc": doc_features[0] if doc_features else [],  # TODO: update for slates
        }
        # flattens the state
        return np.array(
            [
                *refined_state["user"],
                # *(refined_state["doc"] if refined_state["doc"] else ([0] * 20)),
                # 0,
                # 0,
            ]
        )

    @staticmethod
    def prepare_environment():
        env_config = {
            "num_candidates": 100,
            "slate_size": 1,
            "resample_documents": True,
            "seed": 42,
        }
        return interest_evolution.create_environment(env_config)
