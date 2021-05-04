import os
import functools
import attr
from mlfairnessgym.environments.recommenders import movie_lens_utils
from mlfairnessgym.environments.recommenders import recsim_samplers
from mlfairnessgym.environments.recommenders import movie_lens_dynamic as movie_lens
from recsim.simulator import recsim_gym
from gym.envs.registration import register
from gym import Env
from typing import List, Union
import numpy as np

_env_specs = {
    "id": "MovieLensFairness-v0",
    "entry_point": "pydeeprecsys.movielens_fairness_env:MovieLensFairness",
    "max_episode_steps": 100,
}
register(**_env_specs)


class MovieLensFairness(Env):
    def __init__(self, slate_size: int = 1, seed: int = 42):
        self.slate_size = slate_size
        self.internal_env = self.prepare_environment()
        self._rng = np.random.RandomState(seed)

    def step(self, action: Union[int, List[int]]):
        """ Normalize reward and flattens/normalizes state """
        if type(action) in [list, np.ndarray, np.array]:
            state, reward, done, info = self.internal_env.step(action)
            return self.movielens_state_encoder(state, action), reward / 5, done, info
        else:
            state, reward, done, info = self.internal_env.step([action])
            return self.movielens_state_encoder(state, [action]), reward / 5, done, info

    def reset(self):
        """ flattens/normalizes state """
        state = self.internal_env.reset()
        return self.movielens_state_encoder(state, [])

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

    def movielens_state_encoder(
        self, state: dict, action_slate: List[int]
    ) -> List[int]:
        """if the slate size is > 1, we need to guarantee the Single choice (SC)
        assumption, as described in the paper `SLATEQ: A Tractable Decomposition
        for Reinforcement Learning withRecommendation Sets` by randomly selecting
        one of the interactions
        """
        user_features = state["user"]
        response_features = state["response"]
        doc_features = [
            state["doc"][str(action_slate[i])]["genres"]
            for i in range(len(action_slate))
        ]
        if self.slate_size > 1:
            if response_features:
                response_features = (
                    response_features[self._rng.choice(self.slate_size)],
                )
            if doc_features:
                doc_features = [doc_features[self._rng.choice(self.slate_size)]]

        refined_state = {
            "user": user_features,
            "response": response_features,
            "slate_docs": doc_features,
        }
        # flattens the state
        return np.array(
            [
                refined_state["user"]["sex"],
                refined_state["user"]["age"],
                refined_state["user"]["occupation"],
                refined_state["user"]["zip_code"],
                *(
                    refined_state["slate_docs"][0]
                    if refined_state["slate_docs"]
                    else ([0] * 19)
                ),
                (refined_state.get("response") or ({},))[0].get("rating", 0),
                (refined_state.get("responsse") or ({},))[0].get("violence_score", 0),
            ]
        )

    def slate_action_selector(self, qvals: List[float]) -> List[float]:
        """Gets the index of the top N highest elements in the predictor array."""
        return np.argsort(qvals)[-self.slate_size :][::-1]

    def prepare_environment(self):
        current_path = os.path.dirname(__file__)
        data_dir = os.path.join(current_path, "../output")
        embeddings_path = os.path.join(
            current_path,
            "../mlfairnessgym/environments/recommenders/movielens_factorization.json",
        )
        env_config = movie_lens.EnvConfig(
            seeds=movie_lens.Seeds(0, 0),
            data_dir=data_dir,
            embeddings_path=embeddings_path,
        )
        initial_embeddings = movie_lens_utils.load_embeddings(env_config)
        # user constructor
        user_ctor = functools.partial(
            movie_lens.User, **attr.asdict(env_config.user_config)
        )
        dataset = movie_lens_utils.Dataset(
            env_config.data_dir,
            user_ctor=user_ctor,
            movie_ctor=movie_lens.Movie,
            response_ctor=movie_lens.Response,
            embeddings=initial_embeddings,
        )
        # the SingletonSampler will sample each movie once sequentially
        document_sampler = recsim_samplers.SingletonSampler(
            dataset.get_movies(), movie_lens.Movie
        )
        user_sampler = recsim_samplers.UserPoolSampler(
            seed=env_config.seeds.user_sampler,
            users=dataset.get_users(),
            user_ctor=user_ctor,
        )
        user_model = movie_lens.UserModel(
            user_sampler=user_sampler,
            seed=env_config.seeds.user_model,
        )
        env = movie_lens.MovieLensEnvironment(
            user_model,
            document_sampler,
            num_candidates=document_sampler.size(),
            slate_size=self.slate_size,
            resample_documents=False,
        )
        _ = env.reset()
        reward_aggregator = functools.partial(
            movie_lens.multiobjective_reward,
            lambda_non_violent=env_config.lambda_non_violent,
        )
        recsim_env = recsim_gym.RecSimGymEnv(env, reward_aggregator)
        return recsim_env
