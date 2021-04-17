import functools
import attr

from ml_fairness_gym.environments.recommenders import movie_lens_utils
from ml_fairness_gym.environments.recommenders import recsim_samplers
from ml_fairness_gym.environments.recommenders import movie_lens_dynamic as movie_lens

from recsim.simulator import recsim_gym


def prepare_environment():
    data_dir = "./ml_fairness_gym/output"
    env_config = movie_lens.EnvConfig(
        seeds=movie_lens.Seeds(0, 0),
        data_dir=data_dir,
        embeddings_path="./ml_fairness_gym/environments/recommenders/movielens_factorization.json",
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
        slate_size=1,
        resample_documents=False,
    )
    _ = env.reset()
    reward_aggregator = functools.partial(
        movie_lens.multiobjective_reward,
        lambda_non_violent=env_config.lambda_non_violent,
    )
    recsim_env = recsim_gym.RecSimGymEnv(env, reward_aggregator)
    return recsim_env
