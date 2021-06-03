from numpy.random import RandomState


class ExperienceReplayBufferParameters:
    """ Parameters to configure an experience replay buffer. """

    def __init__(
        self,
        max_experiences: int = 50,
        minimum_experiences_to_start_predicting: int = 32,
        batch_size: int = 32,
        random_state: RandomState = RandomState(),
    ):
        if minimum_experiences_to_start_predicting < batch_size:
            raise ValueError("The batch size mus the larger than the burn in")
        self.max_experiences = max_experiences
        self.minimum_experiences_to_start_predicting = (
            minimum_experiences_to_start_predicting
        )
        self.batch_size = batch_size
        self.random_state = random_state


class PERBufferParameters:
    """Parameters to configure the priorititization of experiences in a
    Prioritized-Experience Replay Buffer"""

    def __init__(
        self,
        beta: float = 0.01,
        beta_growth: float = 0.001,
        alpha: float = 0.6,
        epsilon: float = 0.01,
    ):
        self.beta = beta
        self.beta_growth = beta_growth
        self.alpha = alpha
        self.epsilon = epsilon
