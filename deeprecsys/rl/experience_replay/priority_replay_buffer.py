from collections import namedtuple
from typing import Any, List, Tuple
import numpy
from deeprecsys.rl.experience_replay.experience_buffer import (
    Experience,
    ExperienceReplayBuffer,
)
from deeprecsys.rl.experience_replay.buffer_parameters import (
    PERBufferParameters,
    ExperienceReplayBufferParameters,
)
import numpy as np

PriorityExperience = namedtuple(
    "PriorityExperience", field_names=["experience", "priority"]
)


class PrioritizedExperienceReplayBuffer(ExperienceReplayBuffer):
    def __init__(
        self,
        buffer_parameters=ExperienceReplayBufferParameters(),
        per_parameters=PERBufferParameters(),
    ):
        super().__init__(buffer_parameters)
        # beta controls the effect of the weights (how much to learn from each
        # experience in the batch)
        self.beta = per_parameters.beta
        self.beta_growth = per_parameters.beta_growth
        # alpha controls the effect of the priority (how much priority is affected
        # by the loss)
        self.alpha = per_parameters.alpha
        # epsilon guarantees no experience has priority zero
        self.epsilon = per_parameters.epsilon

    def priorities(self) -> numpy.array:
        """ Gets the priority for each experience in the queue """
        return numpy.array(
            [e.priority for e in self.experience_queue], dtype=numpy.float32
        )

    def store_experience(
        self, state: Any, action: Any, reward: float, done: bool, next_state: Any
    ):
        """We include a priority to the experience. if the queue is empty, priority is 1 (max),
        otherwise we check the maximum priority in the queue"""
        priorities = self.priorities()
        priority = priorities.max() if len(priorities) > 0 else 1.0
        if not np.isnan(priority):
            experience = Experience(state, action, reward, done, next_state)
            priority_experience = PriorityExperience(experience, priority)
            # append to the right (end) of the queue
            self.experience_queue.append(priority_experience)

    def update_beta(self):
        """We want to grow the beta value slowly and linearly, starting at a value
        close to zero, and stopping at 1.0. This is for the Importance Sampling"""
        if self.beta < 1.0:
            self.beta += self.beta_growth

    def update_priorities(self, batch: List[Tuple], errors_from_batch: List[float]):
        """We want the priority of elements to be the TD error of plus an epsilon
        constant. The epsilon constant makes sure that no experience ever gets a
        priority zero. This prioritization strategy gives more importance to
        elements that bring more learning to the network."""
        experience_indexes = [b[-1] for b in numpy.array(batch, dtype=numpy.object).T]
        for i in range(len(experience_indexes)):
            error = abs(errors_from_batch[i]) + self.epsilon
            if not np.isnan(error):
                self.experience_queue[experience_indexes[i]] = self.experience_queue[
                    experience_indexes[i]
                ]._replace(priority=error)

    def sample_batch(self) -> List[Tuple]:
        """We sample experiences using their priorities as weights for sampling. The
        effect of the priorities is controlled by the alpha parameter. This is
        already an advantage but it can introduce bias in a network by always
        choosing the same type of experiences for training. In order to fight this, we
        compute the weight of the experience (this is called Importance Sampling,
        or IP). We want the weights to decrease over time, this is controlled by
        the beta parameter."""
        # calculate probabilities (alpha)
        probabilities = self.priorities() ** self.alpha
        p = probabilities / probabilities.sum()
        # sample experiences
        buffer_size = len(self.experience_queue)
        samples = numpy.random.choice(
            a=buffer_size, size=self.batch_size, p=p, replace=False
        )
        experiences = [self.experience_queue[i].experience for i in samples]
        # importance Sampling
        # w_i = (1/N * 1/P_i) ^ beta
        weights = ((1 / buffer_size) * (1 / p[samples])) ** self.beta
        weights = weights / weights.max()
        self.update_beta()
        # return experiences with weights
        return list(zip(*experiences)) + [tuple(weights)] + [tuple(samples)]
