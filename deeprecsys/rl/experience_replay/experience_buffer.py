from abc import ABC, abstractmethod
from collections import namedtuple, deque
from typing import List, Tuple, Any
from deeprecsys.rl.experience_replay.buffer_parameters import (
    ExperienceReplayBufferParameters,
)

Experience = namedtuple(
    "Experience", field_names=["state", "action", "reward", "done", "next_state"]
)


class ExperienceBuffer(ABC):
    @abstractmethod
    def ready_to_predict(self) -> bool:
        pass

    @abstractmethod
    def sample_batch(self) -> List[Tuple]:
        pass

    @abstractmethod
    def store_experience(
        self, state: Any, action: Any, reward: float, done: bool, next_state: Any
    ):
        pass


class ExperienceReplayBuffer(ExperienceBuffer):
    def __init__(
        self,
        parameters=ExperienceReplayBufferParameters(),
    ):
        self.minimum_experiences_to_start_predicting = (
            parameters.minimum_experiences_to_start_predicting
        )
        self.random_state = parameters.random_state
        # create double ended queue to store the experiences
        self.experience_queue = deque(maxlen=parameters.max_experiences)
        self.batch_size = parameters.batch_size

    def sample_batch(self) -> List[Tuple]:
        """ Samples a given number of experiences from the queue """
        # samples the index of `batch_size` different experiences from the replay memory
        samples = self.random_state.choice(
            len(self.experience_queue), self.batch_size, replace=False
        )
        # get the experiences
        experiences = [self.experience_queue[i] for i in samples]
        # returns a flattened list of the samples
        return zip(*experiences)

    def store_experience(
        self, state: Any, action: Any, reward: float, done: bool, next_state: Any
    ):
        """ Stores a new experience in the queue """
        experience = Experience(state, action, reward, done, next_state)
        # append to the right (end) of the queue
        self.experience_queue.append(experience)

    def ready_to_predict(self):
        """Returns true only if we had enough experiences to start predicting
        (measured by the burn in)"""
        return (
            len(self.experience_queue) >= self.minimum_experiences_to_start_predicting
        )
