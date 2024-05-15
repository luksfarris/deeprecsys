from abc import ABC, abstractmethod
from collections import deque, namedtuple
from typing import Any, List, Tuple

from deeprecsys.rl.experience_replay.buffer_parameters import (
    ExperienceReplayBufferParameters,
)

Experience = namedtuple(  # type: ignore
    "Experience", field_names=["state", "action", "reward", "done", "next_state"]
)


class ExperienceBuffer(ABC):
    """Abstract class encapsulating common aspects of an experience replay buffer."""

    @abstractmethod
    def ready_to_predict(self) -> bool:
        """Whether enough experiences were collected"""

    @abstractmethod
    def sample_batch(self) -> List[Tuple]:
        """Sample a batch of experiences from the buffer."""

    @abstractmethod
    def store_experience(self, state: Any, action: Any, reward: float, done: bool, next_state: Any) -> None:
        """Store an experience in the buffer."""


class ExperienceReplayBuffer(ExperienceBuffer):
    """Traditional experience replay buffer. Experiences are sampled randomly without
    replacements within batches. Different batches may contain the same experience.
    """

    def __init__(
        self,
        parameters: ExperienceReplayBufferParameters = None,
    ):
        """Initialize the buffer with the provided parameters."""
        if not parameters:
            parameters = ExperienceReplayBufferParameters()
        self.minimum_experiences_to_start_predicting = parameters.minimum_experiences_to_start_predicting
        self.random_state = parameters.random_state
        # create double ended queue to store the experiences
        self.experience_queue: List = list(deque(maxlen=parameters.max_experiences))
        self.batch_size = parameters.batch_size

    def sample_batch(self) -> List[Tuple]:
        """Sample a given number of experiences from the queue"""
        # samples the index of `batch_size` different experiences from the replay memory
        samples = self.random_state.choice(len(self.experience_queue), self.batch_size, replace=False)
        # get the experiences
        experiences = [self.experience_queue[i] for i in samples]
        # returns a flattened list of the samples
        return zip(*experiences, strict=False)  # type: ignore

    def store_experience(self, state: Any, action: Any, reward: float, done: bool, next_state: Any) -> None:
        """Store a new experience in the queue"""
        experience = Experience(state, action, reward, done, next_state)  # type: ignore
        # append to the right (end) of the queue
        self.experience_queue.append(experience)

    def ready_to_predict(self) -> bool:
        """Return true only if we had enough experiences to start predicting
        (measured by the burn in)
        """
        return len(self.experience_queue) >= self.minimum_experiences_to_start_predicting
