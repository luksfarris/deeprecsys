from gym import Env
from typing import List, Callable
import numpy as np


def movielens_state_encoder(state: dict, action_slate: List[int]) -> List[int]:
    user_features = state["user"]
    response_features = state["response"]
    doc_features = [
        state["doc"][str(action_slate[i])]["genres"] for i in range(len(action_slate))
    ]
    refined_state = {
        "user": user_features,
        "response": response_features,
        "slate_docs": doc_features,
    }
    # flattens the state
    return [
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


def movielens_action_selector(
    action_predictors: List[float], slate_size: int = 1
) -> List[float]:
    """Gets the index of the top N highest elements in the predictor array."""
    return np.argsort(action_predictors)[-slate_size:][::-1]


class RandomAgent:
    def __init__(
        self,
        env: Env,
        state_encoder: Callable = None,
        action_selector: Callable = None,
    ):
        self.env = env
        self.features_from_state = state_encoder or movielens_state_encoder
        self.action_selector = action_selector or movielens_action_selector

    def get_next_action(self, state):
        return self.env.action_space.sample()

    def episode_finished(self):
        pass
