from gym import Env
from typing import List


def movielens_feature_transformer(state: dict, action_slate: List[int]) -> List[int]:
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


def movielens_action_transformer(action) -> List[float]:
    pass


class RandomAgent:
    def __init__(self, env: Env, feature_transformer=None):
        self.env = env
        self.features_from_state = feature_transformer or movielens_feature_transformer

    def get_next_action(self, state):
        return self.env.action_space.sample()

    def episode_finished(self):
        pass
