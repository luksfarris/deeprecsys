from typing import Any, List, Tuple

import numpy
import pandas
from torch import FloatTensor
from torch.utils.data import DataLoader, Dataset

from deeprecsys.neural_networks.binary_classifier import BinaryClassifier


class HomelikeDataset(Dataset):
    def __init__(self) -> None:
        base_path = "/home/farris/Developer/hl-ranking-algorithm/hl_ranking_algorithm/live_ranking/offline_training/"
        self.users = (
            pandas.read_feather(base_path + "users.feather")
            .set_index("session_id")
            .dropna()
        )
        self.history = pandas.read_feather(base_path + "history.feather")
        self.inventory = (
            pandas.read_feather(base_path + "inventory.feather")
            .set_index("pg_id")
            .astype("float64")
        )
        self.length = self.history.action.apply(len).sum()
        self.user_index = 0
        self.user_history: List[Any] = []
        self.user_features = None
        self.num_features = self.users.shape[1] + self.inventory.shape[1]

    def __getitem__(self, *_: Any) -> Tuple[FloatTensor, FloatTensor]:
        if len(self.user_history) == 0:
            self.user_features = self.users.iloc[self.user_index]
            self.user_index += 1
            user_history = self.history.query(
                f"session_id == '{self.users.index[0]}'"
            ).explode("action")
            user_history["reward"] = user_history.apply(
                lambda df: df.action in df.reward, axis="columns"
            )
            self.user_history = user_history[["action", "reward"]].values.tolist()
        action, reward = self.user_history.pop()
        item_features = self.inventory.loc[action].values
        return (
            FloatTensor(numpy.concatenate((item_features, self.user_features), axis=0)),
            FloatTensor([reward]),
        )

    def __len__(self) -> int:
        return self.length


def generate_network_parameters(batch_size: int = 256) -> None:
    dataset = HomelikeDataset()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    classifier = BinaryClassifier(input_shape=dataset.num_features)
    for epoch in range(1):
        for ix, data in enumerate(data_loader, 0):
            features, targets = data
            loss = classifier.update(features, targets)
            print(
                f"Epoch {epoch + 1} Batch {ix}/{int(dataset.length/batch_size)} Loss {loss}"
            )
    classifier.save("visit_simulator.pch")


if __name__ == "__main__":
    generate_network_parameters()


# class HomelikeApartmentSearch(Env):
#     def __init__(self, seed: Optional[int] = 42, steps_per_episode: int = 30):
#         self._rng = np.random.RandomState(seed=seed)
#         self.user_history = []
#         self.steps_per_episode = steps_per_episode
#
#     @staticmethod
#     def _get_user_selection_probabilities(user_id: str):
#         sessions = history.query(f"session_id == '{user_id}'")[["action", "reward"]]
#         sessions_unnested = sessions.explode("action")
#         sessions_unnested["reward"] = sessions_unnested.apply(
#             lambda df: df.action in df.reward, axis="columns"
#         )
#         return (
#             sessions_unnested.reset_index(drop=True)
#             .groupby("action")
#             .mean()
#             .rename(columns={"reward": "select_probability"})
#             .squeeze(axis="columns")
#         )
#
#     def reset(self) -> Dict:
#         random_user = users.session_id.sample(1).values[0]
#         self.selection_probabilities = self._get_user_selection_probabilities(
#             random_user
#         )
#         self.step = 0
#         return {"user": random_user, "step": self.step}
#
#     def step(self, action: str) -> Tuple:
#         simulate_selection = ...
#         return encoded_state, reward / 5, done, info
#
#
# HomelikeApartmentSearch().reset()
