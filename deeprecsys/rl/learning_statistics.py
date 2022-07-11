import matplotlib.pyplot as plt
import pandas
import seaborn as sns
from typing import Any, List, Dict, Optional

sns.set_theme()
sns.set_context("paper")


class LearningStatistics:
    def __init__(
        self, model_name: Optional[str] = None, env_name: Optional[str] = None
    ):
        self.collected_metrics: List[Dict] = []
        self.model_name = model_name
        self.env_name = env_name
        self.timestep = 0
        self.episode = 0

    def append_metric(self, metric_name: str, metric_value: Any):
        self.collected_metrics.append(
            {
                "metric": metric_name,
                "measurement": metric_value,
                "timestep": self.timestep,
                "episode": self.episode,
                "model": self.model_name,
                "env": self.env_name,
            }
        )

    def get_metrics(
        self, metric_name: str, model: Optional[str] = None, env: Optional[str] = None
    ) -> Optional[pandas.Series]:
        measurements = [
            v["measurement"]
            for v in self.collected_metrics
            if (v["metric"] == metric_name and v["model"] == model and v["env"] == env)
        ]
        if measurements:
            return pandas.Series(measurements)

    @property
    def moving_rewards(self) -> Optional[pandas.Series]:
        return self.get_metrics("moving_rewards")

    @property
    def episode_rewards(self) -> Optional[pandas.Series]:
        return self.get_metrics("episode_rewards")

    @property
    def epsilon_values(self) -> Optional[pandas.Series]:
        return self.get_metrics("epsilon_values")

    @property
    def loss_values(self) -> Optional[pandas.Series]:
        return self.get_metrics("loss")

    def plot_rewards(self):
        self.episode_rewards.plot()
        self.moving_rewards.plot()

    def plot_learning_stats(self):
        # generate subplots
        fig, axs = plt.subplots(2, 2)
        fig.suptitle("Agent learning metrics")
        fig.set_figheight(6)
        fig.set_figwidth(12)
        fig.subplots_adjust(hspace=0.3)
        # add data to plots
        self.episode_rewards.plot(ax=axs[0][0], title="Reward Sum")
        self.moving_rewards.plot(ax=axs[0][1], title="Reward Moving Average")
        if self.epsilon_values is not None:
            self.epsilon_values.plot(ax=axs[1][1], title="Epsilon Values")
        if self.loss_values is not None:
            self.loss_values.plot(ax=axs[1][0], title="Loss")
