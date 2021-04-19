import pandas
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
sns.set_context("paper")


class LearningStatistics(object):

    loss = []
    episode_rewards = []
    timestep_rewards = []
    moving_rewards = []
    epsilon_values = []
    beta_values = []

    @staticmethod
    def clear():
        LearningStatistics.loss = []
        LearningStatistics.episode_rewards = []
        LearningStatistics.timestep_rewards = []
        LearningStatistics.moving_rewards = []
        LearningStatistics.epsilon_values = []
        LearningStatistics.beta_values = []

    @staticmethod
    def plot_rewards():
        pandas.Series(LearningStatistics.episode_rewards).plot()
        pandas.Series(LearningStatistics.moving_rewards).plot()

    @staticmethod
    def plot_learning_stats():
        reward_means = pandas.Series(
            [v for v in LearningStatistics.moving_rewards][:1500]
        )
        reward_values = pandas.Series(
            [v for v in LearningStatistics.episode_rewards][:1500]
        )
        fig, axs = plt.subplots(2, 2)
        fig.suptitle("Agent learning metrics")
        fig.set_figheight(6)
        fig.set_figwidth(12)
        fig.subplots_adjust(hspace=0.3)
        reward_values.plot(ax=axs[0][0], title="Reward Sum")
        reward_means.plot(ax=axs[0][1], title="Reward Moving Average")
        if LearningStatistics.epsilon_values:
            eps_values = pandas.Series(
                [v for v in LearningStatistics.epsilon_values][:1500]
            )
            eps_values.plot(ax=axs[1][1], title="Epsilon Values")
        if LearningStatistics.loss:
            loss_values = pandas.Series(
                [v.tolist() for v in LearningStatistics.loss][:1500]
            )
            loss_values.plot(ax=axs[1][0], title="Loss")
