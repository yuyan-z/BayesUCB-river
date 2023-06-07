import gym
import copy
import matplotlib.pyplot as plt
import pandas as pd

from river import bandit
from river import proba
from river import stats

from bayes_ucb import BayesUCB


def pull_func(policy, env):
    return next(policy.pull(range(env.action_space.n)))


def test(policies, env, n_episodes, metrictype):
    trace = []

    for policy_idx, policy in enumerate(policies):
        for episode in range(n_episodes):
            episode_policy = policy.clone()
            episode_env = copy.deepcopy(env)
            episode_env.reset(seed=42)
            episode_env.action_space.seed(123)
            step = 0

            metric = stats.Sum()
            if metrictype == 'mean reward':
                metric = stats.Mean()

            while True:
                action = next(episode_policy.pull(range(env.action_space.n)))
                observation, reward, terminated, truncated, info = episode_env.step(action)
                metric = metric.update(reward)
                episode_policy.update(action, reward)

                trace.append({
                    "episode": episode,
                    "policy_idx": policy_idx,
                    "step": step,
                    "reward": reward,
                    metrictype: metric.get(),
                })
                step += 1

                if terminated or truncated:
                    break

    trace_df = pd.DataFrame(trace)
    return trace_df

if __name__ == '__main__':
    policies = [
        BayesUCB(seed=123),
        bandit.EpsilonGreedy(epsilon=0.9, seed=101),
        bandit.UCB(delta=100),
        bandit.ThompsonSampling(dist=proba.Beta(), seed=101),
    ]

    env = gym.make('river_bandits/CandyCaneContest-v0', max_episode_steps=1000)

    n_episodes = 10

    policy_names = {
        0: 'BayesUCB',
        1: 'EpsilonGreedy',
        2: 'UCB',
        3: 'ThompsonSampling'
    }

    colors = {
        'BayesUCB': 'tab:blue',
        'EpsilonGreedy': 'tab:red',
        'UCB': 'tab:green',
        'ThompsonSampling': 'tab:orange'
    }

    trace_sum = test(policies, env, n_episodes, "sum reward")
    trace_mean = test(policies, env, n_episodes, "mean reward")

    # example
    print(trace_sum.sample(5, random_state=42))

    fig, axes = plt.subplots(nrows=1, ncols=2)

    (
        trace_sum
            .assign(policy=trace_sum.policy_idx.map(policy_names))
            .groupby(['step', 'policy'])
        ["sum reward"].mean()
            .unstack()
            .plot(ax=axes[0], color=colors)
    )

    (
        trace_mean
            .assign(policy=trace_mean.policy_idx.map(policy_names))
            .groupby(['step', 'policy'])
        ["mean reward"].mean()
            .unstack()
            .plot(ax=axes[1], color=colors)
    )

    axes[0].set_ylabel('sum reward')
    axes[1].set_ylabel('mean reward')
    plt.show()
