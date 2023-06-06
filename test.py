import gym
import matplotlib.pyplot as plt
from river import bandit
from river import proba
from river import stats

from bayes_ucb import BayesUCB


def evaluate(policy):
    env = gym.make('river_bandits/CandyCaneContest-v0')
    _ = env.reset(seed=42)
    _ = env.action_space.seed(123)

    metric = stats.Sum()  # cumulative reward
    sum_rewards = []
    while True:
        action = next(policy.pull(range(env.action_space.n)))
        observation, reward, terminated, truncated, info = env.step(action)
        policy = policy.update(action, reward)
        metric = metric.update(reward)
        sum_rewards.append(metric.get())
        if terminated or truncated:
            break

    return sum_rewards


if __name__ == '__main__':
    bayesucb = BayesUCB(seed=123)
    egreedy = bandit.EpsilonGreedy(epsilon=0.9, seed=101)
    ucb = bandit.UCB(delta=100)
    ts = bandit.ThompsonSampling(dist=proba.Beta(), seed=101)

    plt.figure()
    plt.plot(evaluate(bayesucb), 'b', label='BayesUCB')
    plt.plot(evaluate(egreedy), 'r', label='EpsilonGreedy')
    plt.plot(evaluate(ucb), 'g', label='UCB')
    plt.plot(evaluate(ts), 'y', label='ThompsonSampling')
    plt.xlabel("Steps")
    plt.ylabel("Cumulative reward")
    plt.legend()
    plt.show()
