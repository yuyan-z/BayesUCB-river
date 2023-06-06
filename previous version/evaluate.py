import gym
from river import stats
from bayes_ucb import BayesUCB
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # env = gym.make('river_bandits/CandyCaneContest-v0')
    # _ = env.reset(seed=42)
    # _ = env.action_space.seed(123)
    #
    # policy = BayesUCB(seed=123)
    #
    # metric = stats.Sum()
    # while True:
    #     action = policy.pull(range(env.action_space.n))
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     policy = policy.update(action, reward)
    #     metric = metric.update(reward)
    #     if terminated or truncated:
    #         break


    env = gym.make('river_bandits/CandyCaneContest-v0')
    _ = env.reset(seed=42)
    _ = env.action_space.seed(123)

    policy = BayesUCB(n_arms=env.action_space.n)

    metric = stats.Sum()  # cumulative reward
    hist_metrics = []
    while True:
        action = next(policy.pull(range(env.action_space.n)))
        observation, reward, terminated, truncated, info = env.step(action)
        policy = policy.update(action, reward)
        metric = metric.update(reward)
        hist_metrics.append(metric.get())
        if terminated or truncated:
            break

    print('Cumulative reward:', metric.get())

    plt.figure()
    plt.plot(hist_metrics)
    plt.xlabel("Time")
    plt.ylabel("Cumulative reward")
    plt.show()