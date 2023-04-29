import matplotlib.pyplot as plt
from BayesUCB import BayesUCB
from river import stats
import gym
if __name__ == '__main__':
    env = gym.make('river_bandits/CandyCaneContest-v0')
    _ = env.reset(seed=42)
    _ = env.action_space.seed(123)

    policy = BayesUCB(n_arms=env.action_space.n)

    metric = stats.Sum()
    while True:
        action = next(policy.pull(range(env.action_space.n)))
        observation, reward, terminated, truncated, info = env.step(action)
        policy = policy.update(action, reward)
        policy.getReward(action, reward)
        metric = metric.update(reward)
        if terminated or truncated:
            break

    print('Sum:', metric.get())