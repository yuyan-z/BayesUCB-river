from Result import Result
class Environment:
    """Multi-armed bandit environment"""

    def __init__(self, arms):
        self.arms = arms  # arm list
        self.n_arms = len(arms) # number of arms

    def play(self, policy, horizon):
        policy.start()
        result = Result(self.n_arms, horizon)
        for t in range(horizon):
            choice = policy.choice()
            reward = self.arms[choice].draw(t)
            # print(f'reward:{reward}')
            # reward=0
            policy.getReward(choice, reward)
            result.store(t, choice, reward)
        return result
