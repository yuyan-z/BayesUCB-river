import random
from Beta import Beta
class BayesUCB():
    """
    
    """
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.posterior = dict()
        for arm in range(self.n_arms):
            self.posterior[arm] = Beta()
    def start(self):
        self.t = 1;
        for arm in range(self.n_arms):
            self.posterior[arm].reset()
    def getReward(self, arm, reward):
        self.posterior[arm].update(reward)
        self.t += 1
    def computeIndex(self, arm):
        return self.posterior[arm].quantile(1-1./self.t)

    def choice(self):
        """choose an arm with maximal index."""
        index = dict()
        for arm in range(self.n_arms):
            index[arm] = self.computeIndex(arm)
        maxIndex = max(index.values())
        bestArms = [arm for arm in index.keys() if index[arm] == maxIndex]
        return random.choice(bestArms)