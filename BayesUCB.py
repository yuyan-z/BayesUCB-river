import gym
from river import bandit, proba
from river import stats

import random
from Beta import Beta

class BayesUCB(bandit.base.Policy):
    def __init__(self, n_arms: int, reward_obj=None, burn_in=0):
        super().__init__(reward_obj, burn_in)

        self.n_arms = n_arms
        self.t = 1

        self.posterior = dict()
        for arm_id in range(self.n_arms):
            self.posterior[arm_id] = Beta()
            self.posterior[arm_id].reset()

    def _pull(self, arm_ids):
        index = dict()
        for arm_id in arm_ids:
            index[arm_id] = self.computeIndex(arm_id)
        maxIndex = max(index.values())
        bestArms = [arm for arm in index.keys() if index[arm] == maxIndex]

        return random.choice(bestArms)

    def computeIndex(self, arm_id):
        return self.posterior[arm_id].quantile(1 - 1. / self.t)

    def getReward(self, arm, reward):
        self.posterior[arm].update(reward)
        self.t += 1