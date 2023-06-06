import gym
from river import bandit, proba
from river import stats

import random
from Beta import Beta

class BayesUCB(bandit.base.Policy):
    '''
    This class represents a Bayes UCB model. It is a child of the bandit.base.Policy class in River.
    Parameters :
    n_arms : int. Number of arms.
    t : int. Time.
    posterior : dict. Contains the posterior distributions of all the arms.
    reward_obj : The reward object used to measure the performance of each arm. This can be a metric, a statistic, or a distribution.
    burn_in : The number of steps to use for the burn-in phase. Each arm is given the chance to be pulled during the burn-in phase. This is useful to mitigate selection bias.
    '''
    def __init__(self, n_arms: int, reward_obj=None, burn_in=0):
        super().__init__(reward_obj, burn_in)

        self.n_arms = n_arms
        self.t = 1

        self.posterior = dict()
        for arm_id in range(self.n_arms):
            self.posterior[arm_id] = Beta()
            self.posterior[arm_id].reset()

    def _pull(self, arm_ids):
        '''
        Pulls an arm of the bandit. It randomly chooses an arm among those with the highest expected reward.
        Parameters :
        arm_ids : list. List of arms in which to search for the best arm.
        '''
        index = dict()
        for arm_id in arm_ids:
            index[arm_id] = self.computeIndex(arm_id)
        maxIndex = max(index.values())
        bestArms = [arm for arm in index.keys() if index[arm] == maxIndex]

        return random.choice(bestArms)

    def computeIndex(self, arm_id):
        '''
        Returns the quantile of the posterior distribtion of a given arm.
        Parameters :
        arm_id : index of the arm.
        '''
        return self.posterior[arm_id].quantile(1 - 1. / self.t)

    def getReward(self, arm_id, reward):
        '''
        Update the posterior distribution based on the reward associated to this arm.
        Parameters :
        arm_id : index of the arm
        reward : reward that was obtained from the environment.
        '''
        self.posterior[arm_id].update(reward)
        self.t += 1
