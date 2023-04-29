import numpy as np
class Result:
    def __init__(self, n_arms, horizon):
        self.n_arms = n_arms
        self.choices = np.zeros(horizon)
        self.rewards = np.zeros(horizon)
    
    def store(self, t, choice, reward):
        self.choices[t] = choice
        self.rewards[t] = reward
    
    def getNbPulls(self):
        if (self.n_arms==float('inf')):
            self.nbPulls=np.array([])
            pass
        else :
            nbPulls = np.zeros(self.n_arms)
            for choice in self.choices:
                nbPulls[int(choice)] += 1
            return nbPulls
    
    def getRegret(self, bestExpectation):
        return np.cumsum(bestExpectation-self.rewards)