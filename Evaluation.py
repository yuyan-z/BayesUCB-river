import numpy as np
class Evaluation:
  
    def __init__(self, env, pol, nbRepetitions, horizon, tsav=[]):
        if len(tsav)>0:
            self.tsav = tsav
        else:
            self.tsav = np.arange(horizon)
        self.env = env
        self.pol = pol
        self.nbRepetitions = nbRepetitions
        self.horizon = horizon
        self.n_arms = env.n_arms
        self.nbPulls = np.zeros((self.nbRepetitions, self.n_arms))
        self.cum_reward = np.zeros((self.nbRepetitions, len(tsav)))
                 
        for k in range(nbRepetitions):
            if nbRepetitions < 10 or k % (nbRepetitions/10)==0:
                print(k)
            result = env.play(pol, horizon)
            self.nbPulls[k,:] = result.getNbPulls()
            self.cum_reward[k,:] = np.cumsum(result.rewards)[tsav]
     
    def meanReward(self):
        return sum(self.cum_reward[:,-1])/len(self.cum_reward[:,-1])

    def meanNbDraws(self):
        return np.mean(self.nbPulls ,0) 

    def meanRegret(self):
        print(f'max.expectation:{max([arm.expectation for arm in self.env.arms])}')
        return (1+self.tsav)*max([arm.expectation for arm in self.env.arms]) - np.mean(self.cum_reward, 0)
