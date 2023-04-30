from random import betavariate
from scipy.special import btdtri

class Beta:
    """
    This class computes the posterior of Bernoulli (/Beta) distribution.

    Parameters :
    a,b : int. Shape parameters of the Beta distribution 
    """

    def __init__(self, a=1, b=1):
        self.a = a
        self.b = b

    def reset(self, a=0, b=0):
        '''
        Reset the posterior to a "default" configuration.
        Parameters:
        a,b : int. Shape parameters of the Beta distribution.
        '''
        if a == 0:
            a = self.a
        if b == 0:
            b = self.b
        self.N = [a, b]

    def update(self, obs):
        '''
        Update the parameters of the distribution.
        Parameters:
        obs : float. Index of the parameter to be updated.
        '''
        self.N[int(obs)] += 1

    def sample(self):
        '''
        Returns the parameters of the distribution
        '''
        return betavariate(self.N[1], self.N[0])

    def quantile(self, p):
        '''
        Returns the p-th quantile of the beta distribution.
        Parameters :
        p : float. Cumulative probability in [0,1]
        '''
        return btdtri(self.N[1], self.N[0], p)
