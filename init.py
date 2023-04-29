import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Arm import Arm
from BayesUCB import BayesUCB
from Environment import Environment
from Result import Result
from Evaluation import Evaluation
from Beta import Beta
dataset = pd.read_csv('./ucb-data.csv')
arms = [Arm(dataset, i) for i in range(0,8)]
env = Environment(arms)

scenario = 10
nbRep = 10
horizon = 1800

K = env.n_arms

policy = BayesUCB(K)


# tsav = int_(linspace(100,horizon-1,200))
tsav = np.linspace(100,horizon-1,200).round().astype(int)

# print(f"tsav:{tsav}")

k=0

ev = Evaluation(env, policy, nbRep, horizon, tsav)
print(f'mean reward:{ev.meanReward()}')
# print(f'meanNbDraws():{ev.meanNbDraws()}')
mean_regret = ev.meanRegret()
print(f'mean regret:{mean_regret[-1]}')

plt.semilogx(1+tsav, mean_regret)
plt.xlabel('Time')
plt.ylabel('Mean Regret')

plt.show()