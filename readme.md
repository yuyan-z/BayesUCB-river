This is the "Data Stream Mining" project of Yuyan Zhao and Bérénice Jaulmes.

## Documentation
- bayes_ucb.py

It implements the BayesUCB algorithm<sup>[1]</sup> for the multi-armed bandit problem with the [River](https://github.com/online-ml/river/tree/main) library. 
We chose to use a Beta distribution when computing the posterior distribution and the upper confidence bound (UCB) on the expected reward of each arm. The arm with the highest UCB is then pulled. And the posterior’s distribution is updated after each pull.

- test.py

It evaluates the performance of the BayesUCB policy defined, and compares it with the existing bandit algorithms in River. The result is shown in the figure below:
![alt text](https://github.com/ormarv/Project/blob/74bd810b61846435171798f17c88567e06b927ee/results/sum_reward.png))
![alt text](https://github.com/ormarv/Project/blob/74bd810b61846435171798f17c88567e06b927ee/results/mean_reward.png))



- /previous_version

Our current code has been accepted by Max via [Pull requests](https://github.com/yuyan2000/river/blob/main/river/bandit/bayes_ucb.py) to the River. The previous versions are also available at /previous_version





## Reference
[1] Kaufmann, Emilie, Olivier Cappé, and Aurélien Garivier. "On Bayesian upper confidence bounds for bandit problems." Artificial intelligence and statistics. PMLR, 2012.

