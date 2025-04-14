This is the "Data Stream Mining" project of Yuyan Zhao and Bérénice Jaulmes.

## Documentation
- bayes_ucb.py

It implements the BayesUCB algorithm<sup>[1]</sup> for the multi-armed bandit problem with the [River](https://github.com/online-ml/river/tree/main) library. 
We chose to use a Beta distribution to compute the posterior distribution, and use the p-th quantile as the upper confidence bound (UCB) for each arm. The arm with the highest UCB is then pulled. And the posterior distribution for the pulled arm is updated.

- test.py

It evaluates the performance of the BayesUCB policy defined, and compares it with the existing bandit algorithms in River. The result is shown in the figure below:
![alt text](https://github.com/ormarv/Project/blob/eda48be1f579ff57e20a9d4ed57701749f192000/result.png)



- /previous_version

Our code has been accepted by River! You can check out the pull request [Pull requests](https://github.com/online-ml/river/pull/1237). 

## Reference
[1] Kaufmann, Emilie, Olivier Cappé, and Aurélien Garivier. "On Bayesian upper confidence bounds for bandit problems." Artificial intelligence and statistics. PMLR, 2012.

