This is the "Data Stream Mining" project of Yuyan Zhao and Bérénice Jaulmes.

## Documentation
- bayes_ucb.py

It implements the BayesUCB algorithm<sup>[1]</sup> for the multi-armed bandit problem with the [River](https://github.com/online-ml/river/tree/main) library. 
We chose to use a Bernouilli distribution when computing the posterior distribution and the upper confidence bound (UCB) on the expected reward of each arm. The arm with the highest UCB is then pulled. And the posterior’s distribution is updated after each pull.

- test.py

It evaluates the performance of the BayesUCB policy defined, and compares the existing bandit algorithms in the River. The result is shown in the figure below:
![Image text]([https://raw.github.com/yourName/repositpry/master/yourprojectName/img-folder/test.jpg](https://github.com/ormarv/Project/blob/1f832165dbd16a80c890fb8c3b2437ff1e8d6abb/result.png))

- /previous_version

Our current code has been accepted by Max via [Pull requests](https://github.com/yuyan2000/river/blob/main/river/bandit/bayes_ucb.py) to the River. The previous versions are also available at /previous_version





## Reference
[1] Kaufmann, Emilie, Olivier Cappé, and Aurélien Garivier. "On Bayesian upper confidence bounds for bandit problems." Artificial intelligence and statistics. PMLR, 2012.

