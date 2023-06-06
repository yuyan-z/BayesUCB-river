This is the "Data Stream Mining" project of Yuyan Zhao and Bérénice Jaulmes.

It implements the BayesUCB algorithm<sup>[1]</sup> for the multi-armed bandit problem with the [River](https://github.com/online-ml/river/tree/main) library. 
We chose to use a Bernouilli distribution when computing the posterior distribution and the upper confidence bound (UCB) on the expected reward of each arm. The arm with the highest UCB is then pulled. And the posterior’s distribution is updated after each pull.

Our current code has been accepted by Max via pulling the request to the River. The previous versions are also available at 



## Reference
[1] Kaufmann, Emilie, Olivier Cappé, and Aurélien Garivier. "On Bayesian upper confidence bounds for bandit problems." Artificial intelligence and statistics. PMLR, 2012.

