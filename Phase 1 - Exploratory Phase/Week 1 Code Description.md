**Week 1 Code**

These scripts implement the first iteration of a Bayesian optimisation loop for each function using a Gaussian process surrogate model and an Upper Confidence Bound (UCB) acquisition strategy.

For each function, we were provide with a handful of input–output observations. These are used to fit a Gaussian process regression model with a fixed RBF kernel. The surrogate model is then evaluated across a dense grid covering the search domain to obtain predictions of both the mean response and associated uncertainty.

An Upper Confidence Bound acquisition function is computed to balance exploration and exploitation by prioritising points with either high predicted values or high predictive uncertainty. The next evaluation point is selected as the location that maximises this acquisition function.

This forms part of the iterative Bayesian optimisation framework used throughout the project, where surrogate models are progressively refined as additional observations are incorporated across successive weeks. 
Please note that I did not use the next suggested point given for function 8 and instead used a randomly selected point. This was due to my method of initialising candidate points not scaling well to 8 dimensions. This problem was addresed in subsequent iterations by using latin hypercube sampling.
