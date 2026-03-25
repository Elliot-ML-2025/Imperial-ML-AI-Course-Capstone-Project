# Week 6 - Final iteration of exploratory phase

These scripts implement the final iteration in the Exploratory Phase of my Bayesian Optimization loop for each function. At this stage, there was an extra 5 input-output data points from queries in previous week.

Between Week 1 and Week 6, the Bayesian optimisation strategy was refined to improve both the flexibility of the surrogate model and the effectiveness of candidate point selection.

One important change was replacing the RBF kernel with a Matérn kernel in the Gaussian process surrogate model. While the RBF kernel assumes a high degree of smoothness in the underlying objective functions, the Matérn kernel allows more flexible smoothness assumptions and therefore provides a more robust modelling choice across different benchmark functions with potentially varying local structure.

A second improvement was the introduction of Latin hypercube sampling to initialise candidate query points. Rather than evaluating the acquisition function over a simple uniform grid, this approach provides a more space-filling and representative coverage of the search domain, improving the reliability of candidate selection while maintaining computational efficiency.

Finally, the exploration–exploitation strategy evolved immediately after reflecting on the results from Week 1. Initially, I planned to adopt a more exploitative approach for the higher-dimensional functions, as I expected it would be difficult to cover the search space effectively. However, I later recognised that these search spaces remained largely unexplored, meaning the potential upside in uncertain regions could still be significant.

As a result, in subsequent weeks I shifted towards a more exploration-focused strategy. To support this adjustment, I compared the candidate point selected by the UCB acquisition function with the point that would have been chosen by an Expected Improvement acquisition function with relatively low exploration. I then increased the UCB exploration parameter β incrementally (in steps of 0.5) until the selected candidate point reflected a sufficiently exploratory move. This approach helped maintain broader coverage of the search space and reduced the risk of premature convergence to local optima.

Together, these changes strengthened both the surrogate modelling assumptions and the acquisition strategy, resulting in a more stable and effective optimisation framework by the end of Week 6.
