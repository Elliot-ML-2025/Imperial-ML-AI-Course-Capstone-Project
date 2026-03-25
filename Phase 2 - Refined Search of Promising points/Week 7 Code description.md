**Random Forest–Gaussian Process Hybrid for Function Optimization**

This workflow implements a hybrid optimization approach combining Random Forest (RF) regression with a Gaussian Process (GP) to efficiently identify promising points in a high-dimensional function space.

**Data Preparation**

* Loads, cleans, and augments existing function evaluation data.
Performs exploratory analysis, including Pearson correlations and scatter plots for each input against the output.

**Random Forest Regression with Light Hyperparameter Tuning**

A RandomForestRegressor models the input–output relationship.
RandomizedSearchCV performs light hyperparameter tuning over a small, targeted set of options:
* n_estimators = [200, 300, 500] → ensures sufficient trees for stable predictions while keeping computational cost reasonable.
* max_depth = [None, 10, 20] → balances model flexibility with overfitting; shallow trees prevent memorization of noise, while deeper trees capture complex interactions.
* min_samples_leaf = [1, 2, 5] → prevents overfitting to very small subsets; slightly larger leaf sizes smooth predictions without losing key patterns.
* max_features = [0.5, 0.7, 1.0] → controls feature subset selection at splits; lower fractions increase tree diversity, higher fractions allow better capture of important features.

Light tuning is appropriate because the RF serves as a screening tool to identify promising regions rather than providing a final predictive model.

**Candidate Point Generation**

Generates a dense grid of candidate points using Latin Hypercube Sampling (LHS).
Uses the tuned RF to predict outputs across the grid and selects the top-performing candidates. For the lower dimensional functions (1-4 dimensions), the top 200 points would be considered, wheras for the higher dimensional functions, i would consider the top 3-5 hundred points.

**Gaussian Process Refinement**

* Fits a Gaussian Process with a Matern kernel to the selected candidates.
* Computes the Upper Confidence Bound (UCB) for balancing exploration and exploitation.
* A beta of 1.96 was used for the UCB function, corresponding to a 95% confidence interval. This balances exploration and exploitation by considering both the predicted mean and uncertainty, helping the algorithm target promising regions while still allowing for discovery.
* Suggests the next query point based on the highest UCB, for both RF-selected candidates and the entire domain.

This hybrid approach leverages the efficiency of Random Forests for large-scale screening and the uncertainty modeling of Gaussian Processes for principled exploration, enabling efficient sequential optimization in high-dimensional spaces.
