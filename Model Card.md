# Model Card: Bayesian Optimization Loop

## Model Overview:
**Model Name:** Bayesian Optimization Loop

**Model type:** Sequential decision-making/black-box optimization

**Core method:** Surrogate modeling + acquisition optimization 

This model implements a Bayesian optimization framework to efficiently optimize an unknown, expensive-to-evaluate objective function.
It iteratively builds a probabilitstic surrogate model and selects new evaluation points using an acquisition function that balances exploration
and exploitation

## Intended Use

**Primary use cases:**
* Hyperparatmer tuning for machine learning models
* Optimization of expensive simulations or experiments
* Black-box function optimization with limited evaluation budget
* Engineering design and scientific experimentation

**Intended Users:**
* Data scientists
* ML engineers
* Researchers working with expensive objective functions

**Out-of-scope use:**
* Real-time decision systems requiring ultra-low latency
* Problems with extremly high-dimensional search spaces (>20 dimensions)
* Situations where objective evaluations are noisy but modeled appropriately

## Model Architecture

**Surrogate Model:**

* Primary model was a Gaussian Process (GP)
* At each point $x$ in the search space, this provides a mean prediction $\mu(x)$ and uncertainty estimate $\sigma(x).$
* Screening of promising points was used in later rounds to guide the GP. This included random forest regression to identify high potential points or trust regions centered around current maxima

**Acquisition Functions:**

Mostly used (with some exceptions) the Upper Confidence Bound (UCB) acquisition function. Given a point $x$ in the search space,

$$\text{UCB}(x)=\mu(x)+\beta \sigma(x),$$

where $\beta >0$ is a paramater to control the exploration-exploitation balance. In some rounds, I would use expected improvement to provide
a comparison or occassionaly to decide where to search next. The expected improvement $EI$ at $x$ is given by 

$$EI(x):=(\mu(x)-f(x^+)-\eta)\Phi(Z)+\sigma(x)\phi(Z),$$

where 

* $f(x^+)$ is the best observed value so far
* $\eta\ge0$ is an exploration parameter
* $\Phi(.)$ is the standard normal CDF
* $\phi(.)$ is the standard normal PDF
and

$$Z= \frac{\mu(x)-f(x^+)-\eta}{\sigma(x)}$$

**Optimization Loop**

1. Initizlize with a set of sampled points (with possible screening of points)
2. Fit surrogate model to observed data
3. Optimize acquisition function to select next point
4. Evaluate objective function
5. Update dataset
6. Repeat until budget exhausted

## Training Data/ Inputs

**Inputs**

- Search space definition (i.e. what is the domain of the functions)
- Initially sampled points
- Objective function evaluations

**Data assumptions**

Objective function is:

* Expensive to evaluate
* Possibly noisy (noise parameter typically set to 1e-10)
* Smooth or partially smooth (for GP effectivenes). Smoothness assumptions were typically set to 2.5 although for function 1 it was reduced to 1.5 in round 9

## Performance Characteristics

**Strengths**

* Sample efficient optimization
* Handles uncertainty explicitly
* Works well with small evaluation budgets
* Naturally balances exploration and exploitation

**Limitations**

* *Scalability issues:* GP computational cost scales cubically with number of points to evaluate. The number of points needed to cover the search space increases exponentially with dimension so  the model does not scale well to very high dimensions
* *Sensitive to kernel choice and hyperparameters:* There are many different hyperparameters that can be changed and it is hard to get a sense of which is the most important given limited data
* 
