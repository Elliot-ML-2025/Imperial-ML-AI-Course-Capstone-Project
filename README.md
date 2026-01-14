# Imperial-ML-AI-Course-Capstone-Project
Capstone project on black-box ML models - building testing and interpreting complex algorithms
This project is a capstone exploration of Black-Box Optimization (BBO), focused on systematically optimising 8 different expensive, unknown functions using limited data. The primary goal is to identify inputs that maximise the functions. We are provided initial data and are only allowed to query each function 13 times in total. This simulates real-world optimisation problems where data is limited and obtaining new data (or equivalently more information about a black-box function) is costly.

**Inputs and Outputs:**

Inputs - The model receives query points from multi-dimensional spaces with dimensions ranging from 2D to 8D. Each input is from the closed interval $[0,1]$

Outputs - Each function returns a singular real number for a given input.  

**Challenge objectives:**

The goal is to maximise each function. We are given 14 chances to query the function in order to update our model. One round of querying could look as follows:

For a 4D function -

Query point = (0.122345,0.673839,0.949390,0.349599)

Output response - 1.30045505

Since there are limited queries, this means it is important to emphasise careful selection balancing exploration and exploitation. Other constraints include a lack of information about the structure of the function (such as smoothness of the function and how different features interact with one another). 

**Technical Approach**

I use Gaussian Process (GP) regression with an Upper Confidence Bound (UCB) acquisition function to perform black-box optimization. The GP treats the unknown objective as a distribution over functions, providing both a posterior mean (expected performance) and a posterior standard deviation (model uncertainty) at any candidate point. The UCB acquisition function selects the next evaluation by maximizing

$$\text{UCB}(x) := \mu(x)+\beta \sigma(x) $$

explicitly trading off exploitation (high predicted value) and exploration (high uncertainty).

The primary design choice is the exploration parameter $β$, which controls this tradeoff. Large $β$ values favor exploration by prioritizing uncertain regions, while smaller values emphasize exploitation of regions with high predicted performance. I use a linearly decaying $β$ schedule, starting with strong exploration and gradually shifting toward exploitation over the 13 optimization iterations. This reflects the intuition that broad exploration is most valuable early, while later iterations should refine the search around promising regions.

Importantly, β is treated as dimension-dependent. In low-dimensional settings (e.g., 2D with 10 initial samples), the input space is relatively well covered, making aggressive exploration both feasible and beneficial. In higher-dimensional settings (e.g., 8D with 40 samples), the space is sparsely sampled—effectively only a few points per dimension—so exhaustive exploration is unrealistic. In these cases, starting with a lower β acknowledges the limits of coverage in high dimensions and biases the search toward exploitation sooner, improving practical efficiency.

In the first 3 queries, I have focused on exploration, which involves using a higher value for beta. In future rounds, I plan to gradually make beta smaller to focus on exploitation. In higher dimensions, it will be necessary to decrease this beta more aggressively as points are more sparse in higher dimensions. I've decided to place a high emphasis on exploration. This is because the data we have initially been provided is very sparse and so there will likely be regions of high uncertainty, which means high potential for improvement of the optimal value for the function.

After more data has been gathered, I will start to use more ML techniques to support the use of a GP. This will entail using correlation analysis to evaluate the effect of features on the function, which potentially help with dimension reduction. It will also involve classification models such as SVMs to determine promising regions. 
