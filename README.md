# Imperial-ML-AI-Course-Capstone-Project
This project is a capstone exploration of Black-Box Optimization (BBO), focused on systematically optimising 8 different expensive, unknown functions using limited data. The primary goal is to identify inputs that maximise the functions. We are provided initial data and are only allowed to query each function 13 times in total. This simulates real-world optimisation problems where data is limited and obtaining new data (or equivalently more information about a black-box function) is costly.

## Inputs and Outputs

Inputs - The model receives query points from multi-dimensional spaces with dimensions ranging from 2D to 8D. Each input is from the  interval $[0,1)$

Outputs - Each function returns a singular real number for a given input.  

## Challenge objectives

The goal is to maximise each function. We are given 13 chances to query the function in order to update our model. An example of one query could look as follows:

For a 4D function -

Queried point = (0.122345,0.673839,0.949390,0.349599)

Output response - 1.30045505

Since there are limited queries, this means it is important to emphasise careful selection balancing exploration and exploitation. Other constraints include a lack of information about the structure of the function (such as smoothness of the function and how different features interact with one another). 

## Technical Approach

I use Gaussian Process (GP) regression as the foundation for a Bayesian Optimization (BO) loop. In this framework, the unknown function is treated probabilistically: a GP defines a distribution over functions, such that for any finite set of input points, the function values are jointly Gaussian. Concretely, a GP provides a posterior mean $\mu(x)$, which estimates the expected function value at a point $x$, and a posterior standard deviation $\sigma(x)$, which quantifies uncertainty about the function at that point. This allows the model to not only predict function values but also reason about where it is uncertain, which is critical when data is limited.
In Bayesian Optimization, acquisition functions use the GP’s predictions to select the next point to query. The acquisition function balances exploration (sampling points with high uncertainty to gain information) and exploitation (sampling points with high expected values to maximize the objective). By iteratively updating the GP with new data and selecting points that maximise the acquisition function, Bayesian Optimization efficiently searches the input space for the global maximum.

The Upper Confidence Bound (UCB) acquisition function is defined at $x$ by:

$$\text{UCB}(x)=\mu(x)+\beta \sigma(x),$$

where $\beta > 0$ is a parameter controlling the trade-off between exploration and exploitation. A higher $\beta$ favors points with greater uncertainty (exploration), while a lower $\beta$ favors points with higher predicted values (exploitation).

This approach is theoretically grounded in Srinivas et al. (2009, arXiv:0912.3995), which shows that using the GP-UCB acquisition function guarantees sublinear cumulative regret. Regret measures the difference between the function value at the global maximum and the value at points actually sampled. Sublinear regret means that, as more queries are made, the average difference between the best possible outcome and the sampled outcomes decreases over time, implying that the algorithm becomes increasingly effective at identifying the maximum.

In layman’s terms, this means that the GP-UCB strategy ensures we are efficiently learning about the function, increasingly focusing on regions that matter while still occasionally exploring uncertain areas. Mathematically, it gives a performance guarantee: the optimization is not just heuristic, but backed by a provable bound on how “far off” the algorithm can be from the true maximum as queries accumulate.

## Optimization Phases

**Phase 1 - Exploratory Phase:**

This phase spans iteration 1-6. The purpose of this phase is to gain an understanding of the global structure for each function. Using a larger $\beta$ value in the UCB acquisition functions, the algorithm prioritizes uncertain regions of the input space to identify promising areas that may contain the global maximum. This is especially important given the sparse initial data and high dimensionality of some functions.

**Phase 2 - Refined Search of Promising Points:**

This phase spans iterations 7-10. After initial exploration, the strategy shifts slightly toward exploitation. The $\beta$ parameter is reduced to focus on regions predicted to have high function values, increasing the likelihood of improving the observed maximum. In this phase, I incorporate  random forest models to support the GP by identifying promising subspaces.

**Phase 3 - Exploitative Phase:**

This phase spans iterations 11-13. In later iterations, the model concentrates on fine-tuning around identified maxima. The GP-UCB  guides query selection, but with a much stronger emphasis on exploitation concentrating purely on regions containing already identified maxima.

Each phase reflects a deliberate trade-off between learning about the function broadly and efficiently improving the maximum observed value, aligning with both practical and theoretical insights from Bayesian Optimisation. 

More detailed descriptions of the algorithms and methods used in each phase can be found in the corresponding folder named after that phase.

## Data and Code
Data for each submission can be found in the Data folder. This includes both the initial datasets and the additional observations collected after each weekly submission. An Excel spreadsheet is also provided containing the full dataset alongside a week-by-week analysis of improvements for each function.

The code included corresponds to the first and final iterations within each optimisation phase. Between weeks, most updates involved incorporating newly acquired data rather than making major structural changes to the algorithms. Including these versions highlights the overall evolution of the approach across the project. 

## Results
| Function | Dimension | Initial Best | Final Best |
|----------|-----------|------------------|------------|
| Function 1 | 2D | 7.71e-16 | 1.27e-5| 
| Function 2 | 2D | 0.611| 0.655 | 
| Function 3 | 3D | -0.034 | -0.001 | 
| Function 4 | 4D | -4.02 | 0.68 | 
| Function 5 | 4D | 1088 |  7055| 
| Function 6 | 5D | -0.71 | -0.050| 
| Function 7 | 6D |1.36 | 2.75 | 
| Function 8 | 8D | 9.598 |9.988  |

**Note:**
  * Functions 3 and 6 are capped above at 0
  * Function 4 achieved sign flip in submission 8 - all previous outputs were negative
## Repository structure

```
Imperial-ML-AI-Course-Capstone-Project/
├── Data
│   ├── initial_data-3
│   ├── Raw Data
│   ├── Black Box Optimization Output Tracking.xlsx
│   └── Output Tracking Spreadsheet Description.md
├── Phase 1 - Exploratory Phase
│   ├── Capstone Project Week 1 Function 1.ipynb
│   ├── ...
│   ├── Capstone Project Week 1 Function 8.ipynb
│   ├── Week 1 Code Description.md
│   ├── Week 6 - Function 1.ipynb
│   ├── ...
│   ├── Week 6 - Function 8.ipynb
│   └── Week 6 Code Description.md
├── Phase 2 - Refined Search of Promising points
│   ├── Week 10 - Function 1.ipynb
│   ├── ...
│   ├── Week 10 - Function 8.ipynb
│   ├── Week 10 Code Description.md
│   ├── Week 7 - Function 1.ipynb
│   ├── ...
│   ├── Week 7 - Function 8.ipynb
│   └── Week 7 Code description.md
├── Phase 3 - Exploitative Phase
│   ├── Week 11-13 - Function 1.ipynb
│   ├── ...
│   ├── Week 11-13 - Function 8.ipynb
│   └── Week 11-13 code description.md
├── Data Sheet
├── Model Card
└── README.md
```

