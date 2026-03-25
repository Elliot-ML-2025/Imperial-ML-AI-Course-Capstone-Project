**Evolution of the Optimization Strategy (Week 7 → Week 10)**

From Week 7 to Week 10, the core Random Forest–Gaussian Process hybrid model remained largely unchanged, as it continued to deliver consistent and reliable improvements across most functions. Each week, new data points were added to the model iteratively, allowing the hybrid approach to refine its suggestions without major structural changes.

However, for Function 1, the strategy was abandoned after a few iterations. This function featured output values near zero across most of the domain, with a sharp spike at (0.5, 0.5). The Random Forest–GP model struggled to explore this region because:

* The RF’s predictions were heavily influenced by the many near-zero points, making the spike appear as an outlier.
* The GP, when trained on the RF-selected candidates, was biased towards regions with moderate predicted outputs, avoiding the sparse high-value spike.

To overcome this, I switched to a simple Gaussian Process model with an Expected Improvement (EI) acquisition function. EI encourages exploration in regions where the predicted improvement over the current best is high, which in this case helped the model actively probe the sharp spike at (0.5, 0.5). This adjustment leveraged the GP’s uncertainty modeling more directly and prioritized regions that could yield significant gains despite being underrepresented in the initial data.
