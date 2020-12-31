# DPMM

This code performs Dirichlet Process Inference based on Monte Carlo Markov Chain sampling.

### What's inside 

Two algorithms are implemented, corresponding to algorithm 2 (Gibbs Sampler) and Algorithm 3 (Collapsed Gibbs Sampler) from [1].

The user can either choose to provide the Dirichlet Process concentration parameter value or a Gamma prior for this parameter. In the latter case, alpha's value is updated following [2].

### Quick Setup

The script build.sh is provided to build the scala sources. 

See src/pom.xml file for Scala dependencies.

The file run.sh launches a small 2D example with 4 clusters and 150 points each. The results are saved in the "results" directory.

The R script requires the "animation" and "mixtools" packages.

### Results Visualization

An R script if provided to visualize the evolution of the clusters memberships and model log-likelihood, in the 2D case. The script reads the files written in the "results" directory.  Before launching the R script, please make sure to change the result directory path with its true value. 

<p align="center">
  <img src="https://github.com/EtienneGof/DPMM/blob/main/visualization.gif" />
</p>

[1] Neal, R. M. (2000). Markov chain sampling methods for Dirichlet process mixture models. Journal of computational and graphical statistics, 9(2), 249-265.

[2] West, M. (1992). Hyperparameter estimation in Dirichlet process mixture models. ISDS Discussion Paper# 92-A03: Duke University.
