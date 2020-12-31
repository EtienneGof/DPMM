# DPMM

This code performs Dirichlet Process Inference based on Monte Carlo Markov Chain sampling.

### What's inside 

Two algorithms are implemented, corresponding to algorithm 2 (Gibbs Sampler) and Algorithm 3 (Collapsed Gibbs Sampler) from [1].

The user can either choose to provide the Dirichlet Process concentration parameter value or a Gamma prior for this parameter. In the latter case, alpha's value is updated following [2].

### Prerequisites And Installation

See src/pom.xml file for Scala dependencies.
The R script requires the "animation" and "mixtools" packages. 

Two files are provided to build and run the scala sources: build.sh, run.sh

[1] Neal, R. M. (2000). Markov chain sampling methods for Dirichlet process mixture models. Journal of computational and graphical statistics, 9(2), 249-265.

[2] West, M. (1992). Hyperparameter estimation in Dirichlet process mixture models. ISDS Discussion Paper# 92-A03: Duke University.
