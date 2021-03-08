package DPMM

import Common.Tools._
import breeze.linalg.{DenseVector, inv}
import breeze.numerics.log
import breeze.stats.distributions.{Beta, Gamma, MultivariateGaussian}

import scala.annotation.tailrec
import scala.collection.mutable.ListBuffer

// Algorithm 3 of [1], for multivariate Gaussian (based on [2])

// [1] Neal, R. M. (2000). Markov chain sampling methods for Dirichlet process mixture models. Journal of computational and graphical statistics, 9(2), 249-265.
// [2] Murphy, K. P. (2007). Conjugate Bayesian analysis of the Gaussian distribution. def, 1(2σ2), 16.

class CollapsedGibbsSampler(val Data: List[DenseVector[Double]],
                            var prior: NormalInverseWishart = new NormalInverseWishart(),
                            var alpha: Option[Double] = None,
                            var alphaPrior: Option[Gamma] = None,
                            var initByUserMembership: Option[List[Int]] = None) extends Serializable {

  val n: Int = Data.length

  // p(x_i | z_i = new cluster, prior) <=> \int_{\theta} F(y_i, \theta) dG_0(\theta)) in [1] eq. (3.7)
  val priorPredictive: List[Double] = Data.map(prior.predictive)

  var memberships: List[Int] = initByUserMembership match {
    case Some(m) =>
      require(m.length == Data.length)
      m
    case None => List.fill(Data.length)(0)
  }

  // n_k
  var countCluster: ListBuffer[Int] = partitionToOrderedCount(memberships).to[ListBuffer]

  // H_{-i}: prior updated
  var NIWParams: ListBuffer[NormalInverseWishart] = (Data zip memberships).groupBy(_._2).values.map(e => {
    val dataPerCluster = e.map(_._1)
    val clusterIdx = e.head._2
    (clusterIdx, prior.update(dataPerCluster))
  }).toList.sortBy(_._1).map(_._2).to[ListBuffer]

  require(!(alpha.isEmpty & alphaPrior.isEmpty),"Either alpha or alphaPrior must be provided: please provide one of the two parameters.")
  require(!(alpha.isDefined & alphaPrior.isDefined), "Providing both alpha or alphaPrior is not supported: remove one of the two parameters.")

  var updateAlphaFlag: Boolean = alphaPrior.isDefined

  var actualAlphaPrior: Gamma = alphaPrior match {
    case Some(g) => g
    case None => new Gamma(1D,1D)
  }

  var actualAlpha: Double = alpha match {
    case Some(a) =>
      require(a > 0, s"Alpha parameter is optional and should be > 0 if provided, but got $a")
      a
    case None => actualAlphaPrior.mean
  }

  def updateAlpha() {
    val shape = actualAlphaPrior.shape
    val rate =  1D/actualAlphaPrior.scale
    val nCluster = countCluster.length

    val log_x = log(new Beta(actualAlpha + 1, n).draw())
    val pi1 = shape + nCluster + 1
    val pi2 = n * (rate - log_x)
    val pi = pi1/(pi1+pi2)
    val newScale = 1/(rate - log_x)

    actualAlpha = if(sample(List(pi, 1-pi)) == 0){
      Gamma(shape = shape + nCluster, newScale).draw()
    } else {
      Gamma(shape = shape + nCluster - 1, newScale).draw()
    }
  }

  // p(x_i | z_i = existing cluster, x_{-i} in existing cluster) <=> \int_{\theta} F(y_i, \theta) dH_{-i}(\theta)) in [1] eq. (3.7)
  // Important:
  // b) The NIWParams are already updated (they are modified on the fly at every membership update).
  // a) Denominator (n-1+\alpha) is omitted because the probabilities are eventually normalized.
  
  def computeClusterMembershipProbabilities(idx: Int,
                                            verbose: Boolean=false): List[Double] = {
    NIWParams.indices.map(clusterIdx => {
      NIWParams(clusterIdx).predictive(Data(idx)) + log(countCluster(clusterIdx))
    }) .toList
  }

  def drawMembership(idx: Int,
                     verbose : Boolean = false): Int = {

    val probExistingClusterMembership = computeClusterMembershipProbabilities(idx, verbose)
    val posteriorPredictiveXi = priorPredictive(idx)
    val probs = probExistingClusterMembership :+ (posteriorPredictiveXi + log(actualAlpha))
    val normalizedProbs = normalizeProbability(probs)
    sample(normalizedProbs)
  }

  private def removeElementFromCluster(idx: Int): Unit = {
    val currentMembership =  memberships(idx)
    if (countCluster(currentMembership) == 1) {
      countCluster.remove(currentMembership)
      NIWParams.remove(currentMembership)
      memberships = memberships.map(c => { if( c > currentMembership ){ c - 1 } else c })
    } else {
      countCluster.update(currentMembership, countCluster.apply(currentMembership) - 1)
      val updatedNIWParams = NIWParams(currentMembership).removeObservations(List(Data(idx)))
      NIWParams.update(currentMembership, updatedNIWParams)
    }
  }

  private def addElementToCluster(idx: Int, newMembership: Int): Unit = {
    if (newMembership == countCluster.length) { // Creation of a new cluster
      countCluster = countCluster ++ ListBuffer(1)
      val newNIWparam = this.prior.update(List(Data(idx)))
      NIWParams = NIWParams ++ ListBuffer(newNIWparam)
    } else {
      countCluster.update(newMembership, countCluster.apply(newMembership) + 1)
      val updatedNIWParams = NIWParams(newMembership).update(List(Data(idx)))
      NIWParams.update(newMembership, updatedNIWParams)
    }
  }

  def updateMembership(verbose: Boolean = false): Unit = {
    for (i <- 0 until n) {
      removeElementFromCluster(i)
      val newMembership = drawMembership(i)
      memberships = memberships.updated(i, newMembership)
      addElementToCluster(i, newMembership)
    }
  }

  def run(maxIter: Int = 10,
          maxIterBurnin: Int = 10,
          verbose: Boolean = false): (List[List[Int]], List[List[MultivariateGaussian]], List[Double]) = {

    var membershipEveryIteration = List(memberships)
    var componentEveryIteration = List(prior.posteriorSample(Data, memberships).map(e => MultivariateGaussian(e.mean, inv(e.covariance))))
    var likelihoodEveryIteration = List(prior.likelihood(actualAlpha, Data, memberships, countCluster.toList, componentEveryIteration.head))

    @tailrec def go(iter: Int): Unit = {
      println("\n>>>>>> Iteration: " + iter.toString)
      println("\u03B1 = " + actualAlpha.toString)
      println("Cluster sizes: "+ countCluster.mkString(" "))

      if (iter > (maxIter + maxIterBurnin)) {
      } else {
        updateMembership()

        if(updateAlphaFlag){updateAlpha()}

        val components = prior.posteriorSample(Data, memberships)
        val likelihood = prior.likelihood(actualAlpha,
          Data,
          memberships,
          countCluster.toList,
          components)
        membershipEveryIteration = membershipEveryIteration :+ memberships
        componentEveryIteration = componentEveryIteration :+ components
        likelihoodEveryIteration = likelihoodEveryIteration :+ likelihood
        go(iter + 1)
      }
    }

    go(1)

    (membershipEveryIteration, componentEveryIteration, likelihoodEveryIteration)  }
}

