package DPMM

import Common.Tools._
import breeze.linalg.DenseVector
import breeze.numerics.log
import breeze.stats.distributions.{Beta, Gamma, MultivariateGaussian}

import scala.annotation.tailrec
import scala.collection.mutable.ListBuffer

// Algorithm 2 of [1], for multivariate Gaussian (based on [2])

// [1] Neal, R. M. (2000). Markov chain sampling methods for Dirichlet process mixture models. Journal of computational and graphical statistics, 9(2), 249-265.
// [2] Murphy, K. P. (2007). Conjugate Bayesian analysis of the Gaussian distribution. def, 1(2σ2), 16.

class GibbsSampler(val Data: List[DenseVector[Double]],
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

  var components: ListBuffer[MultivariateGaussian] = prior.posteriorSample(Data, memberships).to[ListBuffer]
  
  // n_k
  var countCluster: ListBuffer[Int] = partitionToOrderedCount(memberships).to[ListBuffer]

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

   // p(x_i | z_i = k, \theta_k) <=> F(y_i, \theta_c) in [1] eq. (3.5) and (3.6)
  // Important:
  // a) Basically, based on the likelihood (component.logPfg) and p(z) (n_k / (n - 1  + \alpha)
  // b) Denominator (n-1+\alpha) is omitted because the probabilities are eventually normalized.
  
  def computeClusterMembershipProbabilities(idx: Int,
                                            verbose: Boolean=false): List[Double] = {
    countCluster.indices.map(k => {
      (k, log(countCluster(k).toDouble) + components(k).logPdf(Data(idx)))
    }).toList.sortBy(_._1).map(_._2)
  }

  def drawMembership(i: Int): (Int, MultivariateGaussian) = {
    val probMembership = computeClusterMembershipProbabilities(i)
    val probMembershipNewCluster = log(actualAlpha) + priorPredictive(i)
    val normalizedProbs = normalizeProbability(probMembership :+ probMembershipNewCluster)
    val newMembership = sample(normalizedProbs)
    val newComponent = if(newMembership >= components.length){
      prior.update(List(Data(i))).sample()
    } else components(newMembership)
    (newMembership, newComponent)
  }

  private def removeElementFromItsCluster( currentMembership: Int): Unit = {
    if (countCluster(currentMembership) == 1) {
      countCluster.remove(currentMembership)
      components.remove(currentMembership)
      memberships = memberships.map(c => { if( c > currentMembership ){ c - 1 } else c })
    } else {
      countCluster.update(currentMembership, countCluster.apply(currentMembership) - 1)
    }
  }

  private def addElementToCluster(newMembership: Int, newComponentDensity: MultivariateGaussian): Unit = {
    if (newMembership == countCluster.length) { // Creation of a new cluster
      countCluster = countCluster ++ ListBuffer(1)
      components = components ++ ListBuffer(newComponentDensity)
    } else {
      countCluster.update(newMembership, countCluster.apply(newMembership) + 1)
    }
  }

  def updateMembership(verbose: Boolean = false): Unit = {
    for (i <- 0 until n) {
      val currentMembership = memberships(i)
      removeElementFromItsCluster(currentMembership)
      val (newMembership, newComponentDensity) = drawMembership(i)
      memberships = memberships.updated(i, newMembership)
      addElementToCluster(newMembership, newComponentDensity)
    }
  }

  def run(maxIter: Int = 10,
          verbose: Boolean = false): (List[List[Int]], List[List[MultivariateGaussian]], List[Double]) = {

    var membershipEveryIteration = List(memberships)
    var componentEveryIteration = List(components.toList)
    var likelihoodEveryIteration = List(prior.likelihood(actualAlpha, Data, memberships, countCluster.toList, components.toList))

    @tailrec def go(iter: Int): Unit = {

      println("\n>>>>>> Iteration: " + iter.toString)
      println("\u03B1 = " + actualAlpha.toString)
      println("Cluster sizes: "+ countCluster.mkString(" "))

      if (iter > maxIter) {

      } else {
        updateMembership()
        components = prior.posteriorSample(Data, memberships).to[ListBuffer]
        if(updateAlphaFlag){updateAlpha()}
        val likelihood = prior.likelihood(actualAlpha, Data, memberships, countCluster.toList, components.toList)
        membershipEveryIteration = membershipEveryIteration :+ memberships
        componentEveryIteration = componentEveryIteration :+ components.toList
        likelihoodEveryIteration = likelihoodEveryIteration :+ likelihood
        go(iter + 1)
      }
    }

    go(1)

    (membershipEveryIteration, componentEveryIteration, likelihoodEveryIteration)
  }
}
