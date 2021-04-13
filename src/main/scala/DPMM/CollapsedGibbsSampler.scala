package DPMM

import breeze.linalg.{DenseVector, inv}
import breeze.stats.distributions.{Gamma, MultivariateGaussian}

import scala.annotation.tailrec
import scala.collection.mutable.ListBuffer

class CollapsedGibbsSampler(Data: List[DenseVector[Double]],
                            prior: NormalInverseWishart = new NormalInverseWishart(),
                            alpha: Option[Double] = None,
                            alphaPrior: Option[Gamma] = None,
                            initByUserPartition: Option[List[Int]] = None
                           ) extends Default(Data, prior, alpha, alphaPrior, initByUserPartition) {

  var NIWParams: ListBuffer[NormalInverseWishart] = (Data zip partition).groupBy(_._2).values.map(e => {
    val dataPerCluster = e.map(_._1)
    val clusterIdx = e.head._2
    (clusterIdx, prior.update(dataPerCluster))
  }).toList.sortBy(_._1).map(_._2).to[ListBuffer]

  override def posteriorPredictive(observation: DenseVector[Double], cluster: Int): Double = {
    NIWParams(cluster).predictive(observation)
  }

  def removeElementFromNIW(idx: Int): Unit = {
    val currentPartition =  partition(idx)
    if (countCluster(currentPartition) == 1) {
      NIWParams.remove(currentPartition)
    } else {
      val updatedNIWParams = NIWParams(currentPartition).removeObservations(List(Data(idx)))
      NIWParams.update(currentPartition, updatedNIWParams)
    }
  }

  def addElementToNIW(idx: Int): Unit = {
    val newPartition = partition(idx)
    if (newPartition == countCluster.length) { // Creation of a new cluster
      val newNIWparam = this.prior.update(List(Data(idx)))
      NIWParams = NIWParams ++ ListBuffer(newNIWparam)
    } else {
      val updatedNIWParams = NIWParams(newPartition).update(List(Data(idx)))
      NIWParams.update(newPartition, updatedNIWParams)
    }
  }

  override def updatePartition(verbose: Boolean = false): Unit = {
    for (i <- 0 until n) {
      removeElementFromNIW(i)
      removeElementFromCluster(i)
      drawMembership(i)
      addElementToNIW(i)
      addElementToCluster(i)
    }
  }

  ///////////////////////

  def run(maxIter: Int = 10,
          maxIterBurnin: Int = 10,
          verbose: Boolean = false): (List[List[Int]], List[List[MultivariateGaussian]], List[Double]) = {

    var membershipEveryIteration = List(partition)
    var componentEveryIteration = List(prior.posteriorSample(Data, partition).map(e => MultivariateGaussian(e.mean, inv(e.covariance))))
    var likelihoodEveryIteration = List(prior.DPMMLikelihood(actualAlpha, Data, partition, countCluster.toList, componentEveryIteration.head))

    @tailrec def go(iter: Int): Unit = {

      if(verbose){
        println("\n>>>>>> Iteration: " + iter.toString)
        println("\u03B1 = " + actualAlpha.toString)
        println("Cluster sizes: "+ countCluster.mkString(" "))
      }

      if (iter <= (maxIter + maxIterBurnin)) {

        updatePartition()

        conditionalAlphaUpdate

        val components = prior.posteriorSample(Data, partition)
        val likelihood = prior.DPMMLikelihood(actualAlpha,
          Data,
          partition,
          countCluster.toList,
          components)
        membershipEveryIteration = membershipEveryIteration :+ partition
        componentEveryIteration = componentEveryIteration :+ components
        likelihoodEveryIteration = likelihoodEveryIteration :+ likelihood
        go(iter + 1)
      }
    }

    go(1)

    (membershipEveryIteration, componentEveryIteration, likelihoodEveryIteration)
  }
}

