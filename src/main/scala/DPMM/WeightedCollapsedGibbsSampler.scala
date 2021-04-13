package DPMM

import Common.Tools._
import breeze.linalg.{DenseVector, inv}
import breeze.numerics.log
import breeze.stats.distributions.{Gamma, MultivariateGaussian}

import scala.annotation.tailrec
import scala.collection.mutable.ListBuffer

class WeightedCollapsedGibbsSampler(Data: List[DenseVector[Double]],
                                    weigthPerPoint: List[Int],
                                    prior: NormalInverseWishart = new NormalInverseWishart(),
                                    alpha: Option[Double] = None,
                                    alphaPrior: Option[Gamma] = None,
                                    initByUserPartition: Option[List[Int]] = None
                                 ) extends CollapsedGibbsSampler(Data, prior, alpha, alphaPrior, initByUserPartition) {

  var weightCluster: ListBuffer[Int] = (partition zip weigthPerPoint).groupBy(_._1).mapValues(_.map(_._2).sum).values.to[ListBuffer]

  NIWParams = ((Data zip weigthPerPoint) zip partition).groupBy(_._2).values.map(e => {
    val repeatedData = e.map(_._1).map(e => List.fill(e._2)(e._1)).reduce(_++_)
    val clusterIdx = e.head._2
    (clusterIdx, prior.update(repeatedData))
  }).toList.sortBy(_._1).map(_._2).to[ListBuffer]

  override def computeClusterPartitionProbabilities(idx: Int, verbose: Boolean): List[Double] = {
    weightCluster.indices.map(k => {
        (k, log(weightCluster(k).toDouble) + posteriorPredictive(Data(idx), k))
      }).toList.sortBy(_._1).map(_._2)
  }

  override def drawMembership(i: Int): Unit = {
    val probPartition = computeClusterPartitionProbabilities(i)
    val probPartitionNewCluster = log(actualAlpha) + priorPredictive(i)
    val normalizedProbs = normalizeLogProbability(probPartition :+ probPartitionNewCluster)
    val newPartition = sample(normalizedProbs)
    partition = partition.updated(i, newPartition)
  }

  override def removeElementFromCluster(idx: Int): Unit = {
    val currentMembership = partition(idx)
    if (countCluster(currentMembership) == 1) {
      countCluster.remove(currentMembership)
      weightCluster.remove(currentMembership)
      partition = partition.map(c => { if( c > currentMembership ){ c - 1 } else c })
    } else {
      countCluster.update(currentMembership, countCluster.apply(currentMembership) - 1)
      weightCluster.update(currentMembership, weightCluster.apply(currentMembership) - weigthPerPoint(idx))
    }
  }

  override def addElementToCluster(idx: Int): Unit = {
    val newMembership = partition(idx)
    if (newMembership == countCluster.length) { // Creation of a new cluster
      countCluster = countCluster ++ ListBuffer(1)
      weightCluster = weightCluster ++ ListBuffer(weigthPerPoint(idx))
    } else {
      countCluster.update(newMembership, countCluster.apply(newMembership) + 1)
      weightCluster.update(newMembership, weightCluster.apply(newMembership) + weigthPerPoint(idx))
    }
  }

  override def removeElementFromNIW(idx: Int): Unit = {
    val currentPartition =  partition(idx)
    if (countCluster(currentPartition) == 1) {
      NIWParams.remove(currentPartition)
    } else {
      val updatedNIWParams = NIWParams(currentPartition).weightedRemoveObservation(Data(idx), weigthPerPoint(idx))
      NIWParams.update(currentPartition, updatedNIWParams)
    }
  }

  override def addElementToNIW(idx: Int): Unit = {
    val newPartition = partition(idx)
    if (newPartition == countCluster.length) { // Creation of a new cluster
      val newNIWparam = this.prior.weightedUpdate(Data(idx), weigthPerPoint(idx))
      NIWParams = NIWParams ++ ListBuffer(newNIWparam)
    } else {
      val updatedNIWParams = NIWParams(newPartition).weightedUpdate(Data(idx), weigthPerPoint(idx))
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

  override def run(maxIter: Int = 10,
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

    (membershipEveryIteration, componentEveryIteration, likelihoodEveryIteration)  }
}

