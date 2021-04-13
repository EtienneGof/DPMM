package DPMM

import Common.Tools._
import breeze.linalg.DenseVector
import breeze.numerics.log
import breeze.stats.distributions.Gamma

import scala.collection.mutable.ListBuffer

class Default(Data: List[DenseVector[Double]],
              prior: NormalInverseWishart = new NormalInverseWishart(),
              alpha: Option[Double] = None,
              alphaPrior: Option[Gamma] = None,
              initByUserPartition: Option[List[Int]] = None) extends Serializable {

  val n: Int = Data.length

  val priorPredictive: List[Double] = Data.map(prior.predictive)

  var partition: List[Int] = initByUserPartition match {
    case Some(m) =>
      require(m.length == Data.length)
      m
    case None => List.fill(Data.length)(0)
  }

  var countCluster: ListBuffer[Int] = partitionToOrderedCount(partition).to[ListBuffer]

  require(!(alpha.isEmpty & alphaPrior.isEmpty),"Either alpha or alphaPrior must be provided: please provide one of the two parameters.")
  require(!(alpha.isDefined & alphaPrior.isDefined), "Providing both alpha or alphaPrior is not supported: remove one of the two parameters.")

  var (actualAlphaPrior, actualAlpha, updateAlphaFlag): (Gamma, Double, Boolean) = alphaPrior match {
    case Some(g) => (g, g.mean, true)
    case None => {
      require(alpha.get > 0, s"Alpha parameter is optional and should be > 0 if provided, but got ${alpha.get}")
      (new Gamma(1D,1D), alpha.get, false)
    }
  }

  def conditionalAlphaUpdate: Unit = {
    if(updateAlphaFlag){actualAlpha = updateAlpha(actualAlpha, actualAlphaPrior, countCluster.length, n)}
  }

  def posteriorPredictive(data: DenseVector[Double], cluster: Int): Double = {0D}

  def computeClusterPartitionProbabilities(idx: Int,
                                            verbose: Boolean=false): List[Double] = {
    countCluster.indices.map(k => {
      (k, log(countCluster(k).toDouble) + posteriorPredictive(Data(idx), k))
    }).toList.sortBy(_._1).map(_._2)
  }

  def drawMembership(i: Int): Unit = {
    val probPartition = computeClusterPartitionProbabilities(i)
    val probPartitionNewCluster = log(actualAlpha) + priorPredictive(i)
    val normalizedProbs = normalizeLogProbability(probPartition :+ probPartitionNewCluster)
    val newPartition = sample(normalizedProbs)
    partition = partition.updated(i, newPartition)
  }

  def removeElementFromCluster(idx: Int): Unit = {
    val currentPartition = partition(idx)
    if (countCluster(currentPartition) == 1) {
      countCluster.remove(currentPartition)
      partition = partition.map(c => { if( c > currentPartition ){ c - 1 } else c })
    } else {
      countCluster.update(currentPartition, countCluster.apply(currentPartition) - 1)
    }
  }

  def addElementToCluster(idx: Int): Unit = {
    val newPartition = partition(idx)
    if (newPartition == countCluster.length) { // Creation of a new cluster
      countCluster = countCluster ++ ListBuffer(1)
    } else {
      countCluster.update(newPartition, countCluster.apply(newPartition) + 1)
    }
  }

  def updatePartition(verbose: Boolean = false): Unit = {
    for (i <- 0 until n) {
      removeElementFromCluster(i)
      drawMembership(i)
      addElementToCluster(i)
    }
  }

}
