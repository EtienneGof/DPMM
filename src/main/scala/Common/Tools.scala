package Common

import breeze.linalg.{DenseMatrix, DenseVector, max, sum}
import breeze.numerics.{exp, log}

object Tools extends java.io.Serializable {

  def covariance(X: List[DenseVector[Double]], mode: DenseVector[Double]): DenseMatrix[Double] = {

    require(mode.length==X.head.length)
    val XMat: DenseMatrix[Double] = DenseMatrix(X.toArray:_*)
    val p = XMat.cols

    val modeMat: DenseMatrix[Double] = DenseMatrix.ones[Double](X.length,1) * mode.t
    val XMatCentered: DenseMatrix[Double] = XMat - modeMat
    val covmat = (XMatCentered.t * XMatCentered)/ (X.length.toDouble-1)

    round(covmat,8)
  }

  def mean(X: List[DenseVector[Double]]): DenseVector[Double] = {
    require(X.nonEmpty)
    X.reduce(_+_) / X.length.toDouble
  }

  def sample(probabilities: List[Double]): Int = {
    val dist = probabilities.indices zip probabilities
    val threshold = scala.util.Random.nextDouble
    val iterator = dist.iterator
    var accumulator = 0.0
    while (iterator.hasNext) {
      val (cluster, clusterProb) = iterator.next
      accumulator += clusterProb
      if (accumulator >= threshold)
        return cluster
    }
    sys.error("Error")
  }

  def logSumExp(X: List[Double]): Double ={
    val maxValue = max(X)
    maxValue + log(sum(X.map(x => exp(x-maxValue))))
  }

  def normalizeProbability(probs: List[Double]): List[Double] = {
    val LSE = Common.Tools.logSumExp(probs)
    probs.map(e => exp(e - LSE))
  }

  def factorial(n: Double): Double = {
    if (n == 0) {1} else {n * factorial(n-1)}
  }

  def logFactorial(n: Double): Double = {
    if (n == 0) {0} else {log(n) + logFactorial(n-1)}
  }

  def memberShipToOrderedCount(membership: List[Int]): List[Int] = {
    membership.groupBy(identity).mapValues(_.size).toList.sortBy(_._1).map(_._2)
  }

  def round(m: DenseMatrix[Double], digits:Int): DenseMatrix[Double] = {
    m.map(round(_,digits))
  }

  def round(x: Double, digits:Int): Double = {
    val factor: Double = Math.pow(10,digits)
    Math.round(x*factor)/factor
  }

}
