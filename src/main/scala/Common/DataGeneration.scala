package Common

import Common.Tools._
import breeze.linalg.{*, DenseMatrix, DenseVector, diag}
import breeze.stats.distributions.{MultivariateGaussian, RandBasis}

import scala.util.Random

object DataGeneration  {

  def randomMixture(modes: List[DenseVector[Double]],
                              covariances: List[DenseMatrix[Double]],
                              sizeCluster: List[Int],
                              shuffle: Boolean = false): List[DenseVector[Double]] = {

    require(modes.length == covariances.length, "modes and covariances lengths do not match")
    require(modes.length == sizeCluster.length, "sizeCluster and modes lengths do not match")
    val K = modes.length

    val data = (0 until K).map(k => {
      MultivariateGaussian(modes(k), covariances(k)).sample(sizeCluster(k))
    }).reduce(_++_).toList
    if(shuffle){Random.shuffle(data)} else data
  }

  def matrixToDataByCol(data: DenseMatrix[DenseVector[Double]]): List[List[DenseVector[Double]]] = {
    data(::,*).map(_.toArray.toList).t.toArray.toList
  }

}
