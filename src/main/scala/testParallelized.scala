import Common.DataGeneration
import DPMM.{NormalInverseWishart, ParallelizedCollapsedGibbsSampler}
import breeze.linalg.{DenseMatrix, DenseVector}

object testParallelized {
  def main(args: Array[String]): Unit = {

    val modes = List(DenseVector(2D, 2D),DenseVector(2D, -2),DenseVector(-2D, 2),DenseVector(-2D, -2))

    val covariances = List(
      DenseMatrix(0.05, 0.0, 0.0, 0.05).reshape(2,2),
      DenseMatrix(0.07, 0.0, 0.0, 0.07).reshape(2,2),
      DenseMatrix(0.02, 0.0, 0.0, 0.02).reshape(2,2),
      DenseMatrix(0.03, 0.0, 0.0, 0.03).reshape(2,2))

    val trueClusterSize = List.fill(4)(100)
//    val trueClusterNumber = List(500,500,500,100)

    val data = DataGeneration.randomMixture(modes, covariances, trueClusterSize)

    val empiricMean = Common.Tools.mean(data)
    val empiricCovariance = Common.Tools.covariance(data, empiricMean)
    val prior = new NormalInverseWishart(empiricMean, 1D, empiricCovariance, data.head.length + 1)

    val alpha = 100

//    val alphaPrior = Gamma(shape = 9, scale = 0.5)
//    val gs = new GibbsSampler(data, prior, alpha = Some(alpha))
//
//    gs.run(nIter)
//    val n = data.length

    val cgs = new ParallelizedCollapsedGibbsSampler(
      data,
      prior,
      alpha = Some(alpha),
      truePartition = Common.Tools.getPartitionFromSize(trueClusterSize))

    cgs.run(10,  verbose = true, nWorker = 2, maxIterBurnin =  0)

  }
}
