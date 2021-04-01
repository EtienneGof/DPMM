import Common.DataGeneration
import DPMM.{CollapsedGibbsSampler, GibbsSampler, NormalInverseWishart}
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Gamma
import smile.validation.{NormalizedMutualInformation, adjustedRandIndex, randIndex}

object Main {
  def main(args: Array[String]): Unit = {

    val modes = List(DenseVector(1.7, 1.7),DenseVector(1.7, -1.7),DenseVector(-1.7, 1.7),DenseVector(-1.7, -1.7))

    val covariances = List(
      DenseMatrix(0.25, 0.0, 0.0, 0.25).reshape(2,2),
      DenseMatrix(0.17, 0.1, 0.1, 0.17).reshape(2,2),
      DenseMatrix(0.12, 0.0, 0.0, 0.12).reshape(2,2),
      DenseMatrix(0.3, 0.0, 0.0, 0.3).reshape(2,2))

    val data = DataGeneration.randomMixture(modes, covariances, List.fill(4)(150))

    val empiricMean = Common.Tools.mean(data)
    val empiricCovariance = Common.Tools.covariance(data, empiricMean)
    val prior = new NormalInverseWishart(empiricMean, 1D, empiricCovariance, data.head.length + 1)

    val alpha = 5D
    val nIter = 100

    val alphaPrior = Gamma(shape = 9, scale = 0.5)
    val mm = new GibbsSampler(data, prior, alpha = Some(alpha))

    val (membership, components, likelihoods) = mm.run(nIter)

    val trueMembership = List(0,1,2,3).flatMap(List.fill(150)(_))


    val ARI = adjustedRandIndex(membership.last.toArray, trueMembership.toArray)
    val RI = randIndex(membership.last.toArray, trueMembership.toArray)

    println("ARI = " + ARI.toString)
    println("RI = " + RI.toString)

    Common.IO.writeCompleteMCMC("results", data, membership, components, likelihoods)
  }
}
