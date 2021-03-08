package DPMM

import breeze.linalg.{DenseMatrix, DenseVector, det, inv, sum, trace}
import breeze.numerics.constants.Pi
import breeze.numerics.{log, multiloggamma, pow}
import breeze.stats.distributions.{MultivariateGaussian, Wishart}
import org.apache.commons.math3.special.Gamma

class NormalInverseWishart(val mu: DenseVector[Double] = DenseVector(0D),
                           val kappa: Double = 1D,
                           val psi: DenseMatrix[Double] = DenseMatrix(1D),
                           val nu: Int = 1) {
  val p: Int = psi.rows
  val studentNu: Int = this.nu - p + 1
  val studentPsi: DenseMatrix[Double] = ((this.kappa + 1) / (this.kappa * studentNu)) * this.psi

  def this(dimension: Int) = {
    this(DenseVector(Array.fill(dimension)(0D)),
      1D,
      DenseMatrix.eye(dimension),
      dimension + 1)
    this
  }

  def sample(): MultivariateGaussian = {
    val newSig = Wishart(this.nu, this.psi).sample()
    val newMu = MultivariateGaussian(this.mu, inv(newSig * this.kappa)).draw()
    MultivariateGaussian(newMu, newSig/pow(this.kappa,2))
  }

  def predictive(x: DenseVector[Double]): Double = {
    this.multivariateStudentLogPdf(x, this.mu, studentPsi, studentNu)
  }

  def posteriorPredictive(x: DenseVector[Double], X: List[DenseVector[Double]]): Double = {
    this.update(X).predictive(x)
  }

  def multivariateStudentLogPdf(x: DenseVector[Double], mu: DenseVector[Double], sigma: DenseMatrix[Double], nu: Double): Double = {
    val d = mu.length
    val x_mu = x - mu
    val a = Gamma.logGamma((nu + d)/2D)
    val b = Gamma.logGamma(nu / 2D) + (d/2D) * log(Pi * nu) + .5 *log(det(sigma))
    val c = log(pow(1 + (1D/nu)* (x_mu.t * inv(sigma) * x_mu), -(nu + d)/2D))
    (a - b ) + c
  }

  def update(data: List[DenseVector[Double]]): NormalInverseWishart = {

    val n = data.length.toDouble
    val meanData = data.reduce(_ + _) / n.toDouble
    val newKappa = this.kappa + n
    val newNu = this.nu + n.toInt
    val newMu = (this.kappa * this.mu + n * meanData) / newKappa
    val x_mu0 = meanData - this.mu

    val C = if (n == 1) {
      DenseMatrix.zeros[Double](p,p)
    } else {
      sum(data.map(x => {
        val x_mu = x - meanData
        x_mu * x_mu.t
      }))
    }

    val newPsi = this.psi + C + ((n * this.kappa) / newKappa) * (x_mu0 * x_mu0.t)
    new NormalInverseWishart(newMu, newKappa, newPsi, newNu)
  }

  def checkNIWParameterEquals (SS2: NormalInverseWishart): Boolean = {
    (mu == SS2.mu) & (nu == SS2.nu) & ((psi - SS2.psi).toArray.sum <= 10e-7) & (kappa == SS2.kappa)
  }

  def print(): Unit = {
    println()
    println("mu: " + mu.toString)
    println("kappa: " + kappa.toString)
    println("psi: " + psi.toString)
    println("nu: " + nu.toString)
    println()
  }

  def removeObservations(data: List[DenseVector[Double]]): NormalInverseWishart = {
    val n = data.length.toDouble
    val meanData = data.reduce(_ + _) / n.toDouble
    val newKappa = this.kappa - n
    val newNu = this.nu - n.toInt
    val newMu = (this.kappa * this.mu - n * meanData) / newKappa
    val x_mu0 = meanData - this.mu
    val C = if (n == 1) {
      DenseMatrix.zeros[Double](p,p)
    } else {
      sum(data.map(x => {
        val x_mu = x - meanData
        x_mu * x_mu.t
      }))
    }
    val newPsi = this.psi - C - ((n * this.kappa) / newKappa) * (x_mu0 * x_mu0.t)
    new NormalInverseWishart(newMu, newKappa, newPsi, newNu)
  }

  def logPdf(multivariateGaussian: MultivariateGaussian): Double = {
    val gaussianLogDensity = MultivariateGaussian(this.mu, multivariateGaussian.covariance/this.kappa).logPdf(multivariateGaussian.mean)
    val invWishartLogDensity = InvWishartlogPdf(multivariateGaussian.covariance)
    gaussianLogDensity + invWishartLogDensity
  }

  def InvWishartlogPdf(Sigma:DenseMatrix[Double]): Double = {
    (nu / 2D) * log(det(psi)) -
      ((nu * p)/2D) * log(2) -
      multiloggamma( nu/2D, p) -
      0.5*(nu + p + 1) * log(det(Sigma)) -
      0.5 * trace(psi * inv(Sigma))
  }

  def likelihood(alpha: Double,
                 data: List[DenseVector[Double]],
                 membership: List[Int],
                 countCluster: List[Int],
                 components: List[MultivariateGaussian]): Double = {
    val K = countCluster.length
    val partitionDensity = K * log(alpha) +
      countCluster.map(c => Common.Tools.logFactorial(c - 1)).sum -
      data.indices.map(e => log(alpha+e.toDouble)).sum
    val dataLikelihood = data.indices.map( i => components(membership(i)).logPdf(data(i))).sum
    val paramsDensity = components.map(logPdf).sum
    partitionDensity + paramsDensity + dataLikelihood
  }

  def posteriorSample(Data: List[DenseVector[Double]],
                      rowMembership: List[Int]) : List[MultivariateGaussian] = {
    (Data zip rowMembership).groupBy(_._2).values.par.map(e => {
      val dataInCluster = e.map(_._1)
      val k = e.head._2
      (k, this.update(dataInCluster).sample())
    }).toList.sortBy(_._1).map(_._2)
  }

  def posteriorSample(DataByCol: List[List[DenseVector[Double]]],
                      rowMembership: List[Int],
                      colMembership: List[Int]) : List[List[MultivariateGaussian]] = {
    (DataByCol.transpose zip rowMembership).groupBy(_._2).values.par.map(e => {
      val dataInCol = e.map(_._1)
      val k = e.head._2
      (k,
        (dataInCol.transpose zip rowMembership).groupBy(_._2).values.par.map(f => {
          val dataInBlock = f.map(_._1).reduce(_++_)
          val l = f.head._2
          (l, this.update(dataInBlock).sample())
        }).toList.sortBy(_._1).map(_._2))
    }).toList.sortBy(_._1).map(_._2)
  }

  def posteriorExpectation(Data: List[DenseVector[Double]],
                           rowMembership: List[Int]) : List[MultivariateGaussian] = {
    (Data zip rowMembership).groupBy(_._2).values.par.map(e => {
      val dataInCluster = e.map(_._1)
      val k = e.head._2
      (k, this.update(dataInCluster).expectation())
    }).toList.sortBy(_._1).map(_._2)
  }

  def posteriorExpectation(DataByCol: List[List[DenseVector[Double]]],
                           rowMembership: List[Int],
                           colMembership: List[Int]) : List[List[MultivariateGaussian]] = {
    (DataByCol.transpose zip rowMembership).groupBy(_._2).values.par.map(e => {
      val dataInCol = e.map(_._1)
      val k = e.head._2
      (k,
        (dataInCol.transpose zip rowMembership).groupBy(_._2).values.par.map(f => {
          val dataInBlock = f.map(_._1).reduce(_ ++ _)
          val l = f.head._2
          (l, this.update(dataInBlock).expectation())
        }).toList.sortBy(_._1).map(_._2))
    }).toList.sortBy(_._1).map(_._2)
  }

  def expectation() : MultivariateGaussian = {
    val expectedSigma = this.psi / (this.nu - this.p - 1).toDouble
    val expectedMu = this.mu
    MultivariateGaussian(expectedMu, expectedSigma)
  }

}
