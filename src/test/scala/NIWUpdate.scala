
import Common.Tools._
import DPMM.NormalInverseWishart
import breeze.linalg.{DenseMatrix, DenseVector, inv}
import breeze.numerics.log
import breeze.stats.distributions.MultivariateGaussian
import org.scalatest.FunSuite

class NIWUpdate extends FunSuite {
  test("Update NIW parameters 1 element") {
    {

      val data1 = List(DenseVector(1D), DenseVector(2D) ,DenseVector(3D))
      val data2 = List(DenseVector(1D), DenseVector(2D))
      val data3 = List(DenseVector(3D))

      val initParams = new NormalInverseWishart()
      val params1 = initParams.update(data1)
      val params2 = initParams.update(data2)

      assert(params2.update(data3).checkNIWParameterEquals(params1))
      assert(params1.removeObservations(data3).checkNIWParameterEquals(params2))

    }
  }

  test("Update NIW parameters several elements") {
    {
      val data1 = List(DenseVector(1D), DenseVector(2D) ,DenseVector(3D), DenseVector(4D))
      val data2 = List(DenseVector(1D), DenseVector(2D))
      val data3 = List(DenseVector(3D), DenseVector(4D))
      val initParams = new NormalInverseWishart()
      val params1 = initParams.update(data1)
      val params2 = initParams.update(data2)
      assert(params2.update(data3).checkNIWParameterEquals(params1))
      assert(params1.removeObservations(data3).checkNIWParameterEquals(params2))
    }
  }


  test("Update NIW with data") {
    {
      val data = MultivariateGaussian(DenseVector(0D,0D), DenseMatrix.eye[Double](2)).sample(100).toList
      val globalMean = Common.Tools.mean(data)
      val globalVariance = Common.Tools.covariance(data, globalMean)
      val globalPrecision = inv(globalVariance)
      val prior = new NormalInverseWishart(globalMean, 1D, globalPrecision, data.head.length + 1)
      val updatedPrior = prior.update(data)

      prior.print()
      updatedPrior.print()

      println(globalMean)
      println(globalVariance)
      println(globalPrecision)

      assertResult(globalMean)(prior.mu)
      assertResult(globalMean)(updatedPrior.mu)
      assertResult(globalPrecision)(prior.mu)

    }
  }

  test("Wishart density") {
    {
      val prior = new NormalInverseWishart(
        DenseVector(0,0),
        1,
        2D * DenseMatrix.eye[Double](2),
        3)
      val mv = MultivariateGaussian(DenseVector(0,0),
        DenseMatrix.eye[Double](2))
      val d = prior.logPdf(mv)
      assertResult(round(d,6))(-4.28946)
    }
  }

  test("log factorial") {
    {

      assertResult(logFactorial(5D))(log(factorial(5D)))
    }
  }
}