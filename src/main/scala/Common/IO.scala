package Common

import java.io.{BufferedWriter, FileWriter}

import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.stats.distributions.MultivariateGaussian
import com.opencsv.CSVWriter
import scala.collection.JavaConverters._
import scala.io.Source
import scala.util.{Failure, Try}

object IO {

  def addPrefix(lls: List[List[String]]): List[List[String]] =
    lls.foldLeft((1, List.empty[List[String]])){
      case ((serial: Int, acc: List[List[String]]), value: List[String]) =>
        (serial + 1, (serial.toString +: value) +: acc)
    }._2.reverse


  def writeMatrixDoubleToCsv(fileName: String, Matrix: DenseMatrix[Double], withHeader:Boolean=true): Unit = {
    val header: List[String] = List("id") ++ (0 until Matrix.cols).map(_.toString).toList
    val rows : List[List[String]] = Matrix(*, ::).map(dv => dv.toArray.map(_.toString).toList).toArray.toList
    if(withHeader){
      writeCsvFile(fileName, addPrefix(rows), header)
    } else {
      writeCsvFile(fileName, addPrefix(rows))
    }
  }

  def writeMatrixIntToCsv(fileName: String, Matrix: DenseMatrix[Int], withHeader:Boolean=true): Unit = {
    val header: List[String] = List("id") ++ (0 until Matrix.cols).map(_.toString).toList
    val rows : List[List[String]] = Matrix(*, ::).map(dv => dv.toArray.map(_.toString).toList).toArray.toList
    if(withHeader){
      writeCsvFile(fileName, addPrefix(rows), header)
    } else {
      writeCsvFile(fileName, addPrefix(rows))
    }
  }

  def writeCsvFile(fileName: String,
                   rows: List[List[String]],
                   header: List[String] = List.empty[String],
                   append:Boolean=false
                  ): Try[Unit] =
  {
    val content = if(header.isEmpty){rows} else {header +: rows}
    Try(new CSVWriter(new BufferedWriter(new FileWriter(fileName, append)))).flatMap((csvWriter: CSVWriter) =>
      Try{
        csvWriter.writeAll(
          content.map(_.toArray).asJava
        )
        csvWriter.close()
      } match {
        case e @ Failure(_) =>
          Try(csvWriter.close()).recoverWith{ case _ => e}
        case v => v
      }
    )
  }

  def writeGaussianComponentsParameters(pathOutput: String, components: List[List[MultivariateGaussian]]): Unit = {
    val outputContent = components.map(gaussianList => {
      gaussianList.map(G =>
        List(
          G.mean.toArray.mkString(":"),
          G.covariance.toArray.mkString(":"))).reduce(_++_)
    })
    Common.IO.writeCsvFile(pathOutput, Common.IO.addPrefix(outputContent))
  }

  def writeMCMCLastState(pathOutputBase: String,
                         data: List[DenseVector[Double]],
                         membership: List[List[Int]],
                         components: List[List[MultivariateGaussian]]): Unit = {
    val finalClustering = membership.transpose.map(m => m.groupBy(identity).mapValues(_.size).toList.maxBy(_._2)._1)
    val finalComponent = List(components.last)
    Common.IO.writeMatrixDoubleToCsv(pathOutputBase + "/dataLine.csv", DenseMatrix(data:_*),  withHeader = false)
    Common.IO.writeMatrixIntToCsv(pathOutputBase + "/clustering.csv", DenseMatrix(finalClustering:_*), withHeader = false)
    Common.IO.writeGaussianComponentsParameters(pathOutputBase + "/components.csv", finalComponent)
  }

  def writeCompleteMCMC(pathOutputBase: String,
                        data: List[DenseVector[Double]],
                        membership: List[List[Int]],
                        components: List[List[MultivariateGaussian]],
                        likelihoods: List[Double]): Unit = {
    Common.IO.writeMatrixDoubleToCsv(pathOutputBase + "/data.csv", DenseMatrix(data:_*),  withHeader = false)
    Common.IO.writeMatrixIntToCsv(pathOutputBase + "/membershipHistory.csv", DenseMatrix(membership:_*), withHeader = false)
    Common.IO.writeGaussianComponentsParameters(pathOutputBase + "/componentsHistory.csv", components)
    Common.IO.writeMatrixDoubleToCsv(pathOutputBase + "/likelihoods.csv", DenseMatrix(likelihoods:_*).reshape(likelihoods.length, 1))

  }

}
