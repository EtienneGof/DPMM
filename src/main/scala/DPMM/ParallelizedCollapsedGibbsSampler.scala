package DPMM

import Common.Tools._
import breeze.linalg.{DenseVector, inv}
import breeze.stats.distributions.{Gamma, MultivariateGaussian}
import smile.validation.{adjustedRandIndex, randIndex}

import scala.annotation.tailrec
import scala.collection.mutable.ListBuffer
import scala.util.Random

class ParallelizedCollapsedGibbsSampler(val Data: List[DenseVector[Double]],
                                        var prior: NormalInverseWishart = new NormalInverseWishart(),
                                        var alpha: Option[Double] = None,
                                        var alphaPrior: Option[Gamma] = None,
                                        var initByUserPartition: Option[List[Int]] = None,
                                        var truePartition: List[Int]
                           ) extends Default(Data, prior, alpha, alphaPrior, initByUserPartition) {

  def run(maxIter: Int = 10,
          maxIterBurnin: Int = 10,
          verbose: Boolean = false,
          nWorker: Int = 10): (List[List[Int]], List[List[MultivariateGaussian]], List[Double]) = {

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

        var workerPartition = List.fill((n / nWorker).toInt + 1)((0 until nWorker).toList).reduce(_++_).slice(0, n)
        workerPartition = Random.shuffle(workerPartition)

        val membershipFromWorker: List[String] = (Data.zipWithIndex zip workerPartition).groupBy(_._2).values.par.map(e => {
          val dataIdx = e.map(_._1._2)
          val dataInWorker = e.map(_._1._1)
          val workerIdx = e.head._2
          val dpmWorker = new DPMM.CollapsedGibbsSampler(
            dataInWorker, prior,
            alpha = alpha)
          val memberships = dpmWorker.run(100, verbose = false)._1.last
          val membershipsWithWorkerIdx = memberships.map(m => workerIdx.toString + "-" + m.toString)
          membershipsWithWorkerIdx zip dataIdx
        }).reduce(_++_).sortBy(_._2).map(_._1)

        val relabeledMembership: List[Int] = relabel(membershipFromWorker)
        val countByCluster = partitionToOrderedCount(relabeledMembership)
        println("partition after Worker")
        println(countByCluster)
        println(adjustedRandIndex(relabeledMembership.toArray, truePartition.toArray))
        println(randIndex(relabeledMembership.toArray, truePartition.toArray))

        val meanPerCluster = (Data zip relabeledMembership).groupBy(_._2).values.map(e => {
          val dataPerCluster = e.map(_._1)
          val idxCluster = e.head._2
          (idxCluster, prior.update(dataPerCluster).expectation().mean)
        }).toList.sortBy(_._1)

        meanPerCluster.indices.foreach(i => println(meanPerCluster(i), countByCluster(i)))

        val membershipMeans = if(meanPerCluster.length > 1){
          val dpmMaster = new DPMM.WeightedCollapsedGibbsSampler(
            meanPerCluster.map(_._2),
            countByCluster,
            prior,
            alpha = alpha)
          dpmMaster.run(maxIter = 200)._1.last
        } else { List(0) }

        partition = Data.indices.map(i => {
          membershipMeans(relabeledMembership(i))
        }).toList
        countCluster = partitionToOrderedCount(partition).to[ListBuffer]

        println("partition after Master")
        println(countCluster)
        println(adjustedRandIndex(partition.toArray, truePartition.toArray))
        println(randIndex(partition.toArray, truePartition.toArray))
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

