library(mixtools)
library(animation)

plotClusteringState <- function(data, clustering, components, likelihoods, idx) {
  nComponents = length(unique(clustering))
  par(mfrow = c(2,1))
  plot(data, col = clustering + 1, xlim= c(-4, 4), ylim= c(-4, 4), main = "Cluster Membership Evolution", xlab = "", ylab = "")
  for(k in seq_len(nComponents)){
    mean = as.numeric(strsplit(components[1, (k - 1) *2 + 1],":")[[1]])
    covMat = matrix(as.numeric(strsplit(components[1, k * 2],":")[[1]]),2,2)
    ellipse(mu=mean, sigma=covMat, alpha = .005, npoints = 250, col=k) 
    ellipse(mu=mean, sigma=covMat, alpha = .01, npoints = 250, col=k) 
    ellipse(mu=mean, sigma=covMat, alpha = .05, npoints = 250, col=k) 
    ellipse(mu=mean, sigma=covMat, alpha = .10, npoints = 250, col=k) 
  }
  plot(y=likelihoods, x = 1:length(likelihoods), t="l", xlab = "Iteration", ylab = "LogLikelihood", main = "LogLikelihood Evolution")
  points(x= idx, y= likelihoods[idx], col="red", cex=1)
  points(x= idx, y= likelihoods[idx], col="red", cex=2)
  
}

dirPath = [RESULT PATH HERE]

data = read.csv(paste0(dirPath, "/data.csv"), header = F)[,-1]
# Final clustering obtained after dpmm 
clusteringEveryIteration = read.csv(paste0(dirPath, "/membershipHistory.csv"), header = F)[,-1]
nClusterMax = max(clusteringEveryIteration) + 1
# One line = every components of a given iteration 
componentsEveryIterations = read.csv(paste0(dirPath, "/componentsHistory.csv"),
                      stringsAsFactors = F, header = F, 
                      col.names = paste0("V",seq_len(2*nClusterMax+1)),
                      fill = TRUE)[,-1]
# likelihoods
likelihoods = read.csv(paste0(dirPath, "/likelihoods.csv"),
                                     stringsAsFactors = F, header = F)[-1,-1]


nIter = length(likelihoods)
colnames(data) <- c("X", "Y")

nIteration = nrow(clusteringEveryIteration)
oopt = ani.options(interval = 0.4)

des = c("Bivariate Gaussian DPMM.\n\n")
saveGIF({
  par(mar = c(4, 4, 2, 2))
  for (i in 1:nIter) {
    print(i)
    plotClusteringState(data,
                        clustering = unlist(clusteringEveryIteration[i,]),
                        components = componentsEveryIterations[i, ],
                        likelihoods, i)
    ani.pause()
  }
}, title = "Evolution of DPMM clustering",
description = des,ani.width = 400, ani.height = 700)
