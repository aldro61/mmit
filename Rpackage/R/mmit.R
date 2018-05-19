mmit <- structure(function(formula, target.mat, feature.mat, weights = NULL,depth=0, maxdepth = Inf,margin=0.0,loss="hinge",id = 1L, min_sample = 1) {
  node<- growtree(target.mat, feature.mat, maxdepth = maxdepth, margin = margin)
  tree <- party(node, data = feature.mat)

  return(tree)
})


library(survival)
data(neuroblastomaProcessed, package="penaltyLearning")
survTarget <- with(neuroblastomaProcessed, {Surv(target.mat[, 1], target.mat[,2], type="interval2")})
feature.mat <- data.frame(log.penalty=survTarget, neuroblastomaProcessed$feature.mat)[1:145,]
target.mat <- neuroblastomaProcessed$target.mat[1:145,]
tree <- mmit(log.penalty~.,target.mat, feature.mat, maxdepth = 4,margin = 2.0)
plot(tree)
