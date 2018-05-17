bestsplit <- structure(function
### Compute vector of optimal prediction and cost.
#We have the feature and target value of the data that is to be splitted.
(target.mat,featue.mat,margin=0.0,loss="hinge"){
  #We keep track of the following values for each optimal split
  best_split <- NULL
  best_split$cost <- Inf
  best_split$br <- NULL
  best_split$varid <- NULL
  best_split$row <- NULL

  
  for (index in 2:(length(feature.mat[1,]))){
    feat = feature.mat[, index]
    sorted <- order(feat)
    z<- feat
    feat <- z[sorted]
    rev_feat <-z[rev(sorted)]
    z<- target.mat
    tar <- z[sorted,]
    rev_tar <- z[rev(sorted),]
    
    
    leftpred <- compute_optimal_costs(tar,margin)
    rightpred <- compute_optimal_costs(rev_tar,margin)
    
    split_cost <- leftpred$cost+rightpred$cost
    
    if(any(split_cost<best_split$cost)){
      best_split$cost <- min(split_cost)
      best_split$br <- feat[which.min(split_cost)]
      best_split$varid <- index
      best_split$row <- which.min(split_cost)
    }
    
  }
  return(best_split)
})


data(neuroblastomaProcessed, package="penaltyLearning")
survTarget <- with(neuroblastomaProcessed, {Surv(target.mat[, 1], target.mat[,2], type="interval2")})
feature.mat <- data.frame(log.penalty=survTarget, neuroblastomaProcessed$feature.mat)
target.mat <- neuroblastomaProcessed$target.mat
best_split <- bestsplit(target.mat,feature.mat,2.0,loss="hinge")
View(best_split)
