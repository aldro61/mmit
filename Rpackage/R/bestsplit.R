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
    
    #unique and sorted
    sorted <- order(feat)[!duplicated(feat)] 
    z<- feat
    feat <- z[sorted]
    rev_feat <-z[rev(sorted)]     #reverse order
    z<- target.mat
    tar <- z[sorted,]
    rev_tar <- z[rev(sorted),]     #reverse order
    
    if(length(sorted)==1){
      next
    }
    
    #compute cost, prediction
    leftpred <- compute_optimal_costs(tar,margin)
    rightpred <- compute_optimal_costs(rev_tar,margin)
    
    #unique and removing cases where all examples are in one leaf
    leftpred <- leftpred[sorted,]
    rightpred <- rightpred[sorted,]
    
    #removing NA cases
    split_cost <- na.omit(leftpred$cost+rightpred$cost)

    if(min(split_cost)<best_split$cost){
      best_split$cost <- min(split_cost)
      best_split$br <- feat[which.min(split_cost)]
      best_split$varid <- index
      best_split$row <- which.min(split_cost)
    }
    
  }
  return(best_split)
})
