bestsplit <- structure(function
### Compute vector of optimal prediction and cost.
#We have the feature and target value of the data that is to be splitted.
(target.mat,feature.mat,margin=0.0,loss="hinge"){
  #We keep track of the following values for each optimal split
  best_split <- NULL
  best_split$cost <- Inf
  best_split$br <- NULL
  best_split$varid <- NULL
  best_split$row <- NULL

  #loop for every feature
  for (index in 2:(length(feature.mat[1,]))){
    feat <- feature.mat[, index]

    #sorted
    sorted <- order(feat)
    z<- feat
    feat <- z[sorted]
    z<- target.mat
    tar <- z[sorted,]
    rev_tar <- z[rev(sorted),]   #reverse order

    #1st and last index of duplicate elem
    first_idx <- order(feat)[!duplicated(feat)]
    last_idx <- c(order(feat)[!duplicated(feat)]-1,length(feat))
    last_idx <- last_idx[-1]

    #if no unique elements
    if(length(last_idx)==0){
      next
    }

    #compute cost, prediction
    leftpred <- compute_optimal_costs(tar,margin)
    rightpred <- compute_optimal_costs(rev_tar,margin)

    #unique and removing cases where all examples are in one leaf
    leftpred <- leftpred[last_idx,]
    leftpred <- leftpred[-length(leftpred[,1]),]
    rightpred$moves <- rev(rightpred[,1])
    rightpred$pred <- rev(rightpred[,2])
    rightpred$cost <- rev(rightpred[,3])
    rightpred <- rightpred[first_idx,]
    rightpred <- rightpred[-1,]

    #removing NA cases and summing both orders
    split_cost <- na.omit(leftpred$cost+rightpred$cost)

    #if no split possible
    if(length(split_cost)==0){
      next
    }

    if(min(split_cost)<best_split$cost){
      best_split$cost <- min(split_cost)
      best_split$br <- feat[which.min(split_cost)]
      best_split$varid <- index
      best_split$row <- which.min(split_cost)
    }

  }
  return(best_split)
})
