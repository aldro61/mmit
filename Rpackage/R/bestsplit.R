bestsplit <- structure(function
### Compute vector of optimal prediction and cost.
#We have the feature and target value of the data that is to be splitted.
(target.mat, feature.mat, margin=0.0, loss="hinge"){
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
    feat <- feat[sorted]
    tar <- target.mat[sorted,]
    rev_tar <- target.mat[rev(sorted),]   #reverse order

    #1st and last index of duplicate elem
    first_idx <- order(feat)[!duplicated(feat)]
    last_idx <- c(first_idx-1, length(feat))
    last_idx <- last_idx[-1]

    #if no unique elements
    if(length(last_idx) == 0){
      next
    }

    #compute cost, prediction
    leftleaf <- compute_optimal_costs(tar, margin, loss)
    rightleaf <- compute_optimal_costs(rev_tar, margin, loss)

    #unique and removing cases where all examples are in one leaf
    leftleaf <- leftleaf[last_idx,]
    leftleaf <- leftleaf[-length(leftleaf[,1]),]
    rightleaf$moves <- rev(rightleaf[,1])
    rightleaf$pred <- rev(rightleaf[,2])
    rightleaf$cost <- rev(rightleaf[,3])
    rightleaf <- rightleaf[first_idx,]
    rightleaf <- rightleaf[-1,]

    #removing NA cases and summing both orders
    split_cost <- leftleaf$cost + rightleaf$cost

    #if no split possible
    if(length(split_cost) == 0){
      next
    }

    if(min(split_cost) < best_split$cost){
      best_split$cost <- min(split_cost)
      best_split$br <- feat[which.min(split_cost)]
      best_split$varid <- index
      best_split$row <- which.min(split_cost)
    }

  }
  return(best_split)
})
