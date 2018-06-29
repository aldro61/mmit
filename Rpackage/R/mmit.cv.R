mmit.cv <- structure(function(target.mat, feature.mat, 
                              param_grid, feature_names = NULL, 
                              n_folds = 3, scorer = NULL, loss = "hinge",
                              pruning = TRUE){
  
  ### combinations of all parameters grid
  parameters <- NULL
  row <- nrow(param_grid)
  col <- ncol(param_grid)
  parameters$maxdepth <- rep(param_grid[,1], each = row**(col-1))
  parameters$margin <- rep(param_grid[,2], each = row**(col-2), times = row)
  parameters$min_sample <- rep(param_grid[,3], each = row**(col-3), times = row**2)
  parameters <- as.data.frame(parameters)
  
  ### for param grid run loop to check best value
  cv_result <- NULL
  best_result <- NULL
  best_result$best_score <- -Inf
  for(i in 1:nrow(parameters)){
      ### see all permutation combination of param grid values
      cv_result[[i]] <- fit_and_score(target.mat = target.mat, feature.mat = feature.mat, 
                                       parameters = parameters[i,], feature_names = feature_names, 
                                       n_folds = n_folds, scorer = scorer, loss = loss,
                                       pruning = pruning)

      if(best_result$best_score >= cv_result[[i]]$best_score){
        best_result <- cv_result[[i]]
      }
  }
  
  return(best_result)
  
}, ex=function(){
  
  library(survival)
  data(neuroblastomaProcessed, package="penaltyLearning")
  feature.mat <- data.frame(neuroblastomaProcessed$feature.mat)[1:45,]
  target.mat <- neuroblastomaProcessed$target.mat[1:45,]
  
  param_grid <- rbind(c(Inf, 2, 1), c(4, 3, 5), c(3,4,6), c(2, 5, 10))
  colnames(param_grid) <- c("maxdepth", "margin", "min_sample")
  param_grid <- as.data.frame(param_grid)
  
  result <- mmit.cv(target.mat, feature.mat, param_grid)
  
  
})