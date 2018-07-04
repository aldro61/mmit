mmit.cv <- structure(function(target.mat, feature.mat, 
                              param_grid, feature_names = NULL, 
                              n_folds = 3, scorer = NULL,
                              pruning = TRUE){
  
  ### combinations of all parameters grid
  parameters <- expand.grid(maxdepth = param_grid$maxdepth, margin = param_grid$margin, min_sample = param_grid$min_sample, loss = param_grid$loss)
  parameters <- as.data.frame(parameters)
  
  ### for param grid run loop to check best value
  cv_result <- NULL
  best_result <- NULL
  best_result$best_score <- attr(scorer, "worst")
  
  ### parallelize using foreach, see all permutation combination of param grid values
  ### register parallel backend
  cl <- makeCluster(2)
  registerDoParallel(cl)
  
  cv_result <- foreach(i = 1:nrow(parameters), 
              .packages = c("mmit", "assertthat")) %dopar% 
              fit_and_score(target.mat = target.mat, feature.mat = feature.mat, 
                                           parameters = parameters[i,], feature_names = feature_names, 
                                           n_folds = n_folds, scorer = scorer,
                                           pruning = pruning)
      
  for(i in 1:nrow(parameters)){
    if(attr(scorer, "direction")(best_result$best_score, cv_result[[i]]$best_score) == cv_result[[i]]$best_score){
      best_result <- cv_result[[i]]
    }
  }    
    
  return(best_result)
  
}, ex=function(){
  
  library(survival)
  data(neuroblastomaProcessed, package="penaltyLearning")
  feature.mat <- data.frame(neuroblastomaProcessed$feature.mat)[1:45,]
  target.mat <- neuroblastomaProcessed$target.mat[1:45,]
  
  param_grid <- NULL
  param_grid$maxdepth <- c(Inf, 4, 3)
  param_grid$margin <- c(2, 3, 5)
  param_grid$min_sample <- c(2, 5, 10)
  param_grid$loss <- c("hinge")
  
  result <- mmit.cv(target.mat, feature.mat, param_grid, scorer = mse)
  
  
})