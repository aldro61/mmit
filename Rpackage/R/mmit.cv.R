#' The Cross Validation of Max Margin Interval Tree
#'
#' Performing grid search to select the best parameters via cross validation on the  a regression tree for censored data.
#' 
#' @param target.mat The response variable of the model
#' @param feature.mat a data frame containing the feature variables in the model.
#' @param param_grid the list of paramaters
#' @param n_folds The number of folds
#' @param scorer The Loss calculation function 
#' @param pruning Boolean whether pruning is to be done or not.
#' 
#' @return The list consist of best score, best tree, best parameters and list of all parameter values with cross validation score . 
#' 
#' @author Toby Dylan Hocking, Alexandre Drouin, Torsten Hothorn, Parismita Das
#' 
#' @examples
#' library(mmit)
#' target.mat <- rbind(
#'   c(0,1), c(0,1), c(0,1),
#'   c(2,3), c(2,3), c(2,3))
#' 
#' feature.mat <- rbind(
#'   c(1,0,0), c(1,1,0), c(1,2,0),
#'   c(1,3,0), c(1,4,0), c(1,5,0))
#' 
#' colnames(feature.mat) <- c("a", "b", "c")
#' feature.mat <- data.frame(feature.mat)
#' 
#' param_grid <- NULL
#' param_grid$max_depth <- c(Inf, 4, 3)
#' param_grid$margin <- c(2, 3, 5)
#' param_grid$min_sample <- c(2, 5, 10)
#' param_grid$loss <- c("hinge")
#' 
#' result <- mmit.cv(target.mat, feature.mat, param_grid, scorer = mse)
#' 
mmit.cv <- structure(function(target.mat, feature.mat, 
                              param_grid, n_folds = 3,
                              scorer = NULL, pruning = TRUE){
  
  ### add default value to parameters
  if(is.null(param_grid[["max_depth"]])) param_grid$max_depth <- Inf
  if(is.null(param_grid[["margin"]])) param_grid$margin <- 0.0
  if(is.null(param_grid[["min_sample"]])) param_grid$min_sample <- 0.0
  if(is.null(param_grid[["loss"]])) param_grid$loss <- "hinge"
  
  ### check for unwanted parameters
  assert_that(length(param_grid) <= 4, msg = "unexpected parameters as argument")
  
  ### combinations of all parameters grid
  parameters <- expand.grid(max_depth = param_grid$max_depth, margin = param_grid$margin, min_sample = param_grid$min_sample, loss = param_grid$loss)
  parameters <- as.data.frame(parameters)
  
  ### for param grid run loop to check best value
  cv_results <- NULL
  best_result <- NULL
  best_result$best_score <- attr(scorer, "worst")
  
  #lapply parallel or sequencial
  Lapply <- if(requireNamespace("future.apply")){ 
    future.apply::future_lapply 
  }
  else{ lapply }
  
  fitscore_result <- list()
  fitscore_result <- Lapply(1:nrow(parameters), 
                    function(x) .fit_and_score(target.mat = target.mat, 
                    feature.mat = feature.mat, parameters = parameters[x,], 
                    n_folds = n_folds, scorer = scorer, pruning = pruning, learner = "mmit"))
  
  for(i in 1:nrow(parameters)){
    cv_results <- rbind(cv_results, fitscore_result[[i]]$cv_results)
    if(attr(scorer, "direction")(best_result$best_score, fitscore_result[[i]]$best_score) == fitscore_result[[i]]$best_score){
      best_result <- fitscore_result[[i]]
    }
  }    
  
  best_result$cv_results <- cv_results
  
  return(best_result)
  
})
