#' Cross-validation for model selection with Random Forests of Max Margin Interval Trees
#'
#' Performing grid search to select the best hyperparameters of mmif via cross-validation.
#' 
#' @param target.mat The response variable of the model
#' @param feature.mat A data frame containing the feature variables in the model.
#' @param param_grid A list with values to try for each hyperparameter (max_depth, margin, min_sample, loss, n_trees, n_features).
#' @param n_folds The number of folds for k-fold cross-validation
#' @param scorer The function used to calculate the cross-validation score (e.g., mse, zero_one_loss)
#' 
#' @return The best score, best model (trained with best parameters), best parameters, and list of all parameter values with cross validation score. 
#' 
#' @author Toby Dylan Hocking, Alexandre Drouin, Torsten Hothorn, Parismita Das
#' 
#' @examples
#' library(mmit)
#'
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
#' param_grid$n_trees <- c(10, 20, 30)
#' param_grid$n_features <- c(ceiling(ncol(feature.mat)**0.5))
#' 
#' result <- mmif.cv(target.mat, feature.mat, param_grid, scorer = mse)
#' 
mmif.cv <- structure(function(target.mat, feature.mat, 
                              param_grid, n_folds = 3,
                              scorer = NULL){
  
  ### add default value to parameters
  if(is.null(param_grid[["max_depth"]])) param_grid$max_depth <- Inf
  if(is.null(param_grid[["margin"]])) param_grid$margin <- 0.0
  if(is.null(param_grid[["min_sample"]])) param_grid$min_sample <- 0.0
  if(is.null(param_grid[["loss"]])) param_grid$loss <- "hinge"
  if(is.null(param_grid[["n_trees"]])) param_grid$n_trees <- 10
  if(is.null(param_grid[["n_features"]])) param_grid$n_features <- c(as.integer(ncol(feature.mat)**0.5))
  
  ### check for unwanted parameters
  assert_that(length(param_grid) <= 6, msg = "unexpected parameters as argument")
  
  ### combinations of all parameters grid
  parameters <- expand.grid(max_depth = param_grid$max_depth, margin = param_grid$margin, 
                            min_sample = param_grid$min_sample, loss = param_grid$loss,
                            n_trees = param_grid$n_trees, n_features = param_grid$n_features)
  parameters <- as.data.frame(parameters)
  
  ### for param grid run loop to check best value
  cv_results <- NULL
  best_result <- NULL
  best_result$best_score <- attr(scorer, "worst")
  
  fitscore_result <- list()
  fitscore_result <- future_lapply(1:nrow(parameters), 
                  function(x) .fit_and_score(target.mat = target.mat, feature.mat = feature.mat, 
                  parameters = parameters[x,], learner = "mmif", 
                  n_folds = n_folds, scorer = scorer, pruning = FALSE))

  for(i in 1:nrow(parameters)){
    cv_results <- rbind(cv_results, fitscore_result[[i]]$cv_results)
    if(attr(scorer, "direction")(best_result$best_score, fitscore_result[[i]]$best_score) == fitscore_result[[i]]$best_score){
      best_result <- fitscore_result[[i]]
    }
  }    
  
  best_result$cv_results <- cv_results
  
  return(best_result)
  
}, ex=function(){
  data(neuroblastomaProcessed, package="penaltyLearning")
  feature.mat <- data.frame(neuroblastomaProcessed$feature.mat)[1:45,]
  target.mat <- neuroblastomaProcessed$target.mat[1:45,]
  
  param_grid <- NULL
  param_grid$max_depth <- c(4, 3)
  param_grid$margin <- c(2, 3)
  param_grid$min_sample <- c(5, 20)
  param_grid$loss <- c("hinge", "square")
  param_grid$n_trees <- c(10)
  param_grid$n_features <- c(as.integer(ncol(feature.mat)**0.5))
  
  result <- mmif.cv(target.mat, feature.mat, param_grid, scorer = mse)
})

