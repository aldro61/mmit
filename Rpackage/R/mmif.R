#' Random Forest of Max Margin Interval Tree
#'
#' Performing random forest on ensemble of Max Margin Interval Tree.
#' 
#' @param target.mat The response variable of the model
#' @param feature.mat a data frame containing the feature variables in the model.
#' @param parameters the list of paramaters such as max_depth, margin, loss, min_sample
#' @param n_trees The Number of trees
#' @param n_example  The number of data elements to be sampled from dataset.
#' @param n_features The number of features to be sampled.
#' 
#' @return List of ensemble of trees.
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
#' parameters <- list(max_depth = Inf, margin = 2.0, loss = "hinge", min_sample = 1)
#' 
#' trees <- mmif(target.mat, feature.mat, test, parameters = parameters)
#' 
#' @export
mmif <- structure(function(target.mat, feature.mat, 
                                   parameters, n_trees = 10,
                                   n_example = 100, 
                                   n_features = as.integer(ncol(feature.mat)**0.5)){
 
   ### create n_trees
  all_trees <- list()
  all_pred <- NULL
  for(i in 1 : n_trees){
    
    ### sample n_exalple elements of dataset
    new_target.mat <- NULL
    new_feature.mat <- NULL
    for(j in 1 : n_example){
      x <- sample(nrow(target.mat), 1)
      new_target.mat <- rbind(new_target.mat, target.mat[x,])
      new_feature.mat <- rbind(new_feature.mat, feature.mat[x,])
    }
    
    ### sample features
    w <- rep(1, ncol(feature.mat))
    x <- sample(ncol(feature.mat), n_features)
    new_feature.mat <- new_feature.mat[, x]
    new_target.mat <- data.matrix(new_target.mat)
    
    ### tree
    tree <- mmit(new_target.mat, new_feature.mat, margin = parameters$margin, loss = parameters$loss, 
                 min_sample = parameters$min_sample, max_depth = parameters$max_depth)
    all_trees[[i]] <- tree
  }
  
  return(all_trees)
  
}, ex=function(){
  
  data(neuroblastomaProcessed, package="penaltyLearning")
  feature.mat <- data.frame(neuroblastomaProcessed$feature.mat)[1:45,]
  target.mat <- neuroblastomaProcessed$target.mat[1:45,]
  parameters <- list(max_depth = Inf, margin = 2.0, loss = "hinge", min_sample = 1)
  trees <- mmif(target.mat, feature.mat, parameters = parameters)


})
