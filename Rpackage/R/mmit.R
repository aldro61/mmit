#' The Max Margin Interval Tree
#'
#' Learning a regression tree for censored data.
#'
#' @param target.mat The response variable of the model
#' @param feature.mat a data frame containing the feature variables in the model.
#' @param margin margin paramaters 
#' @param loss The type of loss; (\code{"hinge"}, \code{"square"})
#' @param max_depth The maximum depth criteia
#' @param min_sample The minimum number of sample required 
#' @param weights The weights of feature variables, (default = 1)
#' 
#' @return The learned regression tree as an object of class party.
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
#' 
#' out <- mmit(target.mat, feature.mat)
#' 
#' @export
mmit <- structure(function(target.mat, feature.mat,  
                           max_depth = Inf, margin=0.0, loss="hinge",
                           min_sample = 1, weights = rep(1L, nrow(feature.mat))) {
  ### partynode id and initial depth
  id = 1L
  
  ### arranging feature.mat.
  feature.mat <- model.frame(data = feature.mat)
  
  ### assigning all weights as 1 in the beginning. weights determine which data is to be considered in which node.
  if(all.equal(weights, rep(1L, nrow(feature.mat)))[1] == TRUE){
    type = "none"
  }
  else{
    type = "weighted"
  }
  stopifnot(length(weights) == nrow(feature.mat))
  
  ### tree
  node<- growtree(target.mat, feature.mat, max_depth = max_depth, margin = margin, weights = weights, type = type)
  
  ### for node == root
  if(is.null(model.response(feature.mat))){
    response <- tail(compute_optimal_costs(target.mat, margin, loss, type, weights)$pred, 1)
    response <- rep(response, nrow(feature.mat))
  }
  else{
    response <- model.response(feature.mat)
  }
  
  ### compute terminal node number for each observation
  tree <- party(node, data = feature.mat, fitted = data.frame("(fitted)" = fitted_node(node, data = feature.mat),
                                                          "(response)" = response,
                                                          "(weights)" = weights,
                                                          check.names = FALSE), terms = terms(feature.mat))

  return(tree)
}, ex=function(){
  
  data(neuroblastomaProcessed, package="penaltyLearning")
  feature.mat <- data.frame(neuroblastomaProcessed$feature.mat)[1:45,]
  target.mat <- neuroblastomaProcessed$target.mat[1:45,]
  tree <- mmit(target.mat, feature.mat, max_depth = Inf, margin = 2.0)
  plot(tree)
  
})


