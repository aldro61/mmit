#' The Predict Function for Max Margin Interval Tree
#'
#' Fits the new data into the MMIT model to give prediction values
#'
#' @param object The Max Margin Interval Tree obtained from \code{"mmit()"}
#' @param newdata an optional data frame containing the testing data which is to be predicted.
#' @param perm an optional character vector of variable names. 
#' @param \dots ...
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
#' tree <- mmit(feature.mat, target.mat)
#' 
#' pred <- predict.mmit(tree)
#' 
predict.mmit <- function(object, newdata = NULL, perm = NULL, ...){
    
  fit <- predict(object, newdata, perm)
  n <- lapply(nodeapply(object, ids = fit, info_node), function(x) x$prediction)
  prediction <- c(matrix(unlist(n), nrow = length(n), byrow = T))
  
  return(prediction)
}