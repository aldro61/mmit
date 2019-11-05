#' The Predict Function for Cross Validation of Max Margin Interval Tree
#'
#' Fits the new data into the mmit.cv model to give prediction values
#'
#' @param object Object obtained from \code{"mmit.cv()"}
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
#' fit <- mmit.cv(feature.mat, target.mat)
#' 
#' pred <- predict(fit)
#' 
predict.mmit.cv <- function(object, newdata = NULL, perm = NULL, ...){
  
  prediction <- predict.mmit(object$best_estimator, newdata, perm)
  
  return(prediction)
}