#' The Predict Function for Cross Validation of Random Forest
#'
#' Fits the new data into the mmif.cv model to give prediction values
#'
#' @param object Object obtained from \code{"mmif.cv()"}
#' @param newdata an optional data frame containing the testing data which is to be predicted.
#' @param perm an optional character vector of variable names. 
#' 
#' @return Predictions Average output of each tree in the forest
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
#' fit <- mmif.cv(feature.mat, target.mat)
#' 
#' pred <- mmif.cv.predict(fit)
#' 
mmif.cv.predict <- function(object, newdata = NULL, perm = NULL){
  
  all_pred <- NULL
  forest <- object$best_estimator
  for(i in 1 : length(forest)){
    prediction <- mmit.predict(forest[[i]], newdata)
    all_pred <- rbind(all_pred, prediction)
  }
  
  return(colMeans(all_pred))
}