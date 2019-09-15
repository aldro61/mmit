#' Predictions with random forests of Max Margin Interval Trees
#' 
#' @param forest Ensemble of MMITs
#' @param test_feature.mat A data frame containing the features of the examples for which predictions must be computed.
#' 
#' @return Predictions Average output of each tree in the forest
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
#' forest <- mmif(feature.mat, target.mat)
#' pred <- predict(forest, feature.mat)
#' 
predict.mmif <- function(forest, test_feature.mat = NULL){
  
  all_pred <- NULL
  for(i in 1 : length(forest)){
    prediction <- predict.mmit(forest[[i]], test_feature.mat)
    all_pred <- rbind(all_pred, prediction)
  }
  
  return(colMeans(all_pred))
  
}
