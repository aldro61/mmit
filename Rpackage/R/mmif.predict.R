#' Prediction for Random Forest
#'
#' Predictions as average of all predictions of the random forest.
#' 
#' @param forest Ensembe of forest
#' @param test_feature.mat a data frame containing the test feature variables in the model.
#' 
#' @return Prediction values
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
#' forest <- mmif(target.mat, feature.mat)
#' pred <- mmif.predict(forest, feature.mat)
#' 
#' @export
mmif.predict <- structure(function(forest, test_feature.mat = NULL){
  
  all_pred <- NULL
  for(i in 1 : length(forest)){
    prediction <- mmit.predict(forest[[i]], test_feature.mat)
    all_pred <- rbind(all_pred, prediction)
  }
  
  return(colMeans(all_pred))
  
})