#' Adaboost Prediction for the Max Margin Interval Tree
#'
#' Learning adaptive boosting for the Max Margin Interval Tree
#'
#' @param trees Ensemble of MMITboost trees
#' @param target.mat A data frame containing the features of the examples for which predictions must be computed.
#' 
#' @return Predictions after adaboost
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
#' trees <- mmitboost(target.mat, feature.mat)
#' pred <- mmitboost.predict(trees, feature.mat)
#' 
#' @export
mmitboost.predict <- structure(function(target.mat, mmitboost_results) {
  final_scores <- weighted.mean(mmitboost_results$pred, mmitboost_results$B)
  return(final_scores)
})
