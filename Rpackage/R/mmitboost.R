#' Adaboost for the Max Margin Interval Tree
#'
#' Learning adaptive boosting for the Max Margin Interval Tree
#'
#' @param target.mat The response variable of the model
#' @param feature.mat a data frame containing the feature variables in the model.
#' @param margin margin paramaters 
#' @param loss The type of loss; (\code{"hinge"}, \code{"square"})
#' @param max_depth The maximum depth criteia
#' @param min_sample The minimum number of sample required 
#' 
#' @return List of trees containing each tree in the adaboost method.
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
#' out <- mmitboost(target.mat, feature.mat)
#' 
#' @export
mmitboost <- structure(function(target.mat, feature.mat,  
                           max_depth = Inf, margin=0.0, loss="hinge",
                           min_sample = 1) {}