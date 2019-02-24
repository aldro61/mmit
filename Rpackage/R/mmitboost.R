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
#' @param weights An importance weight for each learning example, (default = 1)
#' @param M An integer for number of iterations of weight update.
#' 
#' @return Predicted Training Data
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
                                max_depth = Inf, margin=0.0, loss="hinge", min_sample = 1, 
                                weights = rep(1L, nrow(feature.mat))/seq(1L, nrow(feature.mat),1),
                                M = 10) {
  final_scores <- rep(0., times = nrow(target.mat))
  trees <- list()
  for(i in 1:M){
    ### regression tree
    tree <- mmit(target.mat, feature.mat, max_depth, margin, loss, min_sample, weights)
    trees[[i]] <- tree
    ### predictions of the model
    prediction <- mmit.predict(tree, feature.mat)
    if(i==1){
      final_scores <- prediction
    }
    
    ###error calc
    scores <- 0.0
    for(i in 1 : length(prediction)){
      
      if(target.mat[i,1] > prediction[i]){
        scores[i] <- (target.mat[i,1] - prediction[i])**2
      }
      else if(target.mat[i,2] <= prediction[i]){
        scores[i] <- (target.mat[i,2] - prediction[i])**2
      }
      else{
        scores[i] <- 0.0
      }
    }
    if(all(scores == 0)){
      break
    }
    error <- sum(weights*scores)/sum(weights)
    alpha <- 0.5 * log( (1. - error) /error)
    weights <- weights*exp(alpha*scores)
    final_scores <- final_scores + alpha*prediction
  }
  return(trees)
}, ex=function(){
  
  data(neuroblastomaProcessed, package="penaltyLearning")
  feature.mat <- data.frame(neuroblastomaProcessed$feature.mat)[1:45,]
  target.mat <- neuroblastomaProcessed$target.mat[1:45,]
  pred <- mmitboost(target.mat, feature.mat, max_depth = Inf, margin = 2.0, weights = rep(1L, nrow(feature.mat)))
  
})
