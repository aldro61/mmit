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
mmitboost <- function(target.mat, feature.mat,   
                                max_depth = Inf, margin=0.0, loss="hinge", min_sample = 1, 
                                weights = rep(1L, nrow(feature.mat))/seq(1L, nrow(feature.mat),1),
                                n_estimators = 100) {
  final_scores <- rep(0., times = nrow(target.mat))
  result <- list()
  for(i in 1:n_estimators){
    ### probability pi
    p <- weights / sum(weights)
    
    ## sample n_examlple elements of dataset using p as probability of occurance
    x <- sample(nrow(feature.mat), nrow(feature.mat), prob = p, replace = TRUE)
    new_feature.mat <- feature.mat[x, ]
    new_target.mat <- target.mat[x,]
    new_target.mat <- data.matrix(new_target.mat)
    weights <- weights[x]
    
    ### regression tree
    tree <- mmit(target.mat, feature.mat, max_depth, margin, loss, min_sample, weights)
    result$trees[[i]] <- tree
    
    ### predictions of the model
    prediction <- mmit.predict(tree, feature.mat)
    result$pred[[i]] <- prediction
    
    ### average loss
    cost <- .compute_loss(target.mat, prediction, margin, loss)
    L <- sum(cost * p)
    ### measure of confidance
    B = L / (1 - L)
    weights <- weights*B**(1-cost)
    result$B[[i]] <- B
    
    if(L < 0.5){
      break
    }
  }
  return(result)
}

.compute_loss <- function(target.mat, prediction, margin, loss){
  lower = target.mat[,1] - prediction + margin
  lower[lower < 0] <- 0
  upper = prediction - target.mat[,2] + margin
  upper[upper < 0] <- 0
  if(loss == 'hinge'){
    cost = lower + upper
  }
  else if(loss == 'square'){
    cost = (lower + upper)**2
  }
  return(cost)
}
