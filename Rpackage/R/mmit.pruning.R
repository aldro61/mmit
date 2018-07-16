#' The Pruned Max Margin Interval Tree
#'
#' Pruning the regression tree for censored data.
#'
#' @param tree The fitted tree using \code{"mmit()"} function
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
#' tree <- mmit(target.mat, feature.mat)
#' 
#' pruned_tree <- mmit.pruning(tree)
#' 
#' @export



#### needs work...eturn type



mmit.pruning <- structure(function(tree){
  ### T1 is the tree after initital pruning of Tmax
  T1 <- init_pruning(tree)
  alpha_trees <- list()
  
  sequential_prune <- function(tree){
    ### find weakest links
    wlink <- weakest_link(tree)
    
    ### min_gt is same for all weakest links
    min_gt <- wlink[1, 1]
    
    ### to prune the weakest links from tail of tree we apply sorting
    ### sort solution 
    sorted_wlink <- order(wlink[, 2])
    if(length(sorted_wlink)>1){
      wlink <- wlink[sorted_wlink,]
    }
    
    ### pruning the tree
    tree <- nodeprune(tree, ids = wlink[, 2])
    
    
    ### if terminal root return current tree
    if(is.terminal(nodeapply(tree, ids = 1)[[1]])){
      alpha_trees[[length(alpha_trees)+1]] <- list(alpha = min_gt, tree = tree)
      return(alpha_trees)
    }
    else{
      alpha_trees <- sequential_prune(tree)
      alpha_trees[[length(alpha_trees)+1]] <- list(alpha = min_gt, tree = tree)
      return(alpha_trees)
    }
  }
  
  if(is.terminal(nodeapply(T1, ids = 1)[[1]])){
    alpha_trees[[length(alpha_trees)+1]] <- list(alpha = 0.0, tree = T1)
    return(alpha_trees)
  }
  else{
    alpha_trees <- sequential_prune(T1)
    alpha_trees[[length(alpha_trees)+1]] <- list(alpha = 0.0, tree = T1)
    return(alpha_trees)
  }
  
})

