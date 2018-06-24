mmit.pruning <- structure(function(tree){
  ### T1 is the tree after initital pruning of Tmax
  T1 <- init_pruning(tree)
  alpha_trees <- NULL
  
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

    for(n in nrow(wlink):1){
      tree <- nodeprune(tree, ids = wlink[n, 2])
    }
    
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

