pruning <- structure(function(tree){
  ### T1 is the tree after initital pruning of Tmax
  T1 <- init_pruning(tree)
  
  sequential_prune <- function(tree){
    ### find weakest links
    wlink <- weakest_link(tree)
    
    ### min_gt is same for all weakest links
    mit_gt <- wlink[1, 1]
    
    ### to prune the weakest links from tail of tree we apply sorting
    ### sort solution 
    sorted_wlink <- order(wlink[, 2])
    if(length(sorted_wlink)>1){
      wlink <- wlink[sorted_wlink,]
    }

    for(n in length(wlink[, 2]):1){
      tree <- nodeprune(tree, ids = wlink[n, 2])
    }
    
    ### if terminal root return current tree
    if(is.terminal(nodeapply(tree, ids = 1)[[1]])){
      return(rbind(c(mit_gt, tree)))
    }
    else{
      return(rbind(c(mit_gt, tree), sequential_prune(tree)))
    }
  }
  
  if(is.terminal(nodeapply(T1, ids = 1)[[1]])){
    return(rbind(c(0.0, T1)))
  }
  else{
    return(rbind(c(0.0, T1), sequential_prune(T1)))
  }
  
})

