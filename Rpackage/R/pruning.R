pruning <- structure(function(tree){
  ### T1 is the tree after initital pruning of Tmax
  T1 <- init_pruning(tree)
  T1 <- nodeapply(T1, ids = 1)[[1]]
  
  sequential_prune <- function(tree){
    ### find weakest links
    soln <- weakest_link(tree)
    
    ### min_gt is same for all weakest links
    mit_gt <- soln[1, 1]
    
    ### prune the weakest links from tail of tree
    ### sort solution
    if(length(order(soln[, 2]))>1){
      soln <- soln[order(soln[, 2]),]
    }
    print(soln)
    for(n in length(soln[, 2]):1){
      ### ignore non int value tell than 1
      if(soln[n, 2]<1){
        soln[n, 2] <- 1
      }
      tree <- nodeprune(tree, ids = soln[n, 2])
    }
    
    ### if terminal root return current tree
    if(is.terminal(tree)){
      return(rbind(c(mit_gt, tree)))
    }
    else{
      return(rbind(c(mit_gt, tree), sequential_prune(tree)))
    }
  }
  
  if(is.terminal(T1)){
    return(rbind(c(0.0, T1)))
  }
  else{
    return(rbind(c(0.0, T1), sequential_prune(T1)))
  }
  
})

