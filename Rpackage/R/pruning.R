pruning <- structure(function(tree){
  ### T1 is the tree after initital pruning of Tmax
  T1 <- init_pruning(tree)
  
  sequential_prune <- function(tree){
    ### find weakest links
    wlink <- weakest_link(tree)
    
    ### min_gt is same for all weakest links
    mit_gt <- wlink[1, 1]
    
    ### prune the weakest links from tail of tree
    ### sort solution
    if(length(order(wlink[, 2]))>1){
      wlink <- wlink[order(wlink[, 2]),]
    }
    print(wlink)
    for(n in length(wlink[, 2]):1){
      ### ignore non int value less than 1
      if(wlink[n, 2]<1){
        wlink[n, 2] <- 1
      }
      tree <- nodeprune(tree, ids = wlink[n, 2])
    }
    
    ### party object to partynode (tree1 and tree are equal)
    tree1 <- nodeapply(tree, ids = 1)[[1]]
    
    ### if terminal root return current tree
    if(is.terminal(tree1)){
      return(rbind(c(mit_gt, tree)))
    }
    else{
      return(rbind(c(mit_gt, tree), sequential_prune(tree)))
    }
  }
  
  ### party object to partynode (tree1 and tree are equal)
  Tree1 <- nodeapply(T1, ids = 1)[[1]]
  
  if(is.terminal(Tree1)){
    return(rbind(c(0.0, T1)))
  }
  else{
    return(rbind(c(0.0, T1), sequential_prune(T1)))
  }
  
})

