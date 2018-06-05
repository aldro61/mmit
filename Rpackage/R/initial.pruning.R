init_pruning <- structure(function(tree){
  
  ### id of inner node
  ter_id <- nodeids(tree, terminal = TRUE)
  all_id <- nodeids(tree, terminal = FALSE)
  inner_id <- all_id[-ter_id]
  
  ### if there is no inner node
  if(length(inner_id) == 0){
    return(NULL)
  }
  
  for(n in 1:length(inner_id)){
    ### node id from the last
    node_id <- tail(inner_id, n = n)[1]
    ### partynode object
    t <- nodeapply(tree, ids = node_id)[[1]]
    node <- as.partynode(t)
    
    ### left right child ids and partynodes
    left_kid_id <- sapply(kids_node(node), id_node)[1]
    right_kid_id <- sapply(kids_node(node), id_node)[2]
    left_kid <- nodeapply(tree, ids = left_kid_id)[[1]]
    right_kid <- nodeapply(tree, ids = right_kid_id)[[1]]
    
    ### cases where node doesnt have 2 leaves
    if(!(is.terminal(left_kid) && is.terminal(right_kid))){
      next
    }
    
    ### node and child costs
    node_cost <- as.numeric(nodeapply(tree, ids = node_id,info_node)[[1]][2])
    left_cost <- as.numeric(nodeapply(tree, ids = left_kid_id, info_node)[[1]][2])
    right_cost <- as.numeric(nodeapply(tree, ids = right_kid_id, info_node)[[1]][2])
    
    ### check if node cost equal to left + right.
    if(isTRUE(all.equal(node_cost, (left_cost + right_cost)))){
      tree <- nodeprune(tree, ids = node_id)
    }
  }
  
  return(tree)
  
})