.init_pruning <- function(tree){
  
  ### id of inner node
  ter_id <- nodeids(tree, terminal = TRUE)
  all_id <- nodeids(tree, terminal = FALSE)
  inner_id <- all_id[-ter_id]
  
  ### if there is no inner node
  if(length(inner_id) == 0){
    return(tree)
  }
  
  for(n in 1:length(inner_id)){
    ### node id from the last
    node_id <- tail(inner_id, n = n)[1]
    ### partynode object
    node <- nodeapply(tree, ids = node_id)[[1]]
    
    
    ### left right child ids and partynodes
    kids <- kids_node(node)
    kid_ids <- sapply(kids, id_node)
    left_kid_id <- kid_ids[1]
    right_kid_id <- kid_ids[2]
    
    ### cases where node doesnt have 2 leaves
    if(!(is.terminal(kids[[1]]) && is.terminal(kids[[2]]))){
      next
    }
    
    ### node and child costs
    node_cost <- as.numeric(nodeapply(tree, ids = node_id,info_node)[[1]][2])
    left_cost <- as.numeric(nodeapply(tree, ids = left_kid_id, info_node)[[1]][2])
    right_cost <- as.numeric(nodeapply(tree, ids = right_kid_id, info_node)[[1]][2])
    
    ### check if node cost equal to left + right.
    assert_that(node_cost >= (left_cost + right_cost))
    
    if(isTRUE(all.equal(node_cost, (left_cost + right_cost)))){
      tree <- nodeprune(tree, ids = node_id)
    }
  }
  
  return(tree)
  
}