init_pruning <- structure(function(tree){
  
  #id of inner node
  ter_id <- nodeids(tree, terminal = TRUE)
  all_id <- nodeids(tree, terminal = FALSE)
  inner_id <- all_id[-ter_id]
  
  if(length(inner_id)==0){
    return(NULL)
  }
  
  for(n in 1:length(inner_id)){
    node <- tail(inner_id, n=n)[1]
    node_info <- nodeapply(tree, ids=node,info_node)
    node_cost <- as.numeric(node_info[[1]][2])
    left_cost <- as.numeric(nodeapply(tree, ids=sapply(kids_node(node), id_node)[1], info_node)[[1]][2])
    right_cost <- as.numeric(nodeapply(tree, ids=sapply(kids_node(node), id_node)[2], info_node)[[1]][2])
    
    if(isTRUE(all.equal(node_cost, (left_cost+right_cost)))){
      tree <- nodeprune(tree, ids = node)
      print(TRUE)
    }
  }
  
  return(tree)
  
})

weakest_link <- structure(function(root){
  print(tree)
  print("  ")
  find_weakest_link <- function(tree){
    ter_id <- nodeids(tree, terminal = TRUE)
    all_id <- nodeids(tree, terminal = FALSE)
    node_cost <- as.numeric(nodeapply(tree, ids=all_id[1],info_node)[[1]][2])
    
    
    #catch numeric zero of the root
    if(length(node_cost) == 0){
      node_cost <- Inf
    }
    
    if(length(ter_id)==length(all_id)){
      print(c(Inf,tree))
      return(c(Inf,tree))
      
    }
    else{
      
      #terminal leaf info
      n <- nodeapply(tree, ids=ter_id,info_node)
      
      #sum of cost of tree
      C_Tt <- sum(matrix(unlist(n),nrow = length(n), byrow = T)[,2])
      
      current_gt <- (node_cost - C_Tt)/(length(ter_id)-1)
      
      c(left$min_gt,left$weakest_links) <- unlist(find_weakest_link(nodeapply(tree, ids=sapply(kids_node(node), id_node)[1])[[1]]))
      
      c(right$min_gt,right$weakest_links) <- rapply(nodeapply(tree, ids=sapply(kids_node(node), id_node)[2])[[1]],find_weakest_link, how = "unlist")
      
      if(isTRUE(all.equal(current_gt, min(left$min_gt, right$min_gt)))){
        if(isTRUE(all.equal(left_min$gt, right$min_gt))){
          return(c(current_gt, tree, left$weakest_links, right$weakest_links))
        }
        else{
          if(left$min_gt < right$min_gt){
            return(c(current_gt, tree, left$weakest_links ))
          }
          else{
            return(c(current_gt, tree, right$weakest_links ))
          }
        }
      }
      else if(current_gt < min(left$min_gt, right$min_gt)){
        return(c(current_gt, tree))
      }
      else if(isTRUE(all.equal(left_min$gt, right$min_gt))){
        return(c(left$min_gt, left$weakest_links, right$weakest_links))
      }
      else if(left$min_gt > right$min_gt){
        return(c(right$min_gt,right$weakest_links))
      }
      else if(left$min_gt < right$min_gt){
        return(c(left$min_gt,left$weakest_links))
      }
    }}
  return(find_weakest_link(root))
})

weakest_link(tree)
