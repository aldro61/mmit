weakest_link <- structure(function(tree){
  tree <- nodeapply(tree, ids = 1)[[1]]
  
  find_weakest_link <- function(tree){
    ### get node id and cost of the current node
    ter_id <- nodeids(tree, terminal = TRUE)
    all_id <- nodeids(tree, terminal = FALSE)
    node_cost <- as.numeric(nodeapply(tree, ids = all_id[1], info_node)[[1]][2])
    node_id <- all_id[1]
    
    ### catch numeric zero of the root
    if(length(node_cost) == 0){
      node_cost <- Inf
    }
    
    ### leaf reached
    if(is.terminal(tree)){
      return(rbind(c(Inf, node_id)))
    }
    else{
      
      ### terminal leaf info
      n <- nodeapply(tree, ids = ter_id, info_node)
      
      ### sum of cost of tree
      C_Tt <- sum(matrix(unlist(n), nrow = length(n), byrow = T)[,2])
      
      ### current alpha value
      current_gt <- (node_cost - C_Tt)/(length(ter_id) - 1)
      
      ### left right child ids and partynodes
      left_kid_id <- sapply(kids_node(tree), id_node)[1]
      right_kid_id <- sapply(kids_node(tree), id_node)[2]
  
      ### weakest link from left tree part
      soln <- find_weakest_link(nodeapply(tree, ids = left_kid_id)[[1]])
      left_min_gt <- tail(soln)[1]
      left_weakest_links <- tail(soln)[2]
      
      ### weakest link from right tree part
      soln <- find_weakest_link(nodeapply(tree, ids = right_kid_id)[[1]])
      right_min_gt <- tail(soln)[1]
      right_weakest_links <- tail(soln)[2]
      
      if(isTRUE(all.equal(current_gt, min(left_min_gt, right_min_gt)))){
        ### if all gt same
        if(isTRUE(all.equal(left_min_gt, right_min_gt))){
          soln <- rbind(c(current_gt, node_id), c(current_gt, left_weakest_links), c(current_gt, right_weakest_links))
          return(soln)
        }
        else{
          ### if left gt lesser but equal to curentgt
          if(left_min_gt < right_min_gt){
            soln <- rbind(c(current_gt, node_id), c(current_gt, left_weakest_links))
            return(soln)
          }
          else{
            ### if right gt lesser but equal to curentgt
            soln <- rbind(c(current_gt, node_id), c(current_gt, right_weakest_links))
            return(soln)
          }
        }
      }
      ### if current gt lesser than child's
      else if(current_gt < min(left_min_gt, right_min_gt)){
        soln <- rbind(c(current_gt, node_id))
        return(soln)
      }
      ### if current gt larger than child's and child's gt same
      else if(isTRUE(all.equal(left_min_gt, right_min_gt))){
        soln <- rbind(c(left_min_gt, node_id), c(left_min_gt, right_weakest_links))
        return(soln)
      }
      ### if current gt larger than child's and left > right
      else if(left_min_gt > right_min_gt){
        soln <- rbind(c(right_min_gt, right_weakest_links))
        return(soln)
      }
      ### if current gt larger than child's and left < right
      else if(left_min_gt < right_min_gt){
        soln <- rbind(c(left_min_gt, left_weakest_links))
        return(soln)
      }
    } 
  }
  return(find_weakest_link(tree))
})

