growtree <- structure(function(target.mat, feature.mat, depth = 0, max_depth = Inf,
                               margin = 0.0, loss="hinge", id = 1L, min_sample = 1,
                               pred=NULL, side = NULL, weights = NULL, type = "none") {
  
  node_info_print <- NULL
  ### creating dataframe of info
  if(!is.null(pred)){
    ###left
    if(side == 1){
      node_info_print <- data.frame(pred$leftpred, pred$leftcost, row.names = "")
      colnames(node_info_print) <- c("prediction", "cost")
    }
    else{
      node_info_print <- data.frame(pred$rightpred, pred$rightcost, row.names = "")
      colnames(node_info_print) <- c("prediction", "cost")
    }
    
  }
  
  ### if node is root.
  if(is.null(side)){
    ### calculate root node cost (as no split thus leftcost + rightcost = 0 + cost )
    node_info_print <- compute_optimal_costs(target.mat, margin, loss, type, weights)
    
    node_info_print <- cbind(node_info_print[[2]], node_info_print[[3]])
    node_info_print <- node_info_print[nrow(node_info_print),]
    node_info_print <- data.frame(node_info_print[1], node_info_print[2], row.names = "")
    colnames(node_info_print) <- c("prediction", "cost")
  }
  
  ### if depth exceeds we stop the further growth of tree
  if (depth >= max_depth) return(partynode(id = id, info = node_info_print))
  

  ### if sample per node is less than minimum, we stop tree growth
  if(sum(weights) <= min_sample) return(partynode(id = id, info = node_info_print))

  ### split the tree at the node.
  sp <- bestsplit(target.mat, feature.mat, weights, margin, loss, node_info_print, type)
  splt <- sp
  
  ### if no split, we stop
  if (is.null(sp)) return(partynode(id = id, info = node_info_print))
  
  ### tree cant grows if splitting cost is more than current node cost
  assert_that(node_info_print$cost >= (sp$leftcost + sp$rightcost))
  
  ### partysplit object
  sp <- partysplit(varid = sp$varid, breaks = sp$br)

  ### splitting into kids
  kidids <- kidids_split(sp, data = feature.mat) #list of right left kids

  ### creating partynode list of kids
  kids <- vector(mode = "list", length = max(kidids, na.rm = TRUE))

  for (kidid in 1:length(kids)) {
    ### select observations for current node
    w <- weights
    w[kidids != kidid] <- 0
    
    ### get next node id
    if (kidid > 1) {
      myid <- max(nodeids(kids[[kidid - 1]]))
    } else {
      myid <- id
    }
    
    # 1 is left side, 2 is right side 
    kids[[kidid]] <- growtree(target.mat, feature.mat, pred = splt, side = kidid, depth = depth + 1,
                              max_depth = max_depth, margin = margin, loss = loss, 
                              id = as.integer(myid + 1), min_sample = min_sample, weights = w)
  }
  
  return(partynode(id = as.integer(id), split = sp, kids = kids, info = node_info_print))
})




