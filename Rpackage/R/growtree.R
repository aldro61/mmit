growtree <- structure(function(target.mat, feature.mat, depth=0, maxdepth = Inf,
                               margin = 0.0, loss="hinge", id = 1L, min_sample = 1,
                               pred=NULL, side = NULL, weights = NULL) {
  
  pid <- NULL
  ### creating dataframe of info
  if(!is.null(pred)){
    pid <- data.frame(pred[[side]], pred[[1]],row.names = "")
    colnames(pid) <- c("prediction", "cost")
  }

  ### if depth exceeds we stop the further growth of tree
  if (depth >= maxdepth) return(partynode(id = id, info = pid))
  

  ### if sample per node is less than minimum, we stop tree growth
  if(sum(weights) <= min_sample) return(partynode(id = id, info = pid))

  ### split the tree at the node.
  sp <- bestsplit(target.mat, feature.mat, weights, margin, loss, pred)
  splt <- sp

  ### if no split, we stop
  if (is.null(sp)) return(partynode(id = id, info = pid))

  ### partysplit object
  sp <- partysplit(varid = sp$varid, breaks = sp$br)

  ### splitting into kids
  kidids <- kidids_split(sp, data = feature.mat) #list of right left kids

  ### creating partynode list of kids
  kids <- vector(mode = "list", length = max(kidids, na.rm = TRUE))
  
  ### if no left right kids, stop tree growth
  if(length(kids) <= 1) return(partynode(id = id, info = pid))


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
    depth <- depth+1
    
    #side = 4+kidid as pred[[5]] is left prediction and pred[[6]] right
    kids[[kidid]] <- growtree(target.mat, feature.mat, pred = splt, side = 4 + kidid, depth = depth,
                              maxdepth = maxdepth, margin = margin, loss = loss, 
                              id = as.integer(myid + 1),min_sample = min_sample, weights = w)
  }
  
  return(partynode(id = as.integer(id), split = sp, kids = kids, info = pid))
})




