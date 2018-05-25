growtree <- structure(function(target.mat, feature.mat, depth=0, maxdepth = Inf,
                               margin=0.0, loss="hinge", id = 1L, min_sample = 1,
                               pred=NULL, side = NULL) {

  #if depth exceeds we stop the further growth of tree
  if (depth >= maxdepth) return(partynode(id = id, info = pred[[side]]))
  

  #if sample per node is less than minimum, we stop tree growth
  if(length(target.mat[,1]) <= min_sample) return(partynode(id = id, info = pred[[side]]))

  #split the tree at the node.
  sp <- bestsplit(target.mat, feature.mat, margin, loss)
  splt <- sp

  #if no split, we stop
  if (is.null(sp)) return(partynode(id = id, info = pred[[side]]))

  #partysplit object
  sp <- partysplit(varid = sp$varid, breaks = sp$br)

  #the kids left and right defined
  kid <- NULL
  kid$left <- NULL
  kid$right <- NULL
  tar <- NULL
  tar$left <- NULL
  tar$right <- NULL

  #splitting into kids
  kidids <- kidids_split(sp, data = feature.mat) #list of right left kids

  #creating partynode list of kids
  kids <- vector(mode = "list", length = max(kidids, na.rm = TRUE))
  
  #if no left right kids, stop tree growth
  if(length(kids) <= 1) return(partynode(id = id, info = pred[[side]]))

  #creating list of data into
  for(i in 1:length(feature.mat[,1])){
    if(kidids[i] == 1){
      kid$left <- rbind(kid$left, feature.mat[i,])
      tar$left <- rbind(tar$left, target.mat[i,])
    }
    else{
      kid$right <- rbind(kid$right, feature.mat[i,])
      tar$right <- rbind(tar$right, target.mat[i,])
    }
  }


  for (kidid in 1:length(kids)) {
    if (kidid > 1) {
      myid <- max(nodeids(kids[[kidid - 1]]))
    } else {
      myid <- id
    }
    depth <- depth+1
    kids[[kidid]] <- growtree(tar[[kidid]], kid[[kidid]], pred=splt, side = 4+kidid, depth = depth,
                              maxdepth = maxdepth, margin = margin, loss = loss, 
                              id = as.integer(myid + 1),min_sample = min_sample)
  }
  return(partynode(id = as.integer(id), split = sp, kids = kids, info = pred[[side]]))
})




