growtree <- structure(function(target.mat, feature.mat, depth=0, maxdepth = Inf,margin=0.0,loss="hinge",id = 1L, min_sample = 1) {

  #if depth exceeds we stop the further growth of tree
  if (depth > maxdepth) return(partynode(id = id))

  #if sample per node is less than minimum, we stop tree growth
  if(length(target.mat[1,])<min_sample) return(partynode(id = id))

  #split the tree at the node.
  #print(length(feature.mat[,2]))
  sp <- bestsplit(target.mat,feature.mat,margin,loss="hinge")
  sp <- partysplit(varid = sp$varid,br = sp$br)

  #if no split, we stop
  if (is.null(sp)) return(partynode(id = id))

  #the kids left and right defined
  kid<-NULL
  kid$left<-NULL
  kid$right<-NULL
  tar<-NULL
  tar$left<-NULL
  tar$right<-NULL

  #splitting into kids
  kidids <- kidids_split(sp, data = feature.mat) #list of right left kids
  #print(kidids)

  #kid <- kidids_node()

  #creating list of data into
  for(i in 1:length(feature.mat[,2])){
    if(kidids[i]==1){
      kid$left <- rbind(kid$left,feature.mat[i,])
      tar$left <- rbind(tar$left,target.mat[i,])
    }
    else{
      kid$right <- rbind(kid$right,feature.mat[i,])
      tar$right <- rbind(tar$right,target.mat[i,])
    }
  }

  #creating partynode list of kids
  kids <- vector(mode = "list", length = max(kidids, na.rm = TRUE))
  #I want a list of elems in left and right********

  #if no left right kids, stop tree growth
  if(length(kids)<=1) return(partynode(id = id))


  for (kidid in 1:length(kids)) {
    if (kidid > 1) {
      myid <- max(nodeids(kids[[kidid - 1]]))
    } else {
      myid <- id
    }
    depth <- depth+1
    #print(length(tar[[kidid]][,2]))
    kids[[kidid]] <- growtree(tar[[kidid]], kid[[kidid]], depth = depth, maxdepth = maxdepth, margin = margin, id = as.integer(myid + 1))
  }
  return(partynode(id = as.integer(id), split = sp, kids = kids))
})




