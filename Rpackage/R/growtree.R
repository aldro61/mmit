growtree <- structure(function(target.mat, feature.mat, depth=0, maxdepth = Inf,margin=0.0,loss="hinge",id = 1L, min_sample = 1) {
  if (depth > maxdepth) return(partynode(id = id))
  if(length(target.mat[1,])<min_sample) return(partynode(id = id))
  
  sp <- bestsplit(target.mat,feature.mat,margin,loss="hinge")
  View(sp)
  sp <- partysplit(varid = sp$varid,br = sp$br)
  
  
  if (is.null(sp)) return(partynode(id = id))
  
  kidids <- kidids_split(sp, data = feature.mat) #list of right left kids
  kids <- vector(mode = "list", length = max(kidids, na.rm = TRUE)) #null why
  
  if(length(kids)<=1) return(partynode(id = id))
  
  for (kidid in 1:length(kids)) {
    if (kidid > 1) {
      myid <- max(nodeids(kids[[kidid - 1]])) #null for id=2
    } else {
      myid <- id  #id = 1L
    }
    depth <- depth+1
    kids[[kidid]] <- growtree(target.mat, feature.mat, depth = depth, maxdepth = maxdepth, margin = margin, id = as.integer(myid + 1))
    #If I give target and feature as response and data, next node splits on same as its not subtree
  }
  return(partynode(id = as.integer(id), split = sp, kids = kids,info = list(p.value = min(info_split(sp)$p.value, na.rm = TRUE))))
})


node<- growtree(target.mat, feature.mat, maxdepth = 2, margin = 2.0)




