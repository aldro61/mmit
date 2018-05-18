growtree <- structure(function(target.mat, feature.mat, depth=0, maxdepth = Inf,margin=0.0,loss="hinge",id = 1L, min_sample = 1) {
  if (depth > maxdepth) return(partynode(id = id))
  if(length(target.mat[1,])<min_sample) return(partynode(id = id))
  
  sp <- bestsplit(target.mat,feature.mat,margin,loss="hinge")
  View(sp)
  sp <- partysplit(varid = sp$varid,br = sp$br)
  
  
  if (is.null(sp)) return(partynode(id = id))
  
  kidids <- kidids_split(sp, data = feature.mat) #list of right left kids
  kids <- vector(mode = "list", length = max(kidids, na.rm = TRUE)) #null why
  #I want a list of elems in left and right********
  
  if(length(kids)<=1) return(partynode(id = id))
  
  for (kidid in 1:length(kids)) {
    if (kidid > 1) {
      myid <- max(nodeids(kids[[kidid - 1]]))
    } else {
      myid <- id
    }
    depth <- depth+1
    kids[[kidid]] <- growtree(target.mat, feature.mat, depth = depth, maxdepth = maxdepth, margin = margin, id = as.integer(myid + 1))
  }
  return(partynode(id = as.integer(id), split = sp, kids = kids,info = list(p.value = min(info_split(sp)$p.value, na.rm = TRUE))))
})


node<- growtree(target.mat, feature.mat, maxdepth = 2, margin = 2.0)




