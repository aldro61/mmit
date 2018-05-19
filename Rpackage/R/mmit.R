mmit <- structure(function(formula, target.mat, feature.mat, weights = NULL,depth=0, maxdepth = Inf,margin=0.0,loss="hinge",id = 1L, min_sample = 1) {
  node<- growtree(target.mat, feature.mat, maxdepth = 2, margin = 2.0)
  tree <- party(node, data = feature.mat)

  return(tree)
})

tree <- mmit(log.penalty~.,target.mat, feature.mat, margin = 5.0)
plot(tree)
