mmit <- structure(function(formula, target.mat, feature.mat, weights = NULL,depth=0, maxdepth = Inf,margin=0.0,loss="hinge",id = 1L, min_sample = 1) {
  
  #arranging such that response is column 1.
  response <- all.vars(formula)[1]
  feature.mat <- model.frame(formula, data = feature.mat)
  
  #tree
  node<- growtree(target.mat, feature.mat, maxdepth = maxdepth, margin = margin)
  
  ## compute terminal node number for each observation
  fitted <- fitted_node(node, data = feature.mat)
  
  tree <- party(node, data = feature.mat,
               fitted = data.frame("(fitted)" = fitted,
                                   "(response)" = feature.mat[[response]]))
               #terms = terms(formula))

  return(tree)
})
