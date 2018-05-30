mmit <- structure(function(formula, target.mat, feature.mat, weights = NULL,
                           depth=0, maxdepth = Inf, margin=0.0, loss="hinge",
                           id = 1L, min_sample = 1) {
  
  ### arranging such that response is not present in feature.mat.
  response <- all.vars(formula)[1]
  feature.mat <- model.frame(formula, data = feature.mat)
  if(response %in% colnames(feature.mat)){
    feature.mat[[1]] <- NULL 
  }
  
  ### assigning all weights as 1 in the beginning. weights determine which data is to be considered in which node.
  if (is.null(weights)) weights <- rep(1L, nrow(feature.mat))
  stopifnot(length(weights) == nrow(feature.mat) & max(abs(weights - floor(weights))) < .Machine$double.eps)
  
  ### tree
  node<- growtree(target.mat, feature.mat, maxdepth = maxdepth, margin = margin, weights = weights)
  
  ### compute terminal node number for each observation
  tree <- party(node, data = feature.mat, fitted = data.frame("(fitted)" = fitted_node(node, data = feature.mat),
                                                          "(response)" = model.response(feature.mat),
                                                          "(weights)" = weights,
                                                          check.names = FALSE), terms = terms(feature.mat))

  return(tree)
})


