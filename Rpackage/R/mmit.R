mmit <- structure(function(formula, target.mat, feature.mat, weights = NULL,
                           depth=0, maxdepth = Inf, margin=0.0, loss="hinge",
                           id = 1L, min_sample = 1) {
  
  #arranging such that response is not present in feature.mat.
  response <- all.vars(formula)[1]
  feature.mat <- model.frame(formula, data = feature.mat)
  if(response %in% colnames(feature.mat)){
    feature.mat$response <- NULL 
  }
  
  #tree
  node<- growtree(target.mat, feature.mat, maxdepth = maxdepth, margin = margin)
  
  ## compute terminal node number for each observation
  tree <- party(node, data = feature.mat, fitted = data.frame("(fitted)" = fitted_node(node, data = feature.mat),
                                                          "(response)" = model.response(feature.mat),
                                                          check.names = FALSE), terms = terms(feature.mat))

  return(tree)
})


#example
#library(survival)
#data(neuroblastomaProcessed, package="penaltyLearning")
#survTarget <- with(neuroblastomaProcessed, {Surv(target.mat[, 1], target.mat[,2], type="interval2")})
#feature.mat <- data.frame(log.penalty=survTarget, neuroblastomaProcessed$feature.mat)[1:45,]
#target.mat <- neuroblastomaProcessed$target.mat[1:45,]
#tree <- mmit(log.penalty~.,target.mat, feature.mat, maxdepth = 2,margin = 1.0)
#plot(tree)
