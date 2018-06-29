mmit.predict <- structure(function(tree, newdata = NULL, perm = NULL){
    
  fit <- predict(tree, newdata, perm)
  n <- nodeapply(tree, ids = fit, info_node)
  prediction <- matrix(unlist(n), nrow = length(n), byrow = T)[, 1]
  
  return(prediction)
})