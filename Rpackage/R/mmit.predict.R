mmit.predict <- structure(function(tree, newdata = NULL, perm = NULL){
    
  fit <- predict(tree, newdata, perm)
  n <- lapply(nodeapply(tree, ids = fit, info_node), function(x) x$prediction)
  prediction <- c(matrix(unlist(n), nrow = length(n), byrow = T))
  
  return(prediction)
})