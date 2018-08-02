#' Random Forest of Max Margin Interval Tree
#'
#' Performing random forest on ensemble of Max Margin Interval Tree.
#' 
#' @param target.mat The response variable of the model
#' @param feature.mat a data frame containing the feature variables in the model.
#' @param margin margin paramaters 
#' @param loss The type of loss; (\code{"hinge"}, \code{"square"})
#' @param max_depth The maximum depth criteia
#' @param min_sample The minimum number of sample required 
#' @param n_trees The Number of trees
#' @param n_features The number of features to be sampled.
#' @param n_cpu The number of cores to register for parallel programing of the code, default value is 1 and n_cpu = -1 to select all cores.
#' 
#' @return List of ensemble of trees.
#' 
#' @author Toby Dylan Hocking, Alexandre Drouin, Torsten Hothorn, Parismita Das
#' 
#' @examples
#' library(mmit)
#' target.mat <- rbind(
#'   c(0,1), c(0,1), c(0,1),
#'   c(2,3), c(2,3), c(2,3))
#' 
#' feature.mat <- rbind(
#'   c(1,0,0), c(1,1,0), c(1,2,0),
#'   c(1,3,0), c(1,4,0), c(1,5,0))
#' 
#' colnames(feature.mat) <- c("a", "b", "c")
#' feature.mat <- data.frame(feature.mat)
#' 
#' trees <- mmif(target.mat, feature.mat, margin = 2.0, n_cpu = -1)
#' 
#' @export
mmif <- structure(function(target.mat, feature.mat, 
                           max_depth = Inf, margin=0.0, loss="hinge",
                           min_sample = 1, n_trees = 10,
                           n_features = as.integer(ncol(feature.mat)**0.5), n_cpu = 1){
 
 
  ### parallelize using foreach, see all permutation combination of param grid values
  ### register parallel backend
  if(n_cpu == -1) n_cpu <- detectCores() 
  assert_that(detectCores() >= n_cpu)
  
  all_trees <- list()
  if(n_cpu > 1){
    cl <- makeCluster(n_cpu)
    registerDoParallel(cl)
    
    ### create n_trees
    all_trees <- foreach(i = 1 : n_trees, .packages = "mmit") %dopar% 
      random_tree(target.mat, feature.mat, 
                  max_depth = max_depth, margin = margin, loss = loss,
                  min_sample = min_sample, n_trees = n_trees,
                  n_features = n_features)
    
    
    stopCluster(cl)
  }
  else{
    for(i in 1 : n_trees) {
      all_trees[[i]] = random_tree(target.mat, feature.mat, 
                                   max_depth = max_depth, margin = margin, loss = loss,
                                   min_sample = min_sample, n_trees = n_trees,
                                   n_features = n_features)
    }
      
  }
  
  return(all_trees)
  
}, ex=function(){
  
  data(neuroblastomaProcessed, package="penaltyLearning")
  feature.mat <- data.frame(neuroblastomaProcessed$feature.mat)[1:45,]
  target.mat <- neuroblastomaProcessed$target.mat[1:45,]
  trees <- mmif(target.mat, feature.mat, max_depth = Inf, margin = 2.0, loss = "hinge", min_sample = 1)

})


random_tree <- function(target.mat, feature.mat, 
                          max_depth = Inf, margin=0.0, loss="hinge",
                          min_sample = 1, n_trees = 10,
                          n_features = as.integer(ncol(feature.mat)**0.5)){
      
  ### sample n_exalple elements of dataset
  x <- sample(nrow(feature.mat), nrow(feature.mat), replace = TRUE)
  new_feature.mat <- feature.mat[x, ]
  new_target.mat <- target.mat[x,]
  new_target.mat <- data.matrix(new_target.mat)
  
  ### if default n_feature value is 1 and no of columns is greater than 1
  if((as.integer(ncol(feature.mat)**0.5) <= 1) && (n_features <= 1) && (ncol(feature.mat) > 1)){
    n_features = 2
  }
  
  ### if user assigned n_feature value is one
  assert_that(n_features > 1)
  assert_that(ncol(feature.mat) >= n_features)
  
  ### sample features
  y <- sample(ncol(feature.mat), n_features)
  new_feature.mat <- new_feature.mat[, y]
  colnames(new_feature.mat) <- c(names(feature.mat)[y])
  new_feature.mat <- data.frame(new_feature.mat)
  
  ### tree
  tree <- mmit(new_target.mat, new_feature.mat, margin = margin, loss = loss, 
               min_sample = min_sample, max_depth = max_depth)
  return(tree)
}
