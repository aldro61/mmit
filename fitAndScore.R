fit_and_score <- structure(function(tree, target.mat, feature.mat, 
                                    parameters, feature_names=NULL, 
                                    n_folds = 3, scorer=NULL, loss = "hinge",
                                    pruning = TRUE){
  ### total length
  l <- length(target.mat[, 1])
  
  ### train and test index for each fold
  fold_split_idx <- NULL
  
  ### initial row
  fold_split_idx$train <- seq(from = 1, to = ((l/n_folds)*(n_folds-1)), by = 1)
  fold_split_idx$test <- seq(from = ((l/n_folds)*(n_folds-1)) + 1, to = l, by = 1)
  
  ### middle values
  for(n in (n_folds-1):2){
    this_row_train <- c(seq(from = 1, to = ((l/n_folds)*(n-1)), by = 1), seq(from = as.integer((l/n_folds)*(n)+1), to = l, by = 1))
    this_row_test <- seq(from = ((l/n_folds)*(n-1)) + 1, to = as.integer((l/n_folds)*(n)))
    
    fold_split_idx$train <- rbind(fold_split_idx$train, this_row_train)
    fold_split_idx$test <- rbind(fold_split_idx$test, this_row_test)
  }
  
  ### end row
  this_row_train <- seq(from = (as.integer((l/n_folds)+1)), to = l, by = 1)
  this_row_test <- seq(from = 1, to = as.integer((l/n_folds)), by = 1)
  
  fold_split_idx$train <- rbind(fold_split_idx$train, this_row_train)
  fold_split_idx$test <- rbind(fold_split_idx$test, this_row_test)
  
  ### fold trees
  fold_tree <- NULL
  for(i in 1:n_folds){
    fold_tree[[i]] <- mmit(target.mat[fold_split_idx$train[i,],], feature.mat[fold_split_idx$train[i,],],  
                                       maxdepth = as.numeric(parameters$maxdepth), margin = as.numeric(parameters$margin), 
                                       loss = loss, min_sample = as.numeric(parameters$min_sample))
  }
  
  ### master tree
  master_tree <- mmit(target.mat, feature.mat,  
                      maxdepth = as.numeric(parameters$maxdepth), margin = as.numeric(parameters$margin), 
                      loss = loss, min_sample = as.numeric(parameters$min_sample))
  
  
  ### if pruning
  if(pruning){
    # Get the pruned master and cross-validation trees
    master_data <- pruning(master_tree)
    master_alphas <- master_data[, 1]
    master_pruned_trees <- master_data[, -1]
    
    fold_alphas <- NULL
    fold_prune_trees <- NULL
    for(i in 1 : n_folds){
      fold_data <- pruning(fold_tree[[i]])
      fold_alphas[[i]] <- fold_data[, 1] 
      fold_prune_trees[[i]] <- fold_data[, -1]
    }
    
    ### alphas list should not contain repeating alpha
    master_alphas <- master_alphas[!duplicated(master_alphas)]
    fold_alphas <- fold_alphas[!duplicated(fold_alphas)]
    
    
    # Compute the test risk for all pruned trees of each fold
    alpha_path_score <- NULL
    for(i in 1 : n_folds){
      for(j in 1 : length(fold_prune_trees[[i]][,1])){
        ### convert pruned tree list to partynode
        node <- fold_prune_trees[[i]][[j]]
        
        fit <- fitted_node(node, feature.mat[fold_split_idx$test[i,],])
        n <- nodeapply(node, ids = fit, info_node)
        ###prediction of test data
        prediction <- matrix(unlist(n), nrow = length(n), byrow = T)[,1]
        ###error calc
        fold_test_scores <- scorer(target.mat[fold_split_idx$test[i,],], prediction)
        
        ### creating a dataframe with init alpha, final alpha, score value
        if(j < (length(fold_alphas[[i]]) - 1)){
          alpha_path_score <- rbind(alpha_path_score, c(fold_alphas[[i]][j], fold_alphas[[i]][j+1], fold_test_scores))
        }
        else{
          alpha_path_score <- rbind(alpha_path_score, c(fold_alphas[[i]][j], Inf, fold_test_scores))
        }
        colnames(alpha_path_score) <- c("init alpha", " final alpha", "score")
        alpha_path_score <- as.data.frame(alpha_path_score)
      }
      
    }
    
    # Prune the master tree based on the CV estimates
    alphas <- NULL
    alpha_cv_scores <- NULL
    alpha_train_scores <- NULL
    alpha_train_objective_values <- NULL
    best_alpha <- -1
    best_score <- Inf
    best_tree <- NULL
    
    for(i in 1:length(master_pruned_trees[,1])){
      if(i < (length(master_alphas) - 1)){
        geo_mean_alpha_k <- sqrt(master_alphas[[i]]* master_alphas[[i + 1]])
      }
      else{
        geo_mean_alpha_k <- Inf
      }
      
      
      ############# incomplete, figuring out how to find where geo_mean_alpha_k lies
      cv_score <- mean(alpha_path_score[, geo_mean_alpha_k])
      #train_score <- scorer(target.mat, prediction)
      #train_objective <- cost of all nodes  
      
      if(cv_score < best_score){
        best_score <- cv_score
        best_alpha <- geo_mean_alpha_k
        best_tree <- master_pruned_trees[[i]]
      }
    }
    
  }
  else{
    ### For each fold, build a decision tree
    fold_test_scores <- NULL
    for(i in 1:n_folds){
      fit <- fitted_node(fold_tree[[i]], feature.mat[fold_split_idx$test[i,],])
      n <- nodeapply(tree, ids = fit,info_node)
      prediction <- matrix(unlist(n), nrow = length(n), byrow = T)[,1]
      fold_test_scores <- c(fold_test_scores, scorer(target.mat, prediction))
    }
    
    ### master tree predictions
    fit <- predict(master_tree, feature.mat)
    n <- nodeapply(tree, ids = fit,info_node)
    prediction <- matrix(unlist(n), nrow = length(n), byrow = T)[,1]
    master_scores <- scorer(target.mat, prediction)
      
    best_alpha <-  0.
    best_score <- rowMeans(fold_test_scores)
    best_tree <- master_tree
    alphas <- c(0.)
    alpha_cv_scores <- c(best_score)
    alpha_train_scores <- c(master_scores)
    #alpha_train_objective_values < c(l.cost_value for l in master_predictor.tree_.leaves)
  }
  ### Append alpha to the parameters
  best_params <- parameters
  best_params$alpha <- best_alpha
  
  ########## return statement not done yet
  
})
    