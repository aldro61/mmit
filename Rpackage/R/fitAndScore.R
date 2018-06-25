fit_and_score <- structure(function(tree, target.mat, feature.mat, 
                                    parameters, feature_names = NULL, 
                                    n_folds = 3, scorer = NULL, loss = "hinge",
                                    pruning = TRUE){
  ### total length
  l <- nrow(target.mat)
  
  ### shuffle the data 
  rand <- sample(nrow(target.mat))
  target.mat <- target.mat[rand,]
  feature.mat <- feature.mat[rand,]
  
  
  ### Create n equally size folds
  folds <- cut(seq(1,nrow(target.mat)),breaks=n_folds,labels=FALSE)
  
  ### train and test index for each fold
  fold_split_idx <- NULL
  
  ### segment the data into test and train
  for(i in 1:n_folds){
    #Segement your data by fold using the which() function 
    fold_split_idx$test <- which(folds==i,arr.ind=TRUE)
    fold_split_idx$train <- which(folds!=i,arr.ind=TRUE)
    
  }
  
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
    master_data <- mmit.pruning(master_tree)
    master_alphas <- lapply(master_data, function(x) x$alpha)
    master_pruned_trees <- lapply(master_data, function(x) x$tree)
    
    fold_alphas <- NULL
    fold_prune_trees <- NULL
    for(i in 1 : n_folds){
      fold_data <- mmit.pruning(fold_tree[[i]])
      fold_alphas[[i]] <- lapply(fold_data, function(x) x$alpha)
      fold_prune_trees[[i]] <- lapply(fold_data, function(x) x$tree)
    }
    
    ### alphas list should not contain repeating alpha
    master_alphas <- master_alphas[!duplicated(master_alphas)]
    fold_alphas <- fold_alphas[!duplicated(fold_alphas)]
    
    
    # Compute the test risk for all pruned trees of each fold
    alpha_path_score <- NULL
    for(i in 1 : n_folds){
      for(j in 1 : nrow(fold_prune_trees[[i]])){
        ### convert pruned tree list to partynode
        node <- fold_prune_trees[[i]][[j]]
        
        fit <- fitted_node(node, feature.mat[fold_split_idx$test[i,],])
        n <- nodeapply(node, ids = fit, info_node)
        print("1")  #### if node = root*
        ###prediction of test data
        prediction <- matrix(unlist(n), nrow = length(n), byrow = T)[,1]
        ###error calc
        fold_test_scores <- scorer(target.mat[fold_split_idx$test[i,],], prediction)
        
        ### creating a dataframe with init alpha, final alpha, score value
        if(j < (length(fold_alphas[[i]]))){
          alpha_path_score <- rbind(alpha_path_score, c(fold_alphas[[i]][j], fold_alphas[[i]][j + 1], fold_test_scores))
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
    
    for(i in 1 : length(master_pruned_trees[,1])){
      if(i < (length(master_alphas))){
        geo_mean_alpha_k <- sqrt(master_alphas[[i]] * master_alphas[[i + 1]])
      }
      else{
        geo_mean_alpha_k <- Inf
      }
      
      ### compute cv_score as mean of each fold scores
      cv_score <- 0
      for(i in 1 : n_folds){
        for(j in 1 : nrow(alpha_path_score)){
          if((geo_mean_alpha_k < alpha_path_score[j, 2]) && (geo_mean_alpha_k >= alpha_path_score[j, 1])){
            cv_score <- cv_score + alpha_path_score[j, 3]
          }
        }
      }
      
      ### calc train score
      fit <- predict(master_tree, feature.mat)
      n <- nodeapply(master_tree, ids = fit, info_node)
      prediction <- matrix(unlist(n), nrow = length(n), byrow = T)[, 1]
      train_score <- scorer(target.mat, prediction)
      
      ### calc cost of all leaves
      ter_id <- nodeids(master_tree, terminal = TRUE)
      n <- nodeapply(master_tree, ids = ter_id, info_node)
      train_objective <- sum(matrix(unlist(n), nrow = length(n), byrow = T)[, 2])
      
      # Log metrics for this alpha value
      alphas <- c(alphas, geo_mean_alpha_k)
      alpha_cv_scores <- c(alpha_cv_scores, cv_score)
      alpha_train_scores <- c(alpha_train_scores, train_score)
      alpha_train_objective_values <- c(alpha_train_objective_values, train_objective)
      
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
    for(i in 1 : n_folds){
      fit <- predict(fold_tree[[i]], feature.mat[fold_split_idx$test[i,],])
      n <- nodeapply(fold_tree[[i]], ids = fit, info_node)
      prediction <- matrix(unlist(n), nrow = length(n), byrow = T)[,1]
      fold_test_scores <- c(fold_test_scores, scorer(target.mat, prediction))
    }
    
    ### master tree predictions
    fit <- predict(master_tree, feature.mat)
    n <- nodeapply(master_tree, ids = fit,info_node)
    prediction <- matrix(unlist(n), nrow = length(n), byrow = T)[, 1]
    master_scores <- scorer(target.mat, prediction)
      
    best_alpha <-  0.
    best_score <- mean(fold_test_scores)
    best_tree <- master_tree
    alphas <- c(0.)
    alpha_cv_scores <- c(best_score)
    alpha_train_scores <- c(master_scores)
    
    ### calc cost of all leaves
    ter_id <- nodeids(master_tree, terminal = TRUE)
    n <- nodeapply(master_tree, ids = ter_id, info_node)
    alpha_train_objective_values <- sum(matrix(unlist(n), nrow = length(n), byrow = T)[, 2])
  }
  ### Append alpha to the parameters
  best_params <- parameters
  best_params$alpha <- best_alpha
  
  ### Generate a big dictionnary of all HP combinations considered (including alpha) and their CV scores
  cv_results <- cbind(parameters$maxdepth, parameters$margin, parameters$min_sample, 
                      alphas, alpha_cv_scores, alpha_train_scores, alpha_train_objective_values)
  colnames(cv_results) <- c("maxdepth", "margin", "min sample","alpha", " cv score", "train score", "train objective vale")
  cv_results <- as.data.frame(cv_results)
  
  output <- NULL
  output$best_score <- best_score
  output$best_estimator <- best_tree
  output$best_params <- best_params
  output$cv_result <- cv_results
  
  
  return(output)
})
    
