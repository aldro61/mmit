fit_and_score <- structure(function(target.mat, feature.mat, 
                                    parameters, n_folds = 3, scorer = NULL, 
                                    pruning = TRUE){
  
  ### shuffle the data 
  rand <- sample(nrow(target.mat))
  target.mat <- target.mat[rand,]
  feature.mat <- feature.mat[rand,]
  
  
  ### Create n equally size folds
  folds <- cut(seq(1,nrow(target.mat)), breaks = n_folds, labels = FALSE)
  
  ### train and test index for each fold
  fold_split_idx <- NULL
  
  ### segment the data into test and train
  for(i in 1:n_folds){
    #Segement your data by fold using the which() function 
    fold_split_idx$test <- rbind(fold_split_idx$test, which(folds == i, arr.ind = TRUE))
    fold_split_idx$train <- rbind(fold_split_idx$train, which(folds != i, arr.ind = TRUE))
    
  }
  
  ### fold trees
  fold_tree <- NULL
  for(i in 1:n_folds){
    fold_tree[[i]] <- mmit(target.mat[fold_split_idx$train[i,],], feature.mat[fold_split_idx$train[i,],],  
                                       max_depth = as.numeric(parameters$max_depth), margin = as.numeric(parameters$margin), 
                                       loss = parameters$loss, min_sample = as.numeric(parameters$min_sample))
  }
  
  ### master tree
  master_tree <- mmit(target.mat, feature.mat,  
                      max_depth = as.numeric(parameters$max_depth), margin = as.numeric(parameters$margin), 
                      loss = parameters$loss, min_sample = as.numeric(parameters$min_sample))
  
 
  ### for traning data
  alpha_train_scores <- NULL
  alpha_train_objective_values <- NULL
  
  ### if pruning
  if(pruning){
    # Get the pruned master and cross-validation trees
    master_data <- mmit.pruning(master_tree)
    master_alphas <- rev(unlist(lapply(master_data, function(x) x$alpha)))
    master_pruned_trees <- rev(lapply(master_data, function(x) x$tree))
    
    fold_alphas <- NULL
    fold_prune_trees <- NULL
    for(i in 1 : n_folds){
      fold_data <- mmit.pruning(fold_tree[[i]])
      
      ### as mmit.pruning gives the alphas in descending order we need to reverse the list.
      fold_alphas[[i]] <- rev(unlist(lapply(fold_data, function(x) x$alpha)))
      fold_prune_trees[[i]] <- rev(lapply(fold_data, function(x) x$tree)) 
    }

    # Compute the test risk for all pruned trees of each fold
    alpha_path_score <- NULL
    for(i in 1 : n_folds){
      
      #dummy value initialised to form dataframe, removed later
      alpha_path_score[[i]] <- c(0, 0, 0)   
      
      for(j in 1 : length(fold_alphas[[i]])){
        ### convert pruned tree list to partynode
        node <- fold_prune_trees[[i]][[j]]
        
        ### predictions of the tree for test data
        prediction <- mmit.predict(node, feature.mat[fold_split_idx$test[i,],])

        ###error calc
        fold_test_scores <- scorer(target.mat[fold_split_idx$test[i,],], prediction)
        
        ### creating a dataframe with init alpha, final alpha, score value
        if(j < length(fold_alphas[[i]])){
          data <- c(fold_alphas[[i]][j], fold_alphas[[i]][j + 1], fold_test_scores)
          alpha_path_score[[i]] <- rbind(alpha_path_score[[i]], data)
        }
        else{
          data <- c(fold_alphas[[i]][j], Inf, fold_test_scores)
          alpha_path_score[[i]] <- rbind(alpha_path_score[[i]], data)
        }
      }
      
      colnames(alpha_path_score[[i]]) <- c("init alpha", " final alpha", "score")
      alpha_path_score[[i]] <- tail(alpha_path_score[[i]], -1)
      alpha_path_score[[i]] <- as.data.frame(alpha_path_score[[i]], row.names = "")
    }
    
    # Prune the master tree based on the CV estimates
    alphas <- NULL
    alpha_cv_scores <- NULL
    best_alpha <- -1
    best_score <- attr(scorer, "worst")
    best_tree <- NULL

    for(i in 1 : length(master_alphas)){
      if(i < length(master_alphas)){
        geo_mean_alpha_k <- sqrt(master_alphas[[i]] * master_alphas[[i + 1]])
      }
      else{
        geo_mean_alpha_k <- Inf
      }

      ### calc train score
      prediction <- mmit.predict(master_pruned_trees[[i]], feature.mat)
      train_score <- scorer(target.mat, prediction)
      
      ### calc cost of all leaves
      ter_id <- nodeids(master_pruned_trees[[i]], terminal = TRUE)
      n <- nodeapply(master_pruned_trees[[i]], ids = ter_id, info_node)
      train_objective <- sum(matrix(unlist(n), nrow = length(n), byrow = T)[, 2])
      
      alpha_train_scores <- c(alpha_train_scores, train_score)
      alpha_train_objective_values <- c(alpha_train_objective_values, train_objective)
      
      ### compute cv_score as mean of each fold scores
      cv_score <- 0
      flag <- 0
      for(j in 1 : n_folds){
        for(k in 1 : nrow(alpha_path_score[[j]])){
          if((geo_mean_alpha_k <= alpha_path_score[[j]][k, ]$` final alpha`) && (geo_mean_alpha_k >= alpha_path_score[[j]][k, ]$`init alpha`)){
            cv_score <- cv_score + alpha_path_score[[j]][k, ]$score
            flag <- 1
            break
          }
        }
      }

      assert_that(flag == 1, msg = "cv_score not updated")
      cv_score <- cv_score/n_folds

      # Log metrics for this alpha value
      alphas <- c(alphas, geo_mean_alpha_k)
      alpha_cv_scores <- c(alpha_cv_scores, cv_score)
      
      if(attr(scorer, "direction")(cv_score, best_score) == cv_score){
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
      prediction <- mmit.predict(fold_tree[[i]], feature.mat[fold_split_idx$test[i,],])
      fold_test_scores <- c(fold_test_scores, scorer(target.mat, prediction))
    }
      
    best_alpha <-  0.
    best_score <- mean(fold_test_scores)
    best_tree <- master_tree
    alphas <- c(0.)
    alpha_cv_scores <- c(best_score)
    
    
    ### calc train score
    prediction <- mmit.predict(master_tree, feature.mat)
    alpha_train_scores <- scorer(target.mat, prediction)
    
    ### calc cost of all leaves
    ter_id <- nodeids(master_tree, terminal = TRUE)
    n <- nodeapply(master_tree, ids = ter_id, info_node)
    alpha_train_objective_values <- sum(matrix(unlist(n), nrow = length(n), byrow = T)[, 2])

  }
  
  ### Append alpha to the parameters
  best_params <- parameters
  best_params$alpha <- best_alpha
  
  ### Generate a big dictionnary of all HP combinations considered (including alpha) and their CV scores
  cv_results <- cbind(parameters$max_depth, parameters$margin, parameters$min_sample, parameters$loss,
                      alphas, alpha_cv_scores, alpha_train_scores, alpha_train_objective_values)
  colnames(cv_results) <- c("max_depth", "margin", "min_sample", "loss", "alpha", " cv_score", "train_score", "train_objective_value")
  cv_results <- as.data.frame(cv_results)
  
  output <- NULL
  output$best_score <- best_score
  output$best_estimator <- best_tree
  output$best_params <- best_params
  output$cv_results <- cv_results
  
  return(output)
  
}, ex=function(){
  
  library(survival)
  data(neuroblastomaProcessed, package="penaltyLearning")
  feature.mat <- data.frame(neuroblastomaProcessed$feature.mat)[1:45,]
  target.mat <- neuroblastomaProcessed$target.mat[1:45,]

  parameters <- NULL
  parameters$max_depth <- Inf
  parameters$margin <- 2
  parameters$min_sample <- 2
  parameters$loss <- c("hinge")
  
  result <- fit_and_score(target.mat, feature.mat, parameters, scorer = mse)
  
})
    
