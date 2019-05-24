fit_and_score <- structure(function(target.mat, feature.mat, 
                                    parameters, n_folds = 3, scorer = NULL, 
                                    learner = NULL, pruning = TRUE){
  learner.predict = paste(learner, ".predict", sep = "")
  
  ### if pruning is true then learner should be mmit
  if(pruning==TRUE){
    assert_that(learner == "mmit")
  }
  
  ### shuffle the data 
  set.seed(1)
  rand <- sample(nrow(target.mat))
  target.mat <- target.mat[rand,]
  feature.mat <- feature.mat[rand,]
  
  
  ### Create n equally size folds
  folds <- cut(seq(1,nrow(target.mat)), breaks = n_folds, labels = FALSE)
  
  ### train and test index for each fold
  fold_split_idx <- list()
  
  ### segment the data into test and train
  for(i in 1:n_folds){
    #Segement your data by fold using the which() function 
    fold_split_idx$test <- rbind(fold_split_idx$test, which(folds == i, arr.ind = TRUE))
    fold_split_idx$train <- rbind(fold_split_idx$train, which(folds != i, arr.ind = TRUE))
    
  }
  
  ### fold models
  fold_model <- list()
  for(i in 1:n_folds){
    arguments <- list(target.mat= target.mat[fold_split_idx$train[i,],], feature.mat = feature.mat[fold_split_idx$train[i,],]) 
    fold_model[[i]] <- do.call(learner, c(arguments, parameters))
  }
  
  ### master model
  arguments <- list(target.mat = target.mat, feature.mat = feature.mat)
  master_model <- do.call(learner, c(arguments, parameters))
  
  ### for traning data
  alpha_train_scores <- NULL
  alpha_train_objective_values <- NULL
  
  
  ### if pruning
  if(pruning){
    # Get the pruned master and cross-validation models
    master_data <- mmit.pruning(master_model)
    master_alphas <- rev(unlist(lapply(master_data, function(x) x$alpha)))
    master_pruned_models <- rev(lapply(master_data, function(x) x$tree))
    
    fold_alphas <- list()
    fold_prune_models <- list()
    for(i in 1 : n_folds){
      fold_data <- mmit.pruning(fold_model[[i]])
      
      ### as mmit.pruning gives the alphas in descending order we need to reverse the list.
      fold_alphas[[i]] <- rev(unlist(lapply(fold_data, function(x) x$alpha)))
      fold_prune_models[[i]] <- rev(lapply(fold_data, function(x) x$tree)) 
    }

    # Compute the test risk for all pruned models of each fold
    alpha_path_score <- list()
    for(i in 1 : n_folds){
      
      #dummy value initialised to form dataframe, removed later
      alpha_path_score[[i]] <- c(0, 0, 0)   
      
      for(j in 1 : length(fold_alphas[[i]])){
        ### convert pruned model list to partynode
        node <- fold_prune_models[[i]][[j]]
        
        ### predictions of the model for test data
        prediction <- do.call(learner.predict, list(node, feature.mat[fold_split_idx$test[i,],]))

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
    
    # Prune the master model based on the CV estimates
    alphas <- NULL
    alpha_cv_scores <- NULL
    best_alpha <- -1.0
    best_score <- attr(scorer, "worst")
    best_model <- NULL

    for(i in 1 : length(master_alphas)){
      if(i < length(master_alphas)){
        geo_mean_alpha_k <- sqrt(master_alphas[[i]] * master_alphas[[i + 1]])
      }
      else{
        geo_mean_alpha_k <- Inf
      }

      ### calc train score
      prediction <- do.call(learner.predict, list(master_pruned_models[[i]], feature.mat))
      train_score <- scorer(target.mat, prediction)
      
      ### calc cost of all leaves
      ter_id <- nodeids(master_pruned_models[[i]], terminal = TRUE)
      n <- nodeapply(master_pruned_models[[i]], ids = ter_id, info_node)
      train_objective <- sum(matrix(unlist(n), nrow = length(n), byrow = T)[, 2])
      
      alpha_train_scores <- c(alpha_train_scores, train_score)
      alpha_train_objective_values <- c(alpha_train_objective_values, train_objective)
      
      ### compute cv_score as mean of each fold scores
      cv_score <- 0
      flag <- 0
      for(j in 1 : n_folds){
        for(k in 1 : nrow(alpha_path_score[[j]])){
          if((geo_mean_alpha_k < alpha_path_score[[j]][k, ]$` final alpha` || is.infinite(alpha_path_score[[j]][k, ]$` final alpha`)) && geo_mean_alpha_k >= alpha_path_score[[j]][k, ]$`init alpha`){
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
        best_model <- master_pruned_models[[i]]
      }
    }
  }
  else{
    alpha_train_scores <- 0.0
    alpha_train_objective_values <- 0.0
    ### For each fold, build a decision model
    fold_test_scores <- NULL
    for(i in 1 : n_folds){
      prediction <- do.call(learner.predict, list(fold_model[[i]], feature.mat[fold_split_idx$test[i,],]))
      fold_test_scores <- c(fold_test_scores, scorer(target.mat, prediction))
    }
      
    best_alpha <-  0.
    best_score <- mean(fold_test_scores)
    best_model <- master_model
    alphas <- c(0.)
    alpha_cv_scores <- c(best_score)
    
    ### calc train score
    prediction <- do.call(learner.predict, list(master_model, feature.mat))
    alpha_train_scores <- scorer(target.mat, prediction)
    
    ### calc cost of all leaves
    if(learner == "mmif"){
      for(i in 1:length(master_model)){
        ter_id <- nodeids(master_model[[i]], terminal = TRUE)
        n <- nodeapply(master_model[[i]], ids = ter_id, info_node)
        alpha_train_objective_values <- alpha_train_objective_values + sum(matrix(unlist(n), nrow = length(n), byrow = T)[, 2])
      }
      alpha_train_objective_values <- alpha_train_objective_values/length(master_model)
    }
    else{
      ter_id <- nodeids(master_model, terminal = TRUE)
      n <- nodeapply(master_model, ids = ter_id, info_node)
      alpha_train_objective_values <- sum(matrix(unlist(n), nrow = length(n), byrow = T)[, 2])
    }
    

  }
  
  ### Generate a big dictionnary of all HP combinations considered (including alpha) and their CV scores
  ### Append alpha to the parameters
  best_params <- parameters
  
  if(learner == "mmit"){
    best_params$alpha <- best_alpha
    cv_results <- cbind(parameters$max_depth, parameters$margin, parameters$min_sample, parameters$loss,
                        alphas, alpha_cv_scores, alpha_train_scores, alpha_train_objective_values)
    colnames(cv_results) <- c("max_depth", "margin", "min_sample", "loss", "alpha", " cv_score", "train_score", "train_objective_value")
    cv_results <- as.data.frame(cv_results)
  }
  else if(learner == "mmif"){
    cv_results <- cbind(parameters$max_depth, parameters$margin, parameters$min_sample, parameters$loss,
                        parameters$n_trees, parameters$n_features, alpha_cv_scores, alpha_train_scores, alpha_train_objective_values)
    colnames(cv_results) <- c("max_depth", "margin", "min_sample", "loss", "n_trees", " n_features", "cv_score", "train_score", "train_objective_value")
    cv_results <- as.data.frame(cv_results)
  }
  
  output <- list()
  output$best_score <- best_score
  output$best_estimator <- best_model
  output$best_params <- best_params
  output$cv_results <- cv_results
  
  return(output)
  
})
    
