# Google Summer of Code, 2018: MMIT

#### Summary of the Maximum Margin Interval Trees (MMIT) project under the R project for statistical computing.

*Student: Parismita Das*

*Mentors : Alexandre Drouin, Torsten Hothorn and Toby Dylan Hocking*

---------------------------------------------------------------

## Table of contents

* [Abstract](#abstract)
* [Introduction](#introduction)
* [First Evaluation](#first-evaluation)
* [Second Evaluation](#second-evaluation)
* [Final Evaluation](#final-evaluation)
* [Future Work](#future-work)
* [Link to commits](#link-to-commits)
* [Tutorials](#tutorials)

## Abstract

There are few R packages available for interval regression, a machine learning problem 
which is important in genomics and medicine. Like usual regression, the goal is to learn 
a function that inputs a feature vector and outputs a real-valued prediction. 
Unlike usual regression, each response in the training set is an interval of acceptable 
values (rather than one value). In the terminology of the survival analysis literature, 
this is regression with “left, right, and interval censored” output/response data.

Max margin interval trees is a new nonlinear model for this problem (Drouin et al., 2017). 
A dynamic programming algorithm is used to find the optimal split point for each feature. 
The dynamic programming algorithm has been implemented in C++ and there are wrappers to this 
solver in R and Python (https://github.com/aldro61/mmit). 
The Python package includes a decision tree learner. 


 ----------------------------------------------------------------------
 ## Introduction
 
 The main goal of this GSOC project is to build the R package that implements the decision tree learner in R, using <b> partykit </b>.
 
 This project consist of:
 
 * Max margin interval tree, which take the interval censored data and its parameters as input and give <b> party </b> object 
 of the learned tree as output.
 
 * A predict function that predicts the Testing data on the MMIT.
 
 * Pruning of the MMIT.
 
 * Cross Validation of MMIT to return the best parameters using grid search.
 
 * The random forest Implementation on the MMIT trees, which outputs the ensemble of trees 
 
 * A predict function for Random Forest that outputs the average of the predicted values of each tree.
 
 * Cross Validation for the Random forest.
 
 The MMIT R package would be useful for predicting all kinds of censored data such as left\right\censored, to give real value prediction.
 with super fast speed due to the use of dynamic programming. 
 
 -------------------------------------------------------------------------------------------------
 ## First Evaluation
 
 For the major part of the first coding period, the main aim was to build the maximum margin interval tree(mmit) 
 along with its  K-fold cross validation and pruning tree. 
 The tree model was build using partykit which made it very easy to visualize the tree structure 
 and get its prediction values. The tree model and pruning was done to a large extent including 
 appropriate documentation and unit tests which helped later on in the project. 
 Although the Cross validation part too longer time than expected.
 
---------------------------------------------------------------------------------------------------
 
## Second Evaluation

After building the MMIT model, most of the second coding phase went on bug fixing and building unit tests and examples, along with completing the cross validation for the MMIT. The cross validation function consist of 
option for enabling and disabling the pruning of MMIT. It does Grid search on all the parameters and gives the best parameters 
along with the list of CV results for all parameters.
To validate the work, Benchmark was created to compare the mmit and pruning model to the python results. The results were almost 
similar for larger datasets, thus validation the models.

--------------------------------------------------------------------------------------------------------

## Final Evaluation

The final phase work consist of building the Random forest and Adaboost for the MMIT, Along with its Cross validation.
As The aim by the end of GSoc is to build a working package , hence before adding new features to the package, testing, bug 
fixing and documentation are completed. Getting the code to work proved to be suprisingly (and notoriously) more complicated than 
imagined, as the main code used for Cross Validation is common for all modules of the package, thus it has to be made easy to incorporate new features. Thus it took a lot of time to debug the code to perfection.

----------------------------------------------------------------------------------------------------------

## Future Work

The Implementation of Adaboost is currently under progress ([link](https://github.com/aldro61/mmit/pull/28)). The future work for the project is to improve the time 
taken for each module and write the vignettes no make it submittable to CRAN.

-----------------------------------------------------------------------------------------------------------

## Link to commits

The links to all the commits are given below:

 * MMIT: [Link](https://github.com/aldro61/mmit/pull/24/commits)
 
 * Pruning and Cross-Validation: [Link](https://github.com/aldro61/mmit/pull/25/commits)
 
 * Random Forest: [Link](https://github.com/aldro61/mmit/pull/26/commits)
 
 * Adaboost: [Link](https://github.com/aldro61/mmit/pull/28)
 
 The links for documentation and benchmark
 
 * Documentation: [Link](https://github.com/aldro61/mmit/tree/master/Rpackage/man)
 
 * Benchmark: [Link](https://github.com/aldro61/mmit/blob/master/benchmark/pred_score.md)
  
  -------------------------------------------------------------------------------------------------------
  
  ## Tutorials
  
  Here is a tutorial how to use the package.
  
  ### mmit()
  
  This function is used to create the maximum margin interval trees. It returns party object as the tree.
  
  #### Usage:
  
  `mmit(target.mat, feature.mat, max_depth = Inf, margin = 0, loss = "hinge",
  min_sample = 1)`
  
  #### Example:
  
```R
library(mmit)
target.mat <- rbind(
  c(0,1), c(0,1), c(0,1),
  c(2,3), c(2,3), c(2,3))

feature.mat <- rbind(
  c(1,0,0), c(1,1,0), c(1,2,0),
  c(1,3,0), c(1,4,0), c(1,5,0))

colnames(feature.mat) <- c("a", "b", "c")
feature.mat <- data.frame(feature.mat)


out <- mmit(target.mat, feature.mat)
plot(out)
```

#### Output:



### mmit.predict()
  
  Fits the new data into the MMIT model to give prediction values
  
  #### Usage:
  
  `mmit.predict(tree, newdata = NULL, perm = NULL)`
  
  #### Example:
  
```R
library(mmit)
target.mat <- rbind(
  c(0,1), c(0,1), c(0,1),
  c(2,3), c(2,3), c(2,3))

feature.mat <- rbind(
  c(1,0,0), c(1,1,0), c(1,2,0),
  c(1,3,0), c(1,4,0), c(1,5,0))

colnames(feature.mat) <- c("a", "b", "c")
feature.mat <- data.frame(feature.mat)

tree <- mmit(target.mat, feature.mat)
pred <- mmit.predict(tree)
print(pred)
```

#### Output:

pred : 0.5 0.5 0.5 2.5 2.5 2.5

### mmit.pruning()
  
Pruning the regression tree for censored data.
  
  #### Usage:
  
  `mmit.pruning(tree)`
  
  #### Example:
  
```R
library(mmit)
target.mat <- rbind(
  c(0,1), c(0,1), c(0,1),
  c(2,3), c(2,3), c(2,3))

feature.mat <- rbind(
  c(1,0,0), c(1,1,0), c(1,2,0),
  c(1,3,0), c(1,4,0), c(1,5,0))

colnames(feature.mat) <- c("a", "b", "c")
feature.mat <- data.frame(feature.mat)


tree <- mmit(target.mat, feature.mat)
pruned_tree <- mmit.pruning(tree)
```

#### Output:

alpha : 0, 3

tree : 

### mmit.cv()
  
Performing grid search to select the best parameters via cross validation on the a regression tree for censored data.
  
  #### Usage:
  
  `mmit.cv(target.mat, feature.mat, param_grid, n_folds = 3, scorer = NULL,
  n_cpu = 1, pruning = TRUE)`
  
  #### Example:
  
```R
library(mmit)
target.mat <- rbind(
  c(0,1), c(0,1), c(0,1),
  c(2,3), c(2,3), c(2,3))

feature.mat <- rbind(
  c(1,0,0), c(1,1,0), c(1,2,0),
  c(1,3,0), c(1,4,0), c(1,5,0))

colnames(feature.mat) <- c("a", "b", "c")
feature.mat <- data.frame(feature.mat)

param_grid <- NULL
param_grid$max_depth <- c(Inf, 4, 3)
param_grid$margin <- c(2, 3, 5)
param_grid$min_sample <- c(2, 5, 10)
param_grid$loss <- c("hinge")

result <- mmit.cv(target.mat, feature.mat, param_grid, scorer = mse)
plot(result$best_estimator)
print(result$cv_results)
```
#### Output:

Best parameters : 

max_depth | margin |min_sample | loss |alpha |
| ---- |:------: | ---------:|---: |---: |
3         |5       |          10|  hinge|      Inf|

### mmif()
  
Learning a random forest of Max Margin Interval Tree.
  
  #### Usage:
  
  `mmif(target.mat, feature.mat, max_depth = Inf, margin = 0, loss = "hinge",
  min_sample = 1, n_trees = 10,
  n_features = ceiling(ncol(feature.mat)^0.5), n_cpu = 1)`
  
  #### Example:
  
```R
library(mmit)

target.mat <- rbind(
  c(0,1), c(0,1), c(0,1),
  c(2,3), c(2,3), c(2,3))

feature.mat <- rbind(
  c(1,0,0), c(1,1,0), c(1,2,0),
  c(1,3,0), c(1,4,0), c(1,5,0))

colnames(feature.mat) <- c("a", "b", "c")
feature.mat <- data.frame(feature.mat)

trees <- mmif(target.mat, feature.mat, margin = 2.0, n_trees = 2, n_cpu = -1)
print(trees)
```

#### Output:



### mmif.predict()
  
Predictions with random forests of Max Margin Interval Trees
  
  #### Usage:
  
  `mmif.predict(forest, test_feature.mat = NULL)`
  
  #### Example:
  
```R
library(mmit)

target.mat <- rbind(
  c(0,1), c(0,1), c(0,1),
  c(2,3), c(2,3), c(2,3))

feature.mat <- rbind(
  c(1,0,0), c(1,1,0), c(1,2,0),
  c(1,3,0), c(1,4,0), c(1,5,0))

colnames(feature.mat) <- c("a", "b", "c")
feature.mat <- data.frame(feature.mat)

forest <- mmif(target.mat, feature.mat)
pred <- mmif.predict(forest, feature.mat)
print(pred)
```

#### Output:

pred : 0.75 0.95 1.95 2.15 2.15 2.15

### mmif.cv()
  
Performing grid search to select the best hyperparameters of mmif via cross-validation.
  
  #### Usage:
  
  `mmif.cv(target.mat, feature.mat, param_grid, n_folds = 3, scorer = NULL,
  n_cpu = 1)`
  
  #### Example:
  
```R
library(mmit)

target.mat <- rbind(
  c(0,1), c(0,1), c(0,1),
  c(2,3), c(2,3), c(2,3))

feature.mat <- rbind(
  c(1,0,0), c(1,1,0), c(1,2,0),
  c(1,3,0), c(1,4,0), c(1,5,0))

colnames(feature.mat) <- c("a", "b", "c")
feature.mat <- data.frame(feature.mat)

param_grid <- NULL
param_grid$max_depth <- c(Inf, 4, 3)
param_grid$margin <- c(2, 3, 5)
param_grid$min_sample <- c(2, 5, 10)
param_grid$loss <- c("hinge")
param_grid$n_trees <- c(10, 20, 30)
param_grid$n_features <- c(ceiling(ncol(feature.mat)**0.5))

result <- mmif.cv(target.mat, feature.mat, param_grid, scorer = mse, n_cpu = -1)
```

#### Output:

Best parameters :

max_depth | margin |min_sample | loss |n_trees |n_features|
| ---- |:------: | ---------:|---: |---: |---: |
3         |5       |          2|  hinge|      10|          2|

### mse()
  
Metric for mean aquare error calculation.
  
  #### Usage:
  
  `mse(y_true, y_pred)`
  
  #### Example:
  
```R
library(mmit)
y_true <- rbind(
  c(0,1), c(0,1), c(0,1),
  c(2,3), c(2,3), c(2,3))

y_pred <- c(0.5, 2, 0, 1.5, 3.5, 2.5)

out <- mse(y_true, y_pred)
```

#### Output:

out : 0.25

### zero_one_loss()
  
Metric for error calculation where the function gives zero value inside the interval else one.
  
  #### Usage:
  
  `zero_one_loss(y_true, y_pred)`
  
  #### Example:
  
```R
library(mmit)
y_true <- rbind(
  c(0,1), c(0,1), c(0,1),
  c(2,3), c(2,3), c(2,3))

y_pred <- c(0.5, 2, 0, 1.5, 3.5, 2.5)

out <- zero_one_loss(y_true, y_pred)
```

#### Output:

out : 0.5
