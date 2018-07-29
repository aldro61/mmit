# Google Summer of Code, 2018: MMIT

#### Summary of the Maximum Margin Interval Trees (MMIT) project under the R project for statistical computing.

*Student: Parismita Das*

*Mentors : Alexandre Drouin, Torsten Hothorn and Toby Dylan Hocking*

---------------------------------------------------------------
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
 
 * Pruning of the MMIT tree.
 
 * Cross Validation to return the best parameters using grid search.
 
 * The random forest Implementation on the MMIT trees, which outputs the ensemble of trees and 
 the predict function outputs the average of the predicted values of each tree.
 
 * Cross Validation for the Random forest.
 
 The MMIT R package would be useful for predicting all kinds of censored data such as left\right\censored, to give real value prediction.
 with super fast speed due to the use of dynamic programming. 
 
 -------------------------------------------------------------------------------------------------
 
 
 
