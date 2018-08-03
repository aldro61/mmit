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

## Future Scopes

-----------------------------------------------------------------------------------------------------------

## Link to commits
 
 
