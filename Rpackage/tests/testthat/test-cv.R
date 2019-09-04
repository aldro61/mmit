library(mmit)
library(testthat)

data(neuroblastomaProcessed, package="penaltyLearning")

set.seed(1)
n.folds <- 5L
fold.vec <- sample(rep(1:n.folds, l=nrow(neuroblastomaProcessed$target.mat)))
test.fold <- 1L
is.train <- fold.vec != test.fold
X.train <- neuroblastomaProcessed$feature.mat[is.train,]
y.train <- neuroblastomaProcessed$target.mat[is.train,]
X.test <- neuroblastomaProcessed$feature.mat[!is.train,]
y.test <- neuroblastomaProcessed$target.mat[!is.train,]

test_that("mmit.cv(X,y) works", {
  fit <- mmit.cv(X.train, y.train)
  pred.vec <- predict(fit, X.test)
  expect_equal(length(pred.vec), nrow(y.test))
})

test_that("mmif.cv(X,y) works", {
  fit <- mmif.cv(X.train, y.train)
  pred.vec <- predict(fit, X.test)
  expect_equal(length(pred.vec), nrow(y.test))
})
