library(testthat)
context("trivial")
library(mmit)

target.mat <- rbind(
  c(-1, Inf),
  c(-2, 3),
  c(-Inf, 1))
input.vec <- c(-2.2, 3, 10)
positive.part <- function(x)ifelse(0 < x, x, 0)
plotMargin <- function(margin){
  curve({
    mat <- function(vec){
      matrix(vec, length(x), length(vec), byrow=TRUE)
    }
    rowSums(positive.part(mat(target.mat[,1]+margin)-x))+
      rowSums(positive.part(x-mat(target.mat[,2]-margin)))
  }, -3, 4, ylab="cost", ylim=c(0, 7))
}

##   -1 0 1 2 3
##    | | | | | 
##    \___x___/
##    \_x_/
plotMargin(0)
out0 <- compute_optimal_costs(
  target.mat, margin=0, loss="hinge")
test_that("margin=0 yields cost 0", {
  expect_equal(out0$pred, c(-0.9999, 1, 0))
  expect_equal(out0$cost, c(0, 0, 0))
})

##   -1 0 1 2 3
##    | | | | |
##     \__x__/
##     \x/
plotMargin(0.5)
out0.5 <- compute_optimal_costs(
  target.mat, margin=0.5, loss="hinge")
test_that("margin=0.5 yields cost 0", {
  expect_equal(out0.5$pred, c(-0.4999, 1, 0))
  expect_equal(out0.5$cost, c(0, 0, 0))
})

##   -1 0 1 2 3
##    | | | | |
##      \_x_/
##      x
plotMargin(1)
out1 <- compute_optimal_costs(
  target.mat, margin=1, loss="hinge")
test_that("margin=1 yields cost 0", {
  expect_equal(out1$pred, c(1e-04, 1, 0))
  expect_equal(out1$cost, c(0, 0, 0))
})

##   -1 0 1 2 3
##    | | | | | 
##       \x/
##    \_x_/
plotMargin(1.5)
out1.5 <- compute_optimal_costs(
  target.mat, margin=1.5, loss="hinge")
test_that("margin=1.5 yields cost 0,1", {
  expect_equal(out1.5$pred, c(0.5001, 1, 0))
  expect_equal(out1.5$cost, c(0, 0, 1))
})

##   -1 0 1 2 3
##    | | | | |
##        x
##      \x/
plotMargin(2)
out2 <- compute_optimal_costs(
  target.mat, margin=2, loss="hinge")
test_that("margin=2 yields cost 0,2", {
  expect_equal(out2$pred, c(1.0001, 1, 0.5))
  expect_equal(out2$cost, c(0, 0, 2))
})

i.rev <- 3:1
input.rev <- input.vec[i.rev]
target.rev <- target.mat[i.rev, ]
rev0 <- compute_optimal_costs(
  target.rev, margin=0, loss="hinge")
test_that("reverse inputs with margin=0 yields 0 cost", {
  expect_equal(rev0$pred, c(1, -0.5, 0))
  expect_equal(rev0$cost, c(0, 0, 0))
})

