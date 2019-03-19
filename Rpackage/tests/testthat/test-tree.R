library(testthat)
context("trivial")
library(mmit)

target.mat <- rbind(
  c(0, 1), c(0, 1), c(0, 1),
  c(2, 3), c(2, 3), c(2, 3))

feature.mat <- rbind(
  c(1, 0, 0), c(1, 1, 0), c(1, 2, 0),
  c(1, 3, 0), c(1, 4, 0), c(1, 5, 0))

weights <- rep(1L, nrow(feature.mat))

colnames(feature.mat) <- c("a", "b", "c")
feature.mat <- data.frame(feature.mat)

#linear hinge, margin = 0, test best split
out <- bestsplit(target.mat, feature.mat, margin=0.0, loss="hinge", weights = weights)
test_that("finding best split for hinge loss, margin as 0", {
  expect_equal(out$leftpred, 0.5)
  expect_equal(out$rightpred, 2.5)
  expect_equal(out$cost, 0.0)
  expect_equal(out$br, 2)
  expect_equal(out$varid, 2)
  expect_equal(out$row, 3)
})

#square hinge, margin = 0, test best split
out<-bestsplit(target.mat, feature.mat, margin=0.0, loss="square", weights = weights)
test_that("finding best split for squared loss, margin as 0", {
  expect_equal(out$leftpred, 0.5)
  expect_equal(out$rightpred, 2.5)
  expect_equal(out$cost, 0.0)
  expect_equal(out$br, 2)
  expect_equal(out$varid, 2)
  expect_equal(out$row, 3)
})

#linear hinge, margin = 0, test mmit
out <- mmit(target.mat, feature.mat)
p <- predict(out)
test_that("predicting the tree mmit() with hinge loss", {
  expect_equal(nodeapply(out, ids = p[[1]], info_node)[[1]][[1]], 0.5)
  expect_equal(nodeapply(out, ids = p[[2]], info_node)[[1]][[1]], 0.5)
  expect_equal(nodeapply(out, ids = p[[3]], info_node)[[1]][[1]], 0.5)
  expect_equal(nodeapply(out, ids = p[[4]], info_node)[[1]][[1]], 2.5)
  expect_equal(nodeapply(out, ids = p[[5]], info_node)[[1]][[1]], 2.5)
  expect_equal(nodeapply(out, ids = p[[6]], info_node)[[1]][[1]], 2.5)
})


#square hinge, margin = 0, test mmit
out <- mmit(target.mat, feature.mat, loss="square")
p <- predict(out)
test_that("predicting the tree mmit() with square loss", {
  expect_equal(nodeapply(out, ids = p[[1]], info_node)[[1]][[1]], 0.5)
  expect_equal(nodeapply(out, ids = p[[2]], info_node)[[1]][[1]], 0.5)
  expect_equal(nodeapply(out, ids = p[[3]], info_node)[[1]][[1]], 0.5)
  expect_equal(nodeapply(out, ids = p[[4]], info_node)[[1]][[1]], 2.5)
  expect_equal(nodeapply(out, ids = p[[5]], info_node)[[1]][[1]], 2.5)
  expect_equal(nodeapply(out, ids = p[[6]], info_node)[[1]][[1]], 2.5)
})

### test for depth 
data(neuroblastomaProcessed, package="penaltyLearning")
feature.mat <- data.frame(neuroblastomaProcessed$feature.mat)[1:45,]
target.mat <- neuroblastomaProcessed$target.mat[1:45,]
weights <- rep(1L, nrow(feature.mat))

#max_depth = 2 , depth = 0
tree <- growtree(target.mat, feature.mat, depth = 0, max_depth = 2, margin = 2.0, weights = weights)
test_that("depth check max_depth = 2, depth = 0", {
  expect_equal(depth(tree), 2)
})

#max_depth = 3 , depth = 1
tree <- growtree(target.mat, feature.mat, depth = 1, max_depth = 3, margin = 2.0, weights = weights)
test_that("depth check max_depth = 3, depth = 1", {
  expect_equal(depth(tree), 2)
})

#max_depth = 3 , depth = 0
tree <- growtree(target.mat, feature.mat, depth = 0, max_depth = 3, margin = 2.0, weights = weights)
test_that("max_depth = 3, depth = 0", {
  expect_equal(depth(tree), 3)
})

