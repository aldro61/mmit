library(testthat)
context("trivial")
library(mmit)

target.mat <- rbind(
  c(0,1), c(0,1), c(0,1),
  c(2,3), c(2,3), c(2,3))

feature.mat <- rbind(
  c(1,0,0), c(1,1,0), c(1,2,0),
  c(1,3,0), c(1,4,0), c(1,5,0))
colnames(feature.mat) <- c("a", "b", "c")
feature.mat <- data.frame(feature.mat)

#linear hinge, margin = 0, test best split
out<-bestsplit(target.mat, feature.mat, margin=0.0, loss="hinge")
test_that("margin=0.0 yields cost 0", {
  expect_equal(out$leftpred, 0.5)
  expect_equal(out$rightpred, 2.5)
  expect_equal(out$cost, 0.0)
  expect_equal(out$br, 2)
  expect_equal(out$varid, 2)
  expect_equal(out$row, 3)
})

#square hinge, margin = 0, test best split
out<-bestsplit(target.mat, feature.mat, margin=0.0, loss="square")
test_that("margin=0.0 yields cost 0", {
  expect_equal(out$leftpred, 0.5)
  expect_equal(out$rightpred, 2.5)
  expect_equal(out$cost, 0.0)
  expect_equal(out$br, 2)
  expect_equal(out$varid, 2)
  expect_equal(out$row, 3)
})

#linear hinge, margin = 0, test mmit
out<-predict(mmit(a~., target.mat, feature.mat))
test_that("margin=0.0 yields cost 0", {
  expect_equal(out[[1]], 0.5)
  expect_equal(out[[2]], 0.5)
  expect_equal(out[[3]], NA)   #why it should be 0.5
  expect_equal(out[[4]], 2.5)
  expect_equal(out[[5]], 2.5)
  expect_equal(out[[6]], NA)
})

#square hinge, margin = 0, test mmit
out<-predict(mmit(a~., target.mat, feature.mat, loss="square"))
test_that("margin=0.0 yields cost 0", {
  expect_equal(out[[1]], 0.5)
  expect_equal(out[[2]], 0.5)
  expect_equal(out[[3]], NA)
  expect_equal(out[[4]], 2.5)
  expect_equal(out[[5]], 2.5)
  expect_equal(out[[6]], NA)
})
