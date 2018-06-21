library(testthat)
context("pruning")
library(mmit)

### check for zero one loss 1
y_true <- rbind(c(0,1), c(1,2), c(2,4))
y_pred <- c(0.5, 1.5, 3)
output <- zero_one_loss(y_true, y_pred)
test_that(" check for zero one loss test1", {
  expect_equal(output, 0.)
})

### check for zero one loss 2
y_true <- rbind(c(0,1), c(1,2), c(2,4))
y_pred <- c(0.5, 2.5, 1)
output <- zero_one_loss(y_true, y_pred)
test_that(" check for zero one loss test2", {
  expect_equal(output, 2/3)
})

### check mean square loss 1
y_true <- rbind(c(0,1), c(1,2), c(2,4))
y_pred <- c(0.5, 1.5, 3)
output <- mse(y_true, y_pred)
test_that(" check mean square loss test1", {
  expect_equal(output, 0.)
})

### check mean square loss 2
y_true <- rbind(c(0,1), c(1,2), c(2,4))
y_pred <- c(1.5, 2.5, 4.5)
output <- mse(y_true, y_pred)
test_that(" check mean square loss test2", {
  expect_equal(output, 0.25)
})

