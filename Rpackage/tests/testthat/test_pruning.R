library(testthat)
context("pruning")
library(mmit)
library(grid)

kids <- NULL
kidskids <- NULL
### the pruning leaf test
info_print <- data.frame(1, 10, row.names = "")
colnames(info_print) <- c("prediction", "cost")
root <- partynode(1, info = info_print)
output <- mmit.pruning(root)
cost <- nodeapply(root, ids = 1, info_node)[[1]][1, 2]
test_that("finding pruning of tree with root as leave", {
  expect_equal(unlist(lapply(output, function(x) x$alpha)), 0.)
  expect_equal(cost, 10.)
})

### test initial pruning (Tmax is terminal)
info_print <- data.frame(1, 5, row.names = "")
colnames(info_print) <- c("prediction", "cost")
kids[[1]] <- partynode(2, info = info_print)

info_print <- data.frame(3, 5, row.names = "")
colnames(info_print) <- c("prediction", "cost")
kids[[2]] <- partynode(3, info = info_print)

info_print <- data.frame(1, 10, row.names = "")
colnames(info_print) <- c("prediction", "cost")
split <- partysplit(varid = as.integer(1), breaks = 2)
root <- partynode(1, split = split, kids = kids, info = info_print)

output <- mmit.pruning(root)
cost <- nodeapply(root, ids = 1, info_node)[[1]][1,2]
test_that("initial pruning", {
  expect_equal(unlist(lapply(output, function(x) x$alpha)), 0.)
  expect_equal(cost, 10.)
})

### weakest link test 
info_print <- data.frame(1, 15, row.names = "")
colnames(info_print) <- c("prediction", "cost")
kids[[1]] <- partynode(2, info = info_print)

info_print <- data.frame(1, 1, row.names = "")
colnames(info_print) <- c("prediction", "cost")
kids[[2]] <- partynode(3, info = info_print)

info_print <- data.frame(1, 20 , row.names = "")
colnames(info_print) <- c("prediction", "cost")
split <- partysplit(varid = as.integer(1), breaks = 2)
root <- partynode(1, split = split, kids = kids, info = info_print)

output <- .weakest_link(root)
test_that("finding weakest link", {
  expect_equal(output[[2]], 1)
})

### pruning test 1
split <- partysplit(varid = as.integer(1), breaks = 2)
#l1.1
info_print <- data.frame(1, 0, row.names = "")
colnames(info_print) <- c("prediction", "cost")
kidskids[[1]] <- partynode(3, info = info_print)
#r1.2
info_print <- data.frame(1, 0, row.names = "")
colnames(info_print) <- c("prediction", "cost")
kidskids[[2]] <- partynode(4, info = info_print)

#l1
info_print <- data.frame(1, 5, row.names = "")
colnames(info_print) <- c("prediction", "cost")
kids[[1]] <- partynode(2, split = split, kids = kidskids, info = info_print)

split <- partysplit(varid = as.integer(1), breaks = 4)
#l2.1
info_print <- data.frame(1, 0, row.names = "")
colnames(info_print) <- c("prediction", "cost")
kidskids[[1]] <- partynode(6, info = info_print)
#r2.2
info_print <- data.frame(1, 1, row.names = "")
colnames(info_print) <- c("prediction", "cost")
kidskids[[2]] <- partynode(7, info = info_print)

#r2
info_print <- data.frame(1, 3, row.names = "")
colnames(info_print) <- c("prediction", "cost")
kids[[2]] <- partynode(5, split = split, kids = kidskids, info = info_print)


#root
split <- partysplit(varid = as.integer(1), breaks = 5)
info_print <- data.frame(1, 10, row.names = "")
colnames(info_print) <- c("prediction", "cost")
root <- partynode(1, split = split, kids = kids, info = info_print)

#pruning
output <- mmit.pruning(root)
test_that("pruning test 1", {
  #alpha
  expect_equal(unlist(lapply(output, function(x) x$alpha)), c(3.5, 2, 0.0))
})



### pruning test 1
split <- partysplit(varid = as.integer(1), breaks = 2)
#l1.1
info_print <- data.frame(1, 2, row.names = "")
colnames(info_print) <- c("prediction", "cost")
kidskids[[1]] <- partynode(3, info = info_print)
#r1.2
info_print <- data.frame(1, 0, row.names = "")
colnames(info_print) <- c("prediction", "cost")
kidskids[[2]] <- partynode(4, info = info_print)

#l1
info_print <- data.frame(1, 20, row.names = "")
colnames(info_print) <- c("prediction", "cost")
kids[[1]] <- partynode(2, split = split, kids = kidskids, info = info_print)

split <- partysplit(varid = as.integer(1), breaks = 4)
#l2.1
info_print <- data.frame(1, 2, row.names = "")
colnames(info_print) <- c("prediction", "cost")
kidskids[[1]] <- partynode(6, info = info_print)
#r2.2
info_print <- data.frame(1, 1, row.names = "")
colnames(info_print) <- c("prediction", "cost")
kidskids[[2]] <- partynode(7, info = info_print)

#r2
info_print <- data.frame(1, 10, row.names = "")
colnames(info_print) <- c("prediction", "cost")
kids[[2]] <- partynode(5, split = split, kids = kidskids, info = info_print)


#root
split <- partysplit(varid = as.integer(1), breaks = 5)
info_print <- data.frame(1, 100, row.names = "")
colnames(info_print) <- c("prediction", "cost")
root <- partynode(1, split = split, kids = kids, info = info_print)

#pruning
output <- mmit.pruning(root)
test_that("pruning test 2", {
  #alpha
  expect_equal(unlist(lapply(output, function(x) x$alpha)), c(70, 18, 7, 0))
})

