library(testthat)
context("pruning")
library(mmit)

kids <- NULL
kidskids <- NULL
### the pruning leaf test
info_print <- data.frame(1, 10,row.names = "")
colnames(info_print) <- c("prediction", "cost")
root <- partynode(1, info = info_print)
output <- pruning(root)
cost <- nodeapply(root, ids = 1, info_node)[[1]][1,2]
test_that("finding pruning of tree with root as leave", {
  expect_equal(output[[1]], 0.)
  expect_equal(cost, 10.)
})

### test initial pruning (Tmax is terminal)
info_print <- data.frame(1, 5,row.names = "")
colnames(info_print) <- c("prediction", "cost")
kids[[1]] <- partynode(2, info = info_print)

info_print <- data.frame(3, 5,row.names = "")
colnames(info_print) <- c("prediction", "cost")
kids[[2]] <- partynode(3, info = info_print)

info_print <- data.frame(1, 10,row.names = "")
colnames(info_print) <- c("prediction", "cost")
split <- partysplit(varid = as.integer(1), breaks = 2)
root <- partynode(1, split = split, kids = kids, info = info_print)

output <- pruning(root)
cost <- nodeapply(root, ids = 1, info_node)[[1]][1,2]
test_that("initial pruning", {
  expect_equal(output[[1]], 0.)
  expect_equal(cost, 10.)
})

### weakest link test 
info_print <- data.frame(1, 15,row.names = "")
colnames(info_print) <- c("prediction", "cost")
kids[[1]] <- partynode(2, info = info_print)

info_print <- data.frame(1, 15,row.names = "")
colnames(info_print) <- c("prediction", "cost")
kids[[2]] <- partynode(3, info = info_print)

info_print <- data.frame(1, 10,row.names = "")
colnames(info_print) <- c("prediction", "cost")
split <- partysplit(varid = as.integer(1), breaks = 2)
root <- partynode(1, split = split, kids = kids, info = info_print)

output <- weakest_link(root)
test_that("finding weakest link", {
  expect_equal(output[[2]], 1)
})

### pruning test 1
split <- partysplit(varid = as.integer(1), breaks = 2)
#l1.1
info_print <- data.frame(1, 0,row.names = "")
colnames(info_print) <- c("prediction", "cost")
kidskids[[1]] <- partynode(3, info = info_print)
#r1.2
info_print <- data.frame(1, 0,row.names = "")
colnames(info_print) <- c("prediction", "cost")
kidskids[[2]] <- partynode(4, info = info_print)

#l1
info_print <- data.frame(1, 5,row.names = "")
colnames(info_print) <- c("prediction", "cost")
kids[[1]] <- partynode(2, split = split, kids = kidskids, info = info_print)

split <- partysplit(varid = as.integer(1), breaks = 4)
#l2.1
info_print <- data.frame(1, 0,row.names = "")
colnames(info_print) <- c("prediction", "cost")
kidskids[[1]] <- partynode(6, info = info_print)
#r2.2
info_print <- data.frame(1, 1,row.names = "")
colnames(info_print) <- c("prediction", "cost")
kidskids[[2]] <- partynode(7, info = info_print)

#r2
info_print <- data.frame(1, 3,row.names = "")
colnames(info_print) <- c("prediction", "cost")
kids[[2]] <- partynode(5, split = split, kids = kidskids, info = info_print)


#root
split <- partysplit(varid = as.integer(1), breaks = 5)
info_print <- data.frame(1, 10,row.names = "")
colnames(info_print) <- c("prediction", "cost")
root <- partynode(1, split = split, kids = kids, info = info_print)

#pruning
output <- pruning(root)
test_that("pruning test 1", {
  #alpha
  expect_equal(unlist(output[,1]), c(0,2,3.5))
})
