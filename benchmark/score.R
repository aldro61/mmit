library(mmit)

dataset <- list.files(path = "/home/parismita/mmit_data")

for(i in 1:length(dataset)){
  feature.mat <- read.csv(paste("~/mmit_data/", dataset[i], "/features.csv", sep=""))
  target.mat <- read.csv(paste("~/mmit_data/", dataset[i], "/targets.csv", sep=""))
  
  #rand <- sample(nrow(target.mat))
  #target.mat <- target.mat[rand,]
  #feature.mat <- feature.mat[rand,]
  if(nrow(feature.mat)<= 200) next
  
  train.feat <- head(feature.mat, nrow(feature.mat)/2)
  test.feat <- tail(feature.mat, nrow(feature.mat)/2)
  
  train.tar <- data.matrix(head(target.mat, nrow(target.mat)/2))
  test.tar <- data.matrix(tail(target.mat, nrow(target.mat)/2))
  
  ptm <- proc.time()
  tree <- mmit(train.tar, train.feat, max_depth = 4, margin = 1.0, min_sample = 0, loss = "hinge")
  fit <- mmit.predict(tree, test.feat)
  score <- paste(dataset[i], " |", mse(test.tar, fit), " |" , nrow(feature.mat), "|")
  #print(proc.time() - ptm)
  #print(length(fit))
  print(score)
  write.table(score, file = "c.out", append = TRUE)
  #plot(tree)
}
