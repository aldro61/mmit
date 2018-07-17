library(mmit)

dataset <- c("auto93", "autohorse", "autompg", "autoprice", "baskball", "bodyfat", "cloud", "cpu", "meta", "sleep")

for(i in 1:10){
  feature.mat <- read.csv(paste("~/mmit_data/", dataset[i], "/features.csv", sep=""))
  target.mat <- read.csv(paste("~/mmit_data/", dataset[i], "/targets.csv", sep=""))
  
  #rand <- sample(nrow(target.mat))
  #target.mat <- target.mat[rand,]
  #feature.mat <- feature.mat[rand,]
  
  train.feat <- head(feature.mat, nrow(feature.mat)/2)
  test.feat <- tail(feature.mat, nrow(feature.mat)/2)
  
  train.tar <- data.matrix(head(target.mat, nrow(target.mat)/2))
  test.tar <- data.matrix(tail(target.mat, nrow(target.mat)/2))
  
  
  tree <- mmit(train.tar, train.feat, max_depth = 4, margin = 1.0)
  fit <- mmit.predict(tree, test.feat)
  score <- mse(test.tar, fit)
  #print(length(fit))
  print(score)
}
