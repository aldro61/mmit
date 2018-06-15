## zero if inside the interval, 1 if outside
zero_one_loss <- function(y_true, y_pred){
  error <- 0.0
  for(i in 1 : length(y_pred)){
    if(!((y_true[i,0] <= y_pred[i]) && (y_pred[i] <= y_true[i,1]))){
      error <- error + 1.0
    }
  }
  return((error / length(y_pred)))
}

### mean square error
mse <- function(y_true, y_pred){
  error <- 0.0
  for(i in 1 : length(y_pred)){
    if(y_true[i,0] < y_pred[i]){
      error <- error + (y_true[i,0] - y_pred[i])**2
    }
    else if(y_true[i,1] > y_pred[i]){
      error <- error + (y_true[i,1] - y_pred[i])**2
    }
  }
  return((error / length(y_pred)))
}