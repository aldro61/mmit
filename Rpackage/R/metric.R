## zero if inside the interval, 1 if outside
zero_one_loss <- structure(function(y_true, y_pred){
  error <- 0.0
  for(i in 1 : length(y_pred)){
    if(!((y_true[i,1] <= y_pred[i]) && (y_pred[i] <= y_true[i,2]))){
      error <- error + 1.0
    }
  }
  return((error / length(y_pred)))
}, 
### attributes
direction = min,
worst = Inf)

### mean square error
mse <- structure(function(y_true, y_pred){
  error <- 0.0
  for(i in 1 : length(y_pred)){
    if(y_true[i,1] > y_pred[i]){
      error <- error + (y_true[i,1] - y_pred[i])**2
    }
    else if(y_true[i,2] <= y_pred[i]){
      error <- error + (y_true[i,2] - y_pred[i])**2
    }
  }
  return((error / length(y_pred)))
}, 
### attributes
direction = min,
worst= Inf)
