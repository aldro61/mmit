#' The Zero One Loss
#'
#' Metric for error calculation where the function gives zero value inside the interval else one. 
#'
#' @param y_true The actual response variable of the model
#' @param y_pred The predicted response value of the model
#' 
#' @return A numeric value which signifies the error quantity.
#' 
#' @author Toby Dylan Hocking, Alexandre Drouin, Torsten Hothorn, Parismita Das
#' 
#' @examples
#' library(mmit)
#' y_true <- rbind(
#'   c(0,1), c(0,1), c(0,1),
#'   c(2,3), c(2,3), c(2,3))
#' 
#' y_pred <- c(0.5, 2, 0, 1.5, 3.5, 2.5)
#' 
#'out <- zero_one_loss(y_true, y_pred)
#'
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


#' The Mean Square Error
#'
#' Metric for mean aquare error calculation.
#'
#' @param y_true The actual response variable of the model
#' @param y_pred The predicted response value of the model
#' 
#' @return A numeric value which signifies the error quantity.
#' 
#' @author Toby Dylan Hocking, Alexandre Drouin, Torsten Hothorn, Parismita Das
#' 
#' @examples
#' library(mmit)
#' y_true <- rbind(
#'   c(0,1), c(0,1), c(0,1),
#'   c(2,3), c(2,3), c(2,3))
#' 
#' y_pred <- c(0.5, 2, 0, 1.5, 3.5, 2.5)
#' 
#'out <- mse(y_true, y_pred)
#' 
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
