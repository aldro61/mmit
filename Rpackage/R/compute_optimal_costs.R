compute_optimal_costs <- structure(function
### Compute vector of optimal prediction and cost.
(target.mat,
### n x 2 matrix of limits.
 margin,
### numeric scalar, margin size parameter.
 loss="hinge",
### character scalar, hinge or square.
type="equal",
### type defines if it will compute from non-weighted solver or weighted.
weights = rep(1, nrow(target.mat))
){
  fun.vec <- c(hinge=0L, square=1L)
  if(!(is.character(loss) & length(loss)==1 & loss %in% names(fun.vec))){
    stop("loss must be either hinge or square")
  }
  stopifnot(
    is.numeric(margin),
    length(margin)==1,
    is.finite(margin),
    is.numeric(target.mat),
    is.matrix(target.mat),
    ncol(target.mat)==2)
  if(any(is.na(target.mat))){
    stop("target.mat must not contain missing values")
  }
  lower.vec <- target.mat[, 1]
  upper.vec <- target.mat[, 2]
  stopifnot(-Inf < upper.vec, lower.vec < Inf)
  neg.vec <- rep(-1, nrow(target.mat))
  inf.vec <- rep(Inf, nrow(target.mat))
  if(type != "weighted"){
    result.list <- .C(
      "compute_optimal_costs_interface",
      n_data=nrow(target.mat),
      lower_vec=as.double(lower.vec),
      upper_vec=as.double(upper.vec),
      margin=as.double(margin),
      loss=as.integer(fun.vec[[loss]]),
      moves_vec=as.integer(neg.vec),
      pred_vec=as.double(inf.vec),
      cost_vec=as.double(inf.vec),
      NAOK=TRUE,
      PACKAGE="mmit")
    with(result.list, data.frame(
      moves=moves_vec,
      pred=pred_vec,
      cost=cost_vec))
  }
  else if(type == "weighted"){
    result.list <- .C(
      "weighted_compute_optimal_costs_interface",
      n_data=nrow(target.mat),
      lower_vec=as.double(lower.vec),
      upper_vec=as.double(upper.vec),
      weights=as.double(weights),
      margin=as.double(margin),
      loss=as.integer(fun.vec[[loss]]),
      moves_vec=as.integer(neg.vec),
      pred_vec=as.double(inf.vec),
      cost_vec=as.double(inf.vec),
      NAOK=TRUE,
      PACKAGE="mmit")
    with(result.list, data.frame(
      moves=moves_vec,
      pred=pred_vec,
      cost=cost_vec))
  }
  
### data.frame with columns moves (number of times the pointer was
### moved for each data point, sum of upper and lower limit moves),
### pred (predicted output value that achieves minimum cost), cost
### (minimum cost value).
}, ex=function(){
  library(mmit)
  target.mat <- rbind(
    c(-1, Inf),
    c(-2, 3),
    c(-Inf, 1))
  compute_optimal_costs(target.mat, 0)
  compute_optimal_costs(target.mat, 2)
})
