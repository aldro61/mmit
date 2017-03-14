compute_optimal_costs <- structure(function
### Compute vector of optimal prediction and cost.
(feature.vec,
### n-vector of inputs
 target.mat,
### n x 2 matrix of limits.
 margin,
### numeric scalar, margin size parameter.
 loss="hinge"
### character scalar, hinge or square.
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
    ncol(target.mat)==2,
    nrow(target.mat)==length(feature.vec),
    is.numeric(feature.vec),
    is.finite(feature.vec))
  if(any(is.na(target.mat))){
    stop("target.mat must not contain missing values")
  }
  lower.vec <- target.mat[, 1]
  upper.vec <- target.mat[, 2]
  stopifnot(-Inf < upper.vec, lower.vec < Inf)
  neg.vec <- rep(-1, length(feature.vec))
  inf.vec <- rep(Inf, length(feature.vec))
  result.list <- .C(
    "compute_optimal_costs_interface",
    n_data=length(feature.vec),
    feature_vec=as.double(feature.vec),
    lower_vec=as.double(lower.vec),
    upper_vec=as.double(upper.vec),
    margin=as.double(margin),
    loss=as.integer(fun.vec[[loss]]),
    moves_vec=as.integer(neg.vec),
    pred_vec=as.double(inf.vec),
    cost_vec=as.double(inf.vec),
    NAOK=TRUE,
    PACKAGE="MMIT")
  with(result.list, data.frame(
    moves=moves_vec,
    pred=pred_vec,
    cost=cost_vec))
}, ex=function(){
  library(MMIT)
  target.mat <- rbind(
    c(-1, Inf),
    c(-2, 3),
    c(-Inf, 1))
  input.vec <- c(-2.2, 3, 10)
  compute_optimal_costs(input.vec, target.mat, 0)
  compute_optimal_costs(input.vec, target.mat, 1.5)
})
