\name{compute_optimal_costs}
\alias{compute_optimal_costs}
\title{compute optimal costs}
\description{Compute vector of optimal prediction and cost.}
\usage{compute_optimal_costs(target.mat, margin, loss = "hinge")}
\arguments{
  \item{target.mat}{n x 2 matrix of limits.}
  \item{margin}{numeric scalar, margin size parameter.}
  \item{loss}{character scalar, hinge or square.}
}

\value{data.frame with columns moves (number of times the pointer was
moved for each data point, sum of upper and lower limit moves),
pred (predicted output value that achieves minimum cost), cost
(minimum cost value).}

\author{Toby Dylan Hocking, Alexandre Drouin}




\examples{
library(mmit)
target.mat <- rbind(
  c(-1, Inf),
  c(-2, 3),
  c(-Inf, 1))
compute_optimal_costs(target.mat, 0)
compute_optimal_costs(target.mat, 2)
}
