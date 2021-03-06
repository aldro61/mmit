% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/mmif.predict.R
\name{predict.mmif}
\alias{predict.mmif}
\title{Predictions with random forests of Max Margin Interval Trees}
\usage{
\method{predict}{mmif}(object, test_feature.mat = NULL, ...)
}
\arguments{
\item{object}{Ensemble of MMITs}

\item{test_feature.mat}{A data frame containing the features of the examples for which predictions must be computed.}

\item{\dots}{...}
}
\value{
Predictions Average output of each tree in the forest
}
\description{
Predictions with random forests of Max Margin Interval Trees
}
\examples{
library(mmit)

target.mat <- rbind(
  c(0,1), c(0,1), c(0,1),
  c(2,3), c(2,3), c(2,3))

feature.mat <- rbind(
  c(1,0,0), c(1,1,0), c(1,2,0),
  c(1,3,0), c(1,4,0), c(1,5,0))

colnames(feature.mat) <- c("a", "b", "c")
feature.mat <- data.frame(feature.mat)

forest <- mmif(feature.mat, target.mat)
pred <- predict(forest, feature.mat)

}
\author{
Toby Dylan Hocking, Alexandre Drouin, Torsten Hothorn, Parismita Das
}
