# Max-margin Interval Trees

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://travis-ci.org/aldro61/mmit.svg?branch=master)](https://travis-ci.org/aldro61/mmit)
[![Code Issues](https://www.quantifiedcode.com/api/v1/project/334605d421f0460ba846af83ebff870d/badge.svg)](https://www.quantifiedcode.com/app/project/334605d421f0460ba846af83ebff870d)

Regression trees for interval censored output data

Drouin, A., Hocking, T. D. & Laviolette, F. (2017). Maximum Margin Interval Trees. Under review.

## Python package installation and usage

The source code of the `mmit` Python package is located in the [mmit](mmit) sub-directory, and can be installed via the following command.

```
python setup.py install
```

The package includes a [scikit-learn style class](https://github.com/aldro61/mmit/blob/master/mmit/learning.py#L35) to learn Maximum Margin Interval Trees.

## R package installation and usage

The source code of the `mmit` R package is located in the
[Rpackage](Rpackage) sub-directory, and can be installed via the following R
commands.

```
if(!require(devtools))install.package("devtools")
devtools::install_github("aldro61/mmit/Rpackage")
```

The package currently provides an interface for the dynamic programming algorithm used to train Maximum Margin Interval Trees. Support for learning trees in R will be implemented in the near future.
