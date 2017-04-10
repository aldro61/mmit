"""
    MMIT: Max Margin Interval Trees
    Copyright (C) 2017 Toby Dylan Hocking, Alexandre Drouin
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from math import sqrt

import numpy as np
from builtins import range
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection._search import check_cv, check_scoring, check_is_fitted
from sklearn.utils.validation import indexable

from .learning import MaxMarginIntervalTree
from .pruning import min_cost_complexity_pruning
from .utils import BetweenDict, check_X_y


def _fit_and_score(estimator, X, y, cv, parameters, scorer=None):
    """
    Performs cross-validation and pruning for MMIT estimators

    """
    n_folds = cv.get_n_splits()

    # Get the indices of the training and testing examples of each fold
    fold_split_idx = list(cv.split(X, y))

    # Initialize the trees to be grown
    logging.debug("Planting seeds")
    fold_predictors = [clone(estimator).set_params(**parameters) for _ in range(n_folds)]
    master_predictor = clone(estimator).set_params(**parameters)

    # For each fold, build an overgrown decision tree
    logging.debug("Growing the cross-validation fold trees")
    for i, (fold_train_idx, _) in enumerate(fold_split_idx):
        logging.debug("Growing the tree for fold %d" % (i + 1))
        # Fit the decision tree
        fold_predictors[i].fit(X[fold_train_idx], y[fold_train_idx])

    # Also build an overgrown decision tree on the entire dataset
    logging.debug("Growing the master tree")
    master_predictor.fit(X, y)

    # Get the pruned master and cross-validation trees
    master_alphas, master_pruned_trees = min_cost_complexity_pruning(master_predictor)
    fold_alphas = []
    fold_pruned_trees = []
    for i in range(n_folds):
        alphas, trees = min_cost_complexity_pruning(fold_predictors[i])
        fold_alphas.append(alphas)
        fold_pruned_trees.append(trees)

    # Compute the test risk for all pruned trees of each fold
    alpha_path_scores_by_fold = []
    for i, (_, fold_test_idx) in enumerate(fold_split_idx):
        fold_test_scores = []
        alpha_path_scores = BetweenDict()
        for j, t in enumerate(fold_pruned_trees[i]):
            fold_test_score = scorer(t, X[fold_test_idx], y[fold_test_idx])
            fold_test_scores.append(fold_test_score)
            if j < len(fold_alphas[i]) - 1:
                key = (fold_alphas[i][j], fold_alphas[i][j + 1])
            else:
                key = (fold_alphas[i][j], np.infty)
            alpha_path_scores[key] = fold_test_score
        alpha_path_scores_by_fold.append(alpha_path_scores)

    # Prune the master tree based on the CV estimates
    min_score = -np.infty
    min_score_tree = None
    for i in range(len(master_alphas) - 1):
        geo_mean_alpha_k = sqrt(master_alphas[i] * master_alphas[i + 1])
        cv_score = np.mean([alpha_path_scores_by_fold[j][geo_mean_alpha_k] for j in range(n_folds)])
        # Note: assumes that alphas are sorted in increasing order, so simplest solution is always preferred
        if cv_score >= min_score:
            min_score = cv_score
            min_score_tree = master_pruned_trees[i]

    return {"score": min_score, "estimator": min_score_tree, "params": parameters}


class GridSearchCV(BaseEstimator):
    def __init__(self, estimator, param_grid={}, cv=None, n_jobs=1, pre_dispatch='2*n_jobs', scoring=None):
        """
        Parameters
        ----------
        estimator : estimator object.
            This is assumed to implement the scikit-learn estimator interface.
            Either estimator needs to provide a ``score`` function,
            or ``scoring`` must be passed.
        param_grid : dict or list of dictionaries
            Dictionary with parameters names (string) as keys and lists of
            parameter settings to try as values, or a list of such
            dictionaries, in which case the grids spanned by each dictionary
            in the list are explored. This enables searching over any sequence
            of parameter settings.
        scoring : string, callable or None, default=None
            A string (see model evaluation documentation) or
            a scorer callable object / function with signature
            ``scorer(estimator, X, y)``.
            If ``None``, the ``score`` method of the estimator is used.
        n_jobs : int, default=1
            Number of jobs to run in parallel.
        pre_dispatch : int, or string, optional
            Controls the number of jobs that get dispatched during parallel
            execution. Reducing this number can be useful to avoid an
            explosion of memory consumption when more jobs get dispatched
            than CPUs can process. This parameter can be:
                - None, in which case all the jobs are immediately
                  created and spawned. Use this for lightweight and
                  fast-running jobs, to avoid delays due to on-demand
                  spawning of the jobs
                - An int, giving the exact number of total jobs that are
                  spawned
                - A string, giving an expression as a function of n_jobs,
                  as in '2*n_jobs'
        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default 3-fold cross validation,
              - integer, to specify the number of folds in a `(Stratified)KFold`,
              - An object to be used as a cross-validation generator.
              - An iterable yielding train, test splits.
            For integer/None inputs, if the estimator is a classifier and ``y`` is
            either binary or multiclass, :class:`StratifiedKFold` is used. In all
            other cases, :class:`KFold` is used.
            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validation strategies that can be used here.

        """
        self.estimator = estimator
        self.cv = cv
        self.n_jobs = n_jobs
        self.param_grid = param_grid
        self.pre_dispatch = pre_dispatch
        self.scoring = scoring

        if not isinstance(self.estimator, MaxMarginIntervalTree):
            raise ValueError("The provided estimator is not of type %s." % str(MaxMarginIntervalTree))

    def fit(self, X, y, groups=None):
        """Run fit with all sets of parameters.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        """
        X, y = check_X_y(X, y)

        estimator = self.estimator
        cv = check_cv(self.cv, y)
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)

        # Get all combinations of hyperparameters
        candidate_params = list(ParameterGrid(self.param_grid))
        n_candidates = len(candidate_params)
        logging.debug("Fitting {0} folds for each of {1} candidates, totalling {2} fits"
                      .format(n_splits, n_candidates, n_candidates * n_splits))

        # Score all parameter combinations in parallel
        cv_results = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)\
                (delayed(_fit_and_score)(clone(self.estimator), X, y, cv, parameters, self.scorer_)
                 for parameters in candidate_params)

        # Find the best parameters based on the CV score
        best_result = {"score": -np.infty, "estimator": clone(estimator), "params": {}}
        for result in cv_results:
            if np.allclose(result["score"], best_result["score"]) and \
                           len(result["estimator"].tree_) < len(best_result["estimator"].tree_):
                best_result = result
            elif np.greater(result["score"], best_result["score"]):
                best_result = result

        # Save the results
        self.best_estimator_ = best_result["estimator"]
        self.best_score_ = best_result["score"]
        self.best_params_ = best_result["params"]
        self.cv_results_ = cv_results

        return self

    def _check_is_fitted(self):
        check_is_fitted(self, ["best_estimator_", "best_params_", "cv_results_", "scorer_"])

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    def score(self, X, y=None):
        """Returns the score on the given data.
        This uses the score defined by ``scoring`` where provided, and the
        ``best_estimator_.score`` method otherwise.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input data, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target relative to X

        Returns
        -------
        score : float
        """
        self._check_is_fitted()
        return self.scorer_(self.best_estimator_, X, y)

    def predict(self, X):
        """Call predict on the estimator with the best found parameters.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
        self._check_is_fitted()
        return self.best_estimator_.predict(X)

    def decision_function(self, X):
        """Call decision_function on the estimator with the best found parameters.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
        self._check_is_fitted()
        return self.best_estimator_.decision_function(X)

    def transform(self, X):
        """Call transform on the estimator with the best found parameters.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
        self._check_is_fitted()
        return self.best_estimator_.transform(X)