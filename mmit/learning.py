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
import logging
import numpy as np
import warnings; warnings.filterwarnings("ignore")  # Disable all warnings

from six import iteritems, itervalues
from six.moves import range
from collections import defaultdict, deque
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_consistent_length, check_is_fitted
from sklearn.ensemble.forest import ForestRegressor, _parallel_build_trees
from scipy.sparse import issparse
from contextlib import closing
from multiprocessing import Pool

from .metrics import *
from .core.solver import compute_optimal_costs
from .model import DecisionStump, RegressionTreeNode
from .utils import check_X_y, float_equal, float_less


from sklearn.tree._tree import DTYPE, DOUBLE
from sklearn.externals.joblib import Parallel, delayed


#DTYPE = np.float32


class SolverError(Exception):
    pass


class MaxMarginIntervalTree(BaseEstimator, RegressorMixin):
    def __init__(self, margin=0.0, loss="linear_hinge", max_depth=np.infty,
                 min_samples_split=0, max_features=None, random_state=None):
        """
        Max margin interval tree

        Parameters
        ----------
        margin: float, default: 0
            The margin on each side of the predicted value
        loss: string, default="linear"
            The loss function to use (linear_hinge, squared_hinge)
        max_depth: int, default: infinity
            The maximum depth of the tree
        min_samples_split: int, default: 0
            The minimum number of examples required to split a node
        max_features: None or int or float, default: None
            The maximum number of features to consider at every node split
            (same procedure as in scikit learn)
        random_state: int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator; If RandomState instance, random_state
            is the random number generator; If None, the random number generator is the RandomState instance used by
            np.random. The random state is used for tiebreaking in the recursive partitioning.

        """
        self.margin = margin
        self.loss = loss
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state


    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse="csr")
            if issparse(X) and (X.indices.dtype != np.intc or
                                X.indptr.dtype != np.intc):
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (self.n_features_, n_features))

        return X


    def fit(self, X, y, feature_names=None, level_callback=None,
            split_callback=None, sample_weight=None, check_input=None):
        """
        Fits the decision tree regressor

        Parameters:
        -----------
        X: array-like, dtype: float, shape: [n_examples, n_features]
            The feature matrix
        y: list of tuples or array-like, dtype: float, shape: [n_examples, 2]
            The (lower, upper) bounds of the intervals associated to each example
        feature_names: array-like, dtype: str, shape: [n_features]
            The name of each feature
        level_callback: func(dict), default: None
            Called each time the tree depth increases
        split_callback: func(node), default: None
            Called each time a tree node split

        """
        if level_callback is None:
            def level_callback(x):
                pass

        if split_callback is None:
            def split_callback(x):
                pass

        random_state = check_random_state(self.random_state)

        self.feature_names_ = feature_names

        # Input validation
        if self.max_depth < 1:
            raise ValueError("The maximum tree depth must be greater than 1.")
        X, y = check_X_y(X, y)

        n_samples, self.n_features_ = X.shape

        # Split the intervals into upper and lower bounds
        y_lower, y_upper = zip(*y)
        y_lower = np.array(y_lower, dtype=np.double)
        y_upper = np.array(y_upper, dtype=np.double)

        def _optimal_split(node):
            """
            Selects the best split according to the splitting criterion

            """
            node_y_lower = y_lower[node.example_idx]
            node_y_upper = y_upper[node.example_idx]

            # We keep track of the following values for each optimal split
            best_cost = np.infty
            best_splits = []  # (feature_idx, threshold)
            best_preds = []  # (left_pred, right_pred)
            best_leaf_costs = []  # (left_cost, right_cost)

            # Selection of the feature indices based the max_features
            # hyperparameter
            if self.max_features is None:
                feat_indices = range(X.shape[1])
            elif isinstance(self.max_features, float):
                n_features = max(1, int(X.shape[1] * self.max_features))
                feat_indices = np.random.randint(0, X.shape[1], size=n_features)
            elif isinstance(self.max_features, int):
                assert self.max_features >= 1
                assert self.max_features <= X.shape[1]
                feat_indices = np.random.randint(0, X.shape[1], size=self.max_features)
            else:
                raise ValueError("Invalid type %s for max_features" % type(self.max_features))


            #for feat_idx in range(X.shape[1]):
            for feat_idx in feat_indices:
                feat = X[node.example_idx, feat_idx]

                # Sort the feature values
                sorter = np.argsort(feat)
                feat_sorted = feat[sorter]
                del feat

                # Sort the interval bounds
                lower_sorted = node_y_lower[sorter]
                upper_sorted = node_y_upper[sorter]

                # Get the unique feature values and their first/last occurrences
                unique_feat_sorted, first_idx_by_value = np.unique(feat_sorted, return_index=True)
                last_idx_by_value = [x - 1 for x in np.hstack((first_idx_by_value[1:], [len(feat_sorted)]))]

                if len(unique_feat_sorted) == 1:
                    continue

                # Get the cost values for each side
                _, left_preds, left_costs = compute_optimal_costs(lower_sorted, upper_sorted, self.margin,
                                                                  0 if self.loss == "linear_hinge" else 1)
                _, right_preds, right_costs = compute_optimal_costs(lower_sorted[::-1].copy(), upper_sorted[::-1].copy(),
                                                                    self.margin, 0 if self.loss == "linear_hinge" else 1)

                # XXX: Runtime test case to ensure that the solver is working correctly. The solution for the cases
                # were the left and right leaves contain all the examples should be exactly the same.
                if not float_equal(left_costs[-1], right_costs[-1], 6) or not float_equal(left_preds[-1], right_preds[-1], 6):
                    raise SolverError("""
MMIT solver error. Please report this to the developers.


Details:
========

Nature of error:
----------------
Cost values: Left={0:.9f}  Right={1:.9f}
Pred values: Left={2:.9f}  Right={3:.9f}


Data:
-----
margin: {4:.9f}

loss type: {5!s}

lower_bounds: {6!s}

upper_bounds: {7!s}

END
""".format(left_costs[-1], right_costs[-1], left_preds[-1], right_preds[-1], self.margin, self.loss,
           lower_sorted.tolist(), upper_sorted.tolist()))

                # Combine the values of duplicate feature values and remove splits where all examples are in one leaf
                unique_left_preds = left_preds[last_idx_by_value][:-1]
                unique_left_costs = left_costs[last_idx_by_value][:-1]
                unique_right_preds = right_preds[::-1][first_idx_by_value][1:]
                unique_right_costs = right_costs[::-1][first_idx_by_value][1:]
                cost_by_split = unique_left_costs + unique_right_costs

                # Check for optimality of the split
                if float_less(cost_by_split.min(), node.cost_value):
                    min_cost_idx = cost_by_split.argmin()

                    if float_equal(cost_by_split[min_cost_idx], best_cost):
                        best_splits.append(dict(feat_idx=feat_idx, threshold=unique_feat_sorted[min_cost_idx]))
                        best_preds.append(dict(left=unique_left_preds[min_cost_idx], right=unique_right_preds[min_cost_idx]))
                        best_leaf_costs.append(dict(left=unique_left_costs[min_cost_idx], right=unique_right_costs[min_cost_idx]))
                    elif float_less(cost_by_split[min_cost_idx], best_cost):
                        best_cost = cost_by_split[min_cost_idx]
                        best_splits = [dict(feat_idx=feat_idx, threshold=unique_feat_sorted[min_cost_idx])]
                        best_preds = [dict(left=unique_left_preds[min_cost_idx], right=unique_right_preds[min_cost_idx])]
                        best_leaf_costs = [dict(left=unique_left_costs[min_cost_idx], right=unique_right_costs[min_cost_idx])]

            # No split exists that decreases the objective value
            if np.isinf(best_cost):
                return None, None, None

            logging.debug("There are %d optimal splits with a cost of %.4f", len(best_splits), best_cost)
            keep_idx = random_state.randint(0, len(best_splits))
            best_split = best_splits[keep_idx]
            best_split_leaf_preds = best_preds[keep_idx]
            best_split_leaf_costs = best_leaf_costs[keep_idx]
            del best_splits, best_preds, best_leaf_costs
            best_rule = DecisionStump(best_split["feat_idx"], best_split["threshold"],
                                      self.feature_names_[best_split["feat_idx"]]
                                      if self.feature_names_ is not None else None)

            # Dispatch the examples to the leaves
            best_rule_classifications = best_rule.classify(X[node.example_idx])
            left_child = RegressionTreeNode(parent=node,
                                            depth=node.depth + 1,
                                            example_idx=node.example_idx[best_rule_classifications],
                                            predicted_value=best_split_leaf_preds["left"],
                                            cost_value=best_split_leaf_costs["left"])
            right_child = RegressionTreeNode(parent=node,
                                             depth=node.depth + 1,
                                             example_idx=node.example_idx[~best_rule_classifications],
                                             predicted_value=best_split_leaf_preds["right"],
                                             cost_value=best_split_leaf_costs["right"])

            return best_rule, left_child, right_child

        logging.debug("Training start.")

        self.rule_importances_ = defaultdict(float)

        # Define the root node
        _, preds, costs = compute_optimal_costs(y_lower, y_upper, self.margin, 0 if self.loss == "linear_hinge" else 1)
        root = RegressionTreeNode(depth=0, example_idx=np.arange(len(y)), predicted_value=preds[-1], cost_value=costs[-1])

        # Initialize the tree building procedure
        nodes_to_split = deque([root])
        runtime_infos = {}
        current_depth = -1

        while len(nodes_to_split) > 0:
            node = nodes_to_split.popleft()

            # Check if we have reached a new depth
            if node.depth != current_depth:
                current_depth = node.depth
                runtime_infos["depth"] = current_depth
                logging.debug("The tree depth is {0:d}".format(current_depth))
                if current_depth > 0:
                    level_callback(runtime_infos)
                if current_depth == self.max_depth:
                    # We have reached the nodes of the last level, which must remain leaves
                    logging.debug("The maximum tree depth has been reached. No more leaves will be split.")
                    break

            # Check if the node to split is a pure leaf
            if node.cost_value == 0:
                logging.debug("The leaf is perfectly predicted (cost = 0). It will not be split.")
                continue

            # Check if the HP constraints allows us to split this node
            if node.n_examples < self.min_samples_split:
                logging.debug("The leaf contains less examples (%d) than the minimum required to split (%d) a node. "
                              "It will not be split." % (node.n_examples, self.min_samples_split))
                continue

            # Find the best rule to split the node
            stump, left_child, right_child = _optimal_split(node)

            # If we were incapable of splitting the node into two non-empty leaves
            if stump is None and left_child is None and right_child is None:
                logging.debug("Found no rule to split the node. The node will not be split.")
                continue

            # Perform the split
            logging.debug("Splitting with rule {0!s}.".format(stump))
            node.rule = stump
            node.left_child = left_child
            node.right_child = right_child
            split_callback(node)

            self.rule_importances_[str(node.rule)] += node.cost_value - \
                                                      node.left_child.cost_value - \
                                                      node.right_child.cost_value

            # Add the children to the splitting queue
            nodes_to_split.append(node.left_child)
            nodes_to_split.append(node.right_child)

            # Update the model in the runtime informations
            runtime_infos["model"] = root

        logging.debug("Done building the tree.")

        # Save the model
        self.tree_ = root

        # Normalize the variable importances
        logging.debug("Normalizing the variable importances.")
        variable_importance_sum = sum(v for v in itervalues(self.rule_importances_))
        self.rule_importances_ = {r: float(i) / variable_importance_sum
                                  for r, i in iteritems(self.rule_importances_)}

        logging.debug("Training finished.")

        return self

    def predict(self, X, check_input=None):
        """
        Estimates the labels of some examples

        Parameters:
        -----------
        X: array-like, dtype: float, shape: (n_examples, n_features)
            The feature matrix

        Returns:
        --------
        y_pred: array-like, dtype: float, shape: (n_examples,)
            The predicted labels

        """
        self.check_is_fitted()
        X = check_array(X)
        return self.tree_.predict(X)

    def score(self, X, y):
        """
        Measures the accuracy of the estimator on some examples

        Parameters:
        -----------
        X: array-like, dtype: float, shape: (n_examples, n_features)
            The feature matrix
        y: list of tuples or array-like, dtype: float, shape: (n_examples, 2)
            The (lower, upper) bounds of the intervals associated to each example

        Returns:
        --------
        score: float
            A score measuring the accuracy of the estimator on the examples

        Notes:
        ------
        For compatiblity with Scikit-Learn's model selection methods, the score is the negative mean squared error,
        where any value predicted inside the target interval leads to no error.

        """
        self.check_is_fitted()
        X, y = check_X_y(X, y)
        return -mean_squared_error(y_pred=self.predict(X), y_true=y)

    def check_is_fitted(self):
        return check_is_fitted(self, ["tree_", "rule_importances_"])


def build_interval_tree(args):
    X, y, kwargs = args
    return MaxMarginIntervalTree(**kwargs).fit(X, y)

class RandomForestIntervalRegressor(ForestRegressor):


    # TODO we gotta override the fit method..... threading backend with joblib
    # doesnt speed up python binding + cpp (vs pure cython implementation,
    # which does)

    def __init__(self,
                 n_estimators=10,
                 margin=0.,
                 loss="linear_hinge",
                 max_depth=np.infty,
                 min_samples_split=0,
                 max_features=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=-1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(RandomForestIntervalRegressor, self).__init__(
            base_estimator=MaxMarginIntervalTree(),
            n_estimators=n_estimators,
            estimator_params=("margin", "loss", "max_depth",
                              "min_samples_split", "max_features",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.margin = margin
        self.loss = loss
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features


    def fit(self, X, y, sample_weight=None):

        """Build a forest of trees from the training set (X, y).
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.
        Returns
        -------
        self : object
            Returns self.
        """
        # Validate or convert input data
        X = check_array(X, accept_sparse="csc", dtype=DTYPE)
        y = check_array(y, accept_sparse='csc', ensure_2d=False, dtype=None)
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        # Remap output
        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Check parameters
        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            # for fitting the trees is internally releasing the Python GIL
            # making threading always more efficient than multiprocessing in
            # that case.
            tree_kwargs = {'margin': self.margin,
                           'loss': self.loss,
                           'max_depth': self.max_depth,
                           'min_samples_split': self.min_samples_split,
                           'max_features': self.max_features}

            # TODO make this deterministic
            print("Multiprocessing: creating trees...%s jobs for %s estimators" % (self.n_jobs, n_more_estimators))
            with closing(Pool(self.n_jobs)) as pool:
                iterable = ((X, y, tree_kwargs) for i in range(n_more_estimators))
                trees = pool.map(build_interval_tree, iterable)
            #print("Done, get results")
            #trees = trees.get()
            print("All done!")

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            self._set_oob_score(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self
