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

from collections import defaultdict, deque
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_consistent_length, check_is_fitted

from .metrics import *
from .core.solver import compute_optimal_costs
from .model import DecisionStump, RegressionTreeNode

float_tol = 1e-6


def _check_X_y(X, y):
    X = check_array(X, force_all_finite=True)
    y = check_array(y, 'csr', force_all_finite=False, ensure_2d=True, dtype=None)
    check_consistent_length(X, y)
    if y.shape[1] != 2:
        raise ValueError("y must contain lower and upper bounds for each interval.")
    return X, y


class SolverError(Exception):
    pass


class MaxMarginIntervalTree(BaseEstimator, RegressorMixin):
    def __init__(self, margin=0.0, loss="hinge", max_depth=np.infty, min_samples_split=0):
        """
        Max margin interval tree

        Parameters
        ----------
        margin: float, default: 0
            The margin on each side of the predicted value
        loss: string, default="hinge"
            The loss function to use (hinge, squared_hinge)
        max_depth: int, default: infinity
            The maximum depth of the tree
        min_samples_split: int, default: 0
            The minimum number of examples required to split a node

        """
        self.margin = margin
        self.loss = loss
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y, level_callback=None, split_callback=None):
        """
        Fits the decision tree regressor

        Parameters:
        -----------
        X: array-like, dtype: float, shape: (n_examples, n_features)
            The feature matrix
        y: list of tuples or array-like, dtype: float, shape: (n_examples, 2)
            The (lower, upper) bounds of the intervals associated to each example
        level_callback: func(dict), default: None
            Called each time the tree depth increases
        split_callback: func(node), default: None
            Called each time a tree node split

        """
        if level_callback is None:
            level_callback = lambda x: None

        if split_callback is None:
            split_callback = lambda x: None

        # Input validation
        if self.max_depth < 1:
            raise ValueError("The maximum tree depth must be greater than 1.")
        X, y = _check_X_y(X, y)

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

            for feat_idx in xrange(X.shape[1]):
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
                                                                  0 if self.loss == "hinge" else 1)
                _, right_preds, right_costs = compute_optimal_costs(lower_sorted[::-1].copy(), upper_sorted[::-1].copy(),
                                                                    self.margin, 0 if self.loss == "hinge" else 1)

                # XXX: Runtime test case to ensure that the solver is working correctly. The solution for the cases
                # were the left and right leaves contain all the examples should be exactly the same.
                if np.abs(left_costs[-1] - right_costs[-1]) > float_tol or np.abs(left_preds[-1] - right_preds[-1]) > float_tol:
                    raise SolverError("MMIT solver error. Please report this to the developers.")

                # Combine the values of duplicate feature values and remove splits where all examples are in one leaf
                unique_left_preds = left_preds[last_idx_by_value][:-1]
                unique_left_costs = left_costs[last_idx_by_value][:-1]
                unique_right_preds = right_preds[::-1][first_idx_by_value][1:]
                unique_right_costs = right_costs[::-1][first_idx_by_value][1:]
                cost_by_split = unique_left_costs + unique_right_costs

                # Cancel all splits that generate leaves that predict infinity or -infinity
                cost_by_split[np.isinf(np.abs(unique_left_preds))] = np.infty
                cost_by_split[np.isinf(np.abs(unique_right_preds))] = np.infty

                # Check for optimality of the split
                if cost_by_split.min() < node.cost_value:
                    min_cost_idx = (unique_left_costs + unique_right_costs).argmin()

                    if np.allclose(cost_by_split[min_cost_idx], best_cost):
                        best_splits.append((feat_idx, unique_feat_sorted[min_cost_idx]))
                        best_preds.append((unique_left_preds[min_cost_idx], unique_right_preds[min_cost_idx]))
                        best_leaf_costs.append((unique_left_costs[min_cost_idx], unique_right_costs[min_cost_idx]))
                    elif cost_by_split[min_cost_idx] < best_cost:
                        best_cost = cost_by_split[min_cost_idx]
                        best_splits = [(feat_idx, unique_feat_sorted[min_cost_idx])]
                        best_preds = [(unique_left_preds[min_cost_idx], unique_right_preds[min_cost_idx])]
                        best_leaf_costs = [(unique_left_costs[min_cost_idx], unique_right_costs[min_cost_idx])]

            # No split exists that decreases the objective value
            if best_cost == np.infty:
                return None, None, None

            # TODO: we could add a tiebreaker
            logging.debug("There are %d optimal splits with a cost of %.4f", len(best_splits), best_cost)
            best_split = best_splits[0]
            best_split_pred = best_preds[0]
            best_split_leaf_costs = best_leaf_costs[0]
            del best_splits, best_preds, best_leaf_costs
            best_rule = DecisionStump(best_split[0], best_split[1])

            # Dispatch the examples to the leaves
            best_rule_classifications = best_rule.classify(X[node.example_idx])
            left_child = RegressionTreeNode(parent=node,
                                            depth=node.depth + 1,
                                            example_idx=node.example_idx[best_rule_classifications],
                                            predicted_value=best_split_pred[0],
                                            cost_value=best_split_leaf_costs[0])
            right_child = RegressionTreeNode(parent=node,
                                             depth=node.depth + 1,
                                             example_idx=node.example_idx[~best_rule_classifications],
                                             predicted_value=best_split_pred[1],
                                             cost_value=best_split_leaf_costs[1])

            return best_rule, left_child, right_child

        logging.debug("Training start.")

        self.rule_importances_ = defaultdict(float)

        # Define the root node
        _, preds, costs = compute_optimal_costs(y_lower, y_upper, self.margin, 0 if self.loss == "hinge" else 1)
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
                logging.debug("The tree depth is %d" % current_depth)
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

            # If we were incapable of splitting the node into two non-empty leafs
            if stump is None:
                logging.debug("Found no rule to split the node. The node will not be split.")
                continue

            # Perform the split
            logging.debug("Splitting with rule %s." % node.rule)
            node.rule = stump
            node.left_child = left_child
            node.right_child = right_child
            split_callback(node)

            # Update rule importances (decrease in cost)
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
        variable_importance_sum = sum(v for v in self.rule_importances_.itervalues())
        self.rule_importances_ = dict((r, i / variable_importance_sum) for r, i in self.rule_importances_.iteritems())

        logging.debug("Training finished.")

    def predict(self, X):
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
        X, y = _check_X_y(X, y)
        return -mean_squared_error(y_pred=self.predict(X), y_true=y)

    def check_is_fitted(self):
        return check_is_fitted(self, ["tree_", "rule_importances_"])


if __name__ == "__main__":
    # Use this to display verbose messages
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s.%(msecs)d %(levelname)s mmit:%(module)s - %(funcName)s: %(message)s")

    np.random.seed(42)
    n_examples = 500
    n_features = 100
    X = np.random.randint(0, 500, (n_examples, n_features)).astype(np.double)
    y_upper = np.random.randint(0, 10, n_examples)
    y_lower = -1 * y_upper + np.random.randint(0, 20, 1)
    y_lower[y_lower > y_upper] = y_upper[y_lower > y_upper]
    y = zip(y_lower, y_upper)
    print "Intervals are:", y
    print

    pred = MaxMarginIntervalTree(margin=0.5, max_depth=10)
    pred.fit(X, y)
    print
    print "Training set predictions:"
    print pred.predict(X)
    print
    print "The tree contains %d rules:" % len(pred.tree_.rules)
    print pred.tree_
    print
    print "The rule importances are:"
    for k, v in pred.rule_importances_.iteritems():
        print "%s: %.3f" % (k, v)

    print pred
    print pred.score(X, y)
