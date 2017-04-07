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

from copy import deepcopy


def min_cost_complexity_pruning(estimator):
    """
    Prunes a decision tree using minimum cost-complexity pruning

    Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984).
    Classification and regression trees. Wadsworth & Brooks. Monterey, CA.

    Parameters:
    -----------
    estimator: MaxMarginIntervalTree
        The regression tree model to prune

    Returns:
    --------
    alphas: array-like, dtype: np.float, shape: (n_pruned_trees,)
        The sequence of increasing alpha values leading to the pruning of some nodes. Larger alpha values lead to
        stronger regularization. The last value results in a tree containing only the root node as a leaf.
    pruned_trees: array-like, dtype: MaxMarginIntervalTree, shape: (n_pruned_trees,)
        The sequence of increasingly small trees resulting from minimum cost-complexity pruning with each value of alpha

    """
    def _initial_pruning(root):

        # Get all nodes that have two leaves as children
        def _get_leaf_parents(root):
            def __get_leaf_parents(node):
                leaf_parents = []
                if not node.is_leaf:
                    if node.left_child.is_leaf and node.right_child.is_leaf:
                        leaf_parents.append(node)
                    else:
                        leaf_parents += __get_leaf_parents(node.left_child)
                        leaf_parents += __get_leaf_parents(node.right_child)
                return leaf_parents
            return __get_leaf_parents(root)

        # Perform the initial pruning (Tmax -> T1)
        def __initial_pruning(parents):
            if len(parents) == 0:
                return
            node = parents.pop()
            if np.allclose(node.cost_value, node.left_child.cost_value + node.right_child.cost_value):
                logging.debug("Converting node %s into a leaf" % node)
                del node.rule
                node.rule = None
                del node.left_child
                node.left_child = None
                del node.right_child
                node.right_child = None
                if not node.is_root and node.parent.left_child.is_leaf and node.parent.right_child.is_leaf:
                    logging.debug("Adding the new leaf's parent to the list of leaf parents")
                    parents.append(node.parent)
            __initial_pruning(parents)
        __initial_pruning(_get_leaf_parents(root))

    def _find_weakest_links(root):
        def __find_weakest_links(node):
            if node.is_leaf:
                return np.inf, [node]
            else:
                C_Tt = sum(l.cost_value for l in node.leaves)
                current_gt = float(node.cost_value - C_Tt) / (len(node.leaves) - 1)
                left_min_gt, left_weakest_links = __find_weakest_links(node.left_child)
                right_min_gt, right_weakest_links = __find_weakest_links(node.right_child)

                if np.allclose(current_gt, min(left_min_gt, right_min_gt)):
                    if np.allclose(left_min_gt, right_min_gt):
                        return current_gt, [node] + left_weakest_links + right_weakest_links
                    else:
                        return current_gt, [node] + (left_weakest_links if left_min_gt < right_min_gt
                                                     else right_weakest_links)
                elif current_gt < min(left_min_gt, right_min_gt):
                    return current_gt, [node]
                elif np.allclose(left_min_gt, right_min_gt):
                    return left_min_gt, left_weakest_links + right_weakest_links
                elif left_min_gt > right_min_gt:
                    return right_min_gt, right_weakest_links
                elif left_min_gt < right_min_gt:
                    return left_min_gt, left_weakest_links
                else:
                    raise Exception("Unhandled case detected!")
        return __find_weakest_links(root)

    def _sequential_prune(root):
        Tmax = deepcopy(root)

        logging.debug("Initial pruning (Tmax >> T1)")
        _initial_pruning(Tmax)
        T1 = Tmax
        del Tmax

        def __sequential_prune(root):
            root = deepcopy(root)

            # Find the weakest link in the tree
            logging.debug("Computing link strenghts for each node")
            min_gt, weakest_links = _find_weakest_links(root)

            # Prune the weakest link (and save alpha)
            logging.debug("Pruning occurs at alpha %.9f" % min_gt)
            for n in weakest_links:
                del n.rule
                n.rule = None
                del n.left_child
                n.left_child = None
                del n.right_child
                n.right_child = None

            # Repeat until only the root node remains
            return [(min_gt, root)] + (__sequential_prune(root) if not root.is_leaf else [])

        logging.debug("Pruning sequentially until only the root remains (T1 >> ... >> {root}")
        return [(0, T1)] + __sequential_prune(T1)

    # Validate that the decision tree has been fitted
    estimator.check_is_fitted()

    logging.debug("Extracting the model from the estimator")
    root = deepcopy(estimator.tree_)

    logging.debug("Generating the sequence of pruned trees")
    alphas, trees = zip(*_sequential_prune(root))

    logging.debug("Creating MaxMarginIntervalTree estimators for each pruned tree")
    pruned_trees = []
    for t in trees:
        p = deepcopy(estimator)
        p.tree_ = t
        pruned_trees.append(p)

    return alphas, pruned_trees
