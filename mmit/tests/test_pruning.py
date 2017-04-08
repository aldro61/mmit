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
from __future__ import print_function
import numpy as np
import sys

from sklearn.utils.validation import _NotFittedError
from unittest import TestCase

from ..pruning import min_cost_complexity_pruning
from ..tree import MaxMarginIntervalTree, RegressionTreeNode


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class PruningTests(TestCase):
    def test_requires_fitted_estimator(self):
        """
        Requires fitted estimator

        """
        estimator = MaxMarginIntervalTree()
        self.assertRaises(_NotFittedError, min_cost_complexity_pruning, estimator)

    def test_initial_pruning(self):
        """
        Initial pruning (Tmax -> T1)

        """
        estimator = MaxMarginIntervalTree()
        estimator.rule_importances_ = []

        root = RegressionTreeNode(0, [], rule="rule1", cost_value=10.)

        # Add a split that does not decrease the objective (should be pruned in the initial pruning, i.e., alpha = 0)
        root.left_child = RegressionTreeNode(1, [], cost_value=5, parent=root)
        root.right_child = RegressionTreeNode(1, [], cost_value=5, parent=root)

        estimator.tree_ = root

        alphas, pruned_trees = min_cost_complexity_pruning(estimator)

        assert len(alphas) == len(pruned_trees) == 1
        assert pruned_trees[0].tree_.is_leaf and pruned_trees[0].tree_.is_root
        np.testing.assert_almost_equal(actual=alphas, desired=[0.])
        np.testing.assert_almost_equal(actual=pruned_trees[0].tree_.cost_value, desired=10.)

    def test_recursive_pruning_1(self):
        """
        Recursive pruning #1

        """
        estimator = MaxMarginIntervalTree()
        estimator.rule_importances_ = []

        root = RegressionTreeNode(0, [], rule="root", cost_value=10.)

        root_l = root.left_child = RegressionTreeNode(1, [], rule="root_l", cost_value=5, parent=root)
        root_l_l = root_l.left_child = RegressionTreeNode(2, [], cost_value=0, parent=root_l)
        root_l_r = root_l.right_child = RegressionTreeNode(2, [], cost_value=0, parent=root_l)

        root_r = root.right_child = RegressionTreeNode(1, [], rule="root_r", cost_value=3, parent=root)
        root_r_l = root_r.left_child = RegressionTreeNode(2, [], cost_value=0, parent=root_r)
        root_r_r = root_r.right_child = RegressionTreeNode(2, [], cost_value=1, parent=root_r)

        estimator.tree_ = root
        alphas, pruned_trees = min_cost_complexity_pruning(estimator)
        expected_rules = [["root", "root_l", "root_r"],
                          ["root", "root_l"],
                          []]
        expected_alphas = [0., 2., 3.5]
        for i, pt in enumerate(pruned_trees):
            np.testing.assert_equal(actual=sorted(expected_rules[i]), desired=sorted(pt.tree_.rules))
        np.testing.assert_equal(actual=alphas, desired=expected_alphas)

    def test_recursive_pruning_2(self):
        """
        Recursive pruning #1

        """
        estimator = MaxMarginIntervalTree()
        estimator.rule_importances_ = []

        root = RegressionTreeNode(0, [], rule="root", cost_value=100.)

        root_l = root.left_child = RegressionTreeNode(1, [], rule="root_l", cost_value=20, parent=root)
        root_l_l = root_l.left_child = RegressionTreeNode(2, [], cost_value=2, parent=root_l)
        root_l_r = root_l.right_child = RegressionTreeNode(2, [], cost_value=0, parent=root_l)

        root_r = root.right_child = RegressionTreeNode(1, [], rule="root_r", cost_value=10, parent=root)
        root_r_l = root_r.left_child = RegressionTreeNode(2, [], rule="root_r_l", cost_value=2, parent=root_r)
        root_r_r = root_r.right_child = RegressionTreeNode(2, [], cost_value=1, parent=root_r)

        root_r_l_l = root_r_l.left_child = RegressionTreeNode(3, [], cost_value=0, parent=root_r_l)
        root_r_l_r = root_r_l.right_child = RegressionTreeNode(3, [], cost_value=0, parent=root_r_l)

        estimator.tree_ = root
        alphas, pruned_trees = min_cost_complexity_pruning(estimator)
        expected_rules = [["root", "root_l", "root_r", "root_r_l"],
                          ["root", "root_l", "root_r"],
                          ["root", "root_l"],
                          ["root"],
                          []]
        expected_alphas = [0., 2., 7., 18., 70.]
        for i, pt in enumerate(pruned_trees):
            np.testing.assert_equal(actual=sorted(expected_rules[i]), desired=sorted(pt.tree_.rules))
        np.testing.assert_equal(actual=alphas, desired=expected_alphas)

    def test_recursive_pruning_3(self):
        """
        Recursive pruning #3

        """
        estimator = MaxMarginIntervalTree()
        estimator.rule_importances_ = []

        root = RegressionTreeNode(0, [], rule="root", cost_value=100.)

        root_l = root.left_child = RegressionTreeNode(1, [], rule="root_l", cost_value=20, parent=root)
        root_l_l = root_l.left_child = RegressionTreeNode(2, [], cost_value=2, parent=root_l)
        root_l_r = root_l.right_child = RegressionTreeNode(2, [], cost_value=0, parent=root_l)

        root_r = root.right_child = RegressionTreeNode(1, [], rule="root_r", cost_value=10, parent=root)
        root_r_l = root_r.left_child = RegressionTreeNode(2, [], rule="root_r_l", cost_value=2, parent=root_r)
        root_r_r = root_r.right_child = RegressionTreeNode(2, [], cost_value=1, parent=root_r)

        # This is a split that doesnt decrease objective (should be pruned in the initial pruning, i.e., alpha = 0)
        root_r_l_l = root_r_l.left_child = RegressionTreeNode(3, [], cost_value=1, parent=root_r_l)
        root_r_l_r = root_r_l.right_child = RegressionTreeNode(3, [], cost_value=1, parent=root_r_l)

        estimator.tree_ = root
        alphas, pruned_trees = min_cost_complexity_pruning(estimator)
        expected_rules = [["root", "root_l", "root_r"],
                          ["root", "root_l"],
                          ["root"],
                          []]
        expected_alphas = [0., 7., 18., 70.]
        for i, pt in enumerate(pruned_trees):
            np.testing.assert_equal(actual=sorted(expected_rules[i]), desired=sorted(pt.tree_.rules))
        np.testing.assert_equal(actual=alphas, desired=expected_alphas)
