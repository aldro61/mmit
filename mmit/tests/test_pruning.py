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

from unittest import TestCase

from ..pruning import min_cost_complexity_pruning
from ..tree import MaxMarginIntervalTree, RegressionTreeNode


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class PruningTests(TestCase):
    def test_1(self):
        """
        Test initial pruning (Tmax -> T1)
        """
        estimator = MaxMarginIntervalTree()
        estimator.rule_importances_ = []

        root = RegressionTreeNode(0, [], rule="rule1", cost_value=10.)

        # Add a split that does not decrease the objective (should be pruned in the initial pruning)
        root.left_child = RegressionTreeNode(1, [], rule="rule2", cost_value=5, parent=root)
        root.right_child = RegressionTreeNode(1, [], rule="rule3", cost_value=5, parent=root)

        estimator.tree_ = root

        alphas, pruned_trees = min_cost_complexity_pruning(estimator)

        assert len(alphas) == len(pruned_trees) == 1
        assert pruned_trees[0].tree_.is_leaf and pruned_trees[0].tree_.is_root
        np.testing.assert_almost_equal(actual=alphas, desired=[0.])
        np.testing.assert_almost_equal(actual=pruned_trees[0].tree_.cost_value, desired=10.)