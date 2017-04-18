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

from ..model import RegressionTreeNode, DecisionStump


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class TreeTests(TestCase):
    def setUp(self):
        self.root = RegressionTreeNode(0, [1, 2, 3, 4, 5, 6, 7, 8], rule=DecisionStump(0, 5), cost_value=10., predicted_value=1.)
        self.root_left = RegressionTreeNode(1, [1, 2, 3, 4], rule=DecisionStump(1, 10), cost_value=5., predicted_value=10., parent=self.root)
        self.root_right = RegressionTreeNode(1, [5, 6, 7, 8], cost_value=2., predicted_value=0.1, parent=self.root)
        self.root.left_child = self.root_left
        self.root.right_child = self.root_right
        self.root_left_left = RegressionTreeNode(2, [1, 2], cost_value=0., predicted_value=1., parent=self.root.left_child)
        self.root_left_right = RegressionTreeNode(2, [3, 4], cost_value=0., predicted_value=0., parent=self.root.left_child)
        self.root.left_child.left_child = self.root_left_left
        self.root.left_child.right_child = self.root_left_right

    def test_is_leaf_single_node_tree(self):
        """
        is_leaf -- single node tree

        """
        assert RegressionTreeNode(0, []).is_leaf

    def test_is_leaf_general_case(self):
        """
        is_leaf -- general case

        """
        assert not self.root.is_leaf
        assert not self.root.left_child.is_leaf
        assert self.root.left_child.left_child.is_leaf
        assert self.root.left_child.right_child.is_leaf
        assert self.root.right_child.is_leaf

    def test_is_root_single_node_tree(self):
        """
        is_root -- single node tree

        """
        assert RegressionTreeNode(0, []).is_root

    def test_is_root_general_case(self):
        """
        is_root -- general case

        """
        assert self.root.is_root
        assert not self.root.left_child.is_root
        assert not self.root.left_child.left_child.is_root
        assert not self.root.left_child.right_child.is_root
        assert not self.root.right_child.is_root

    def test_get_leaves_single_node_tree(self):
        """
        get_leaves -- single node tree

        """
        tree = RegressionTreeNode(0, [])
        assert len(tree.leaves) == 1
        assert all(actual == expected for actual, expected in zip(tree.leaves, [tree]))

    def test_get_leaves_general_case(self):
        """
        get_leaves -- general case

        """
        expected = [self.root_right, self.root_left_left, self.root_left_right]
        assert len(self.root.leaves) == len(expected)
        assert all(actual in expected for actual in self.root.leaves)

    def test_n_examples(self):
        """
        n_examples

        """
        assert self.root.n_examples == 8
        assert self.root.left_child.n_examples == 4
        assert self.root.left_child.left_child.n_examples == 2
        assert self.root.left_child.right_child.n_examples == 2
        assert self.root.right_child.n_examples == 4

    def test_rules_single_node_tree(self):
        """
        get_rules -- single node tree

        """
        assert len(RegressionTreeNode(0, []).rules) == 0

    def test_rules_general_case(self):
        """
        get_rules -- general case

        """
        expected = [DecisionStump(0, 5), DecisionStump(1, 10)]
        assert len(self.root.rules) == len(expected)
        assert all(actual in expected for actual in self.root.rules)

    def test_len_single_node_tree(self):
        """
        len -- single node tree

        """
        assert len(RegressionTreeNode(0, []).rules) == 0

    def test_len_general_case(self):
        """
        len -- general case

        """
        assert len(self.root) == 2

    def test_predict(self):
        """
        predict

        """
        X = np.array([[5, 10]])
        assert np.allclose(self.root.predict(X), self.root_left_left.predicted_value)

        X = np.array([[-999., -1000.]])
        assert np.allclose(self.root.predict(X), self.root_left_left.predicted_value)

        X = np.array([[5., 10.0001]])
        assert np.allclose(self.root.predict(X), self.root_left_right.predicted_value)

        X = np.array([[5., 10000.]])
        assert np.allclose(self.root.predict(X), self.root_left_right.predicted_value)

        X = np.array([[5.0001, 0.]])
        assert np.allclose(self.root.predict(X), self.root_right.predicted_value)

        X = np.array([[5.0001, 100.]])
        assert np.allclose(self.root.predict(X), self.root_right.predicted_value)

        X = np.array([[10000., 0.]])
        assert np.allclose(self.root.predict(X), self.root_right.predicted_value)
