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

from sklearn.base import clone
from sklearn.utils.validation import _NotFittedError
from unittest import TestCase

from .. import MaxMarginIntervalTree
from ..model import DecisionStump


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class MMITTests(TestCase):

    def test_predict_requires_fitted_estimator(self):
        """
        predict requires fit

        """
        estimator = MaxMarginIntervalTree()
        self.assertRaises(_NotFittedError, estimator.predict, [])

    def test_set_params(self):
        """
        set_params

        """
        estimator = MaxMarginIntervalTree()
        parameters = dict(margin=42, loss="squared_hinge", max_depth=3333, min_samples_split=2)
        estimator_with_params = clone(estimator).set_params(**parameters)
        assert np.allclose(estimator_with_params.margin, 42)
        assert estimator_with_params.loss == "squared_hinge"
        assert np.allclose(estimator_with_params.max_depth, 3333)
        assert np.allclose(estimator_with_params.min_samples_split, 2)

    def test_get_params(self):
        """
        get_params

        """
        estimator = MaxMarginIntervalTree(margin=42, loss="squared_hinge", max_depth=3333, min_samples_split=2)
        parameters = estimator.get_params()
        assert np.allclose(parameters["margin"], 42)
        assert parameters["loss"] == "squared_hinge"
        assert np.allclose(parameters["max_depth"], 3333)
        assert np.allclose(parameters["min_samples_split"], 2)

    def test_noiseless_dataset_1_linear(self):
        """
        learning on noiseless dataset #1 (linear hinge)

        """
        y = np.array([[0, 1]] * 5 + [[2, 3]] * 5)
        X = np.hstack((np.ones(len(y)).reshape(-1, 1),  # Bad feature
                       np.arange(len(y)).reshape(-1, 1),  # Good feature
                       np.zeros(len(y)).reshape(-1, 1)))  # Bad feature

        estimator = MaxMarginIntervalTree(margin=0.0, loss="linear_hinge")
        estimator.fit(X, y)

        assert len(estimator.tree_.rules) == 1
        assert estimator.tree_.rules[0] == DecisionStump(1, 4)
        assert estimator.tree_.leaves[0].predicted_value == 0.5
        assert estimator.tree_.leaves[1].predicted_value == 2.5
        assert estimator.tree_.leaves[0].cost_value == 0.0
        assert estimator.tree_.leaves[1].cost_value == 0.0
        
    def test_noiseless_dataset_1_squared(self):
        """
        learning on noiseless dataset #1 (squared hinge)

        """
        y = np.array([[0, 1]] * 5 + [[2, 3]] * 5)
        X = np.hstack((np.ones(len(y)).reshape(-1, 1),  # Bad feature
                       np.arange(len(y)).reshape(-1, 1),  # Good feature
                       np.zeros(len(y)).reshape(-1, 1)))  # Bad feature

        estimator = MaxMarginIntervalTree(margin=0.0, loss="squared_hinge")
        estimator.fit(X, y)

        assert len(estimator.tree_.rules) == 1
        assert estimator.tree_.rules[0] == DecisionStump(1, 4)
        assert estimator.tree_.leaves[0].predicted_value == 0.5
        assert estimator.tree_.leaves[1].predicted_value == 2.5
        assert estimator.tree_.leaves[0].cost_value == 0.0
        assert estimator.tree_.leaves[1].cost_value == 0.0
