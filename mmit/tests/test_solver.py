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

from numpy import infty as inf
from unittest import TestCase

from ..core import solver


def breakpoint_pos(y, s, e):
    return y - s * e


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def loss(u, y, s, e, degree):
    f = 1.0 * s * (u - y) + e
    f[f < 0] = 0.
    return f ** degree


def _random_testing(loss_degree):
    n_tests = 1000
    n_points = 100
    n_decimals = 2
    estimator_sample_size = 10000
    estimator_tol_decimals = 5

    for i in xrange(n_tests):
        # Random test case generation
        margin = round(np.random.rand(), n_decimals)
        target_lower = np.round(np.random.rand(n_points), n_decimals)
        target_upper = np.round(np.random.rand(n_points), n_decimals)

        n_inf = np.random.randint(n_points)
        target_lower[: n_inf] = -np.infty

        n_inf = np.random.randint(n_points)
        target_upper[: n_inf] = np.infty

        np.random.shuffle(target_lower)
        np.random.shuffle(target_upper)

        # Get the solver's solution
        moves, preds, costs = solver.compute_optimal_costs(target_lower, target_upper, margin, loss_degree - 1)

        # Empirical minimum approximation
        # --------------------------------

        # Remove infinity values
        target_lower = target_lower[~np.isinf(target_lower)]
        target_upper = target_upper[~np.isinf(target_upper)]

        # Determine the interval where all breakpoints are contained
        min_breakpoint = min(breakpoint_pos(min(target_lower), -1, margin),
                             breakpoint_pos(min(target_upper), 1, margin))
        max_breakpoint = max(breakpoint_pos(max(target_lower), -1, margin),
                             breakpoint_pos(max(target_upper), 1, margin))

        # Gross sampling of the interval
        x = np.linspace(min_breakpoint - 1, max_breakpoint + 1, estimator_sample_size)
        min_idx = np.argmin(sum(loss(x, b, -1, margin, loss_degree) for b in target_lower) + sum(
            loss(x, b, 1, margin, loss_degree) for b in target_upper))

        # Fine sampling around the minimum
        x2 = np.linspace(x[min_idx - 1], x[min_idx + 1], estimator_sample_size)
        estimated_min = np.min(sum(loss(x2, b, -1, margin, loss_degree) for b in target_lower) + sum(
            loss(x2, b, 1, margin, loss_degree) for b in target_upper))

        # Check solution
        try:
            np.testing.assert_almost_equal(costs[-1], estimated_min, decimal=estimator_tol_decimals)
        except AssertionError:
            eprint()
            eprint("Random test #%d -- Loss degree: %d" % (i + 1, loss_degree))
            eprint("The margin is: %.4f" % margin)
            eprint("The lower and upper bounds are:")
            eprint("Lower: %s" % target_lower.tolist())
            eprint("Upper: %s" % target_upper.tolist())
            eprint("All bounds: %s" % (
                target_lower[~np.isinf(target_lower)].tolist() + target_upper[~np.isinf(target_upper)].tolist()))
            eprint("All signs: %s" % ((np.ones((~np.isinf(target_lower)).sum()) * -1).tolist() + (
            np.ones((~np.isinf(target_upper)).sum()).tolist())))
            eprint("Cost -- expected:", estimated_min, ", actual:", costs[-1])
            eprint("\n\n")
            raise AssertionError()


class SolverTests(TestCase):
    def setUp(self):
        self.target = [(-1, np.infty),
                       (-2, 3),
                       (-np.infty, 1)]
        self.target_lower, self.target_upper = zip(*self.target)
        self.target_lower = np.array(self.target_lower)
        self.target_upper = np.array(self.target_upper)

    def test_1(self):
        """
        margin=0 yields total cost 0
        """
        margin = 0.0
        moves, preds, costs = solver.compute_optimal_costs(self.target_lower, self.target_upper, margin, 0)
        np.testing.assert_allclose(preds, [inf, 1, 0])
        np.testing.assert_allclose(costs, [0, 0, 0])

    def test_2(self):
        """
        margin=0.5 yields total cost 0
        """
        margin = 0.5
        moves, preds, costs = solver.compute_optimal_costs(self.target_lower, self.target_upper, margin, 0)
        np.testing.assert_allclose(preds, [inf, 1, 0])
        np.testing.assert_allclose(costs, [0, 0, 0])

    def test_3(self):
        """
        margin=1 yields total cost 0
        """
        margin = 1.0
        moves, preds, costs = solver.compute_optimal_costs(self.target_lower, self.target_upper, margin, 0)
        np.testing.assert_allclose(preds, [inf, 1, 0])
        np.testing.assert_allclose(costs, [0, 0, 0])

    def test_4(self):
        """
        margin=1.5 yields total cost 1
        """
        margin = 1.5
        moves, preds, costs = solver.compute_optimal_costs(self.target_lower, self.target_upper, margin, 0)
        np.testing.assert_allclose(preds, [inf, 1, 0])
        np.testing.assert_allclose(costs, [0, 0, 1])

    def test_5(self):
        """
        margin=2 yields total cost 2
        """
        margin = 2
        moves, preds, costs = solver.compute_optimal_costs(self.target_lower, self.target_upper, margin, 0)
        np.testing.assert_allclose(preds, [inf, 1, 0.5])
        np.testing.assert_allclose(costs, [0, 0, 2])

    def test_6(self):
        """
        the zero challenge
        """
        margin = 0
        target_lower = np.zeros(1)
        target_upper = np.zeros(1)
        moves, preds, costs = solver.compute_optimal_costs(target_lower, target_upper, margin, 0)
        np.testing.assert_allclose(preds, [0])
        np.testing.assert_allclose(costs, [0])

    def test_7(self):
        """
        repeated insertion of the same interval
        """
        margin = 0
        # Create a \x/ with slopes of -3 and +3
        target_lower = np.array([0, 0, 0], dtype=np.double)
        target_upper = np.array([0, 0, 0], dtype=np.double)
        # The optimal cost and prediction should be 0
        moves, preds, costs = solver.compute_optimal_costs(target_lower, target_upper, margin, 0)
        np.testing.assert_allclose(preds, [0, 0, 0])
        np.testing.assert_allclose(costs, [0, 0, 0])
        # Now add points until the minimum changes. It should take 3 insertions to get a flat minimum region.
        # First
        target_lower = np.hstack((target_lower, [-inf]))
        target_upper = np.hstack((target_upper, [-1]))
        moves, preds, costs = solver.compute_optimal_costs(target_lower, target_upper, margin, 0)
        np.testing.assert_allclose(preds, [0, 0, 0, 0])
        np.testing.assert_allclose(costs, [0, 0, 0, 1])
        # Second
        target_lower = np.hstack((target_lower, [-inf]))
        target_upper = np.hstack((target_upper, [-1]))
        moves, preds, costs = solver.compute_optimal_costs(target_lower, target_upper, margin, 0)
        np.testing.assert_allclose(preds, [0, 0, 0, 0, 0])
        np.testing.assert_allclose(costs, [0, 0, 0, 1, 2])
        # Third
        target_lower = np.hstack((target_lower, [-inf]))
        target_upper = np.hstack((target_upper, [-1]))
        moves, preds, costs = solver.compute_optimal_costs(target_lower, target_upper, margin, 0)
        np.testing.assert_allclose(preds, [0, 0, 0, 0, 0, -0.5])
        np.testing.assert_allclose(costs, [0, 0, 0, 1, 2, 3])

    def test_8(self):
        """
        solution is independent of the order of the intervals
        """
        margin = 0
        moves, preds, costs = solver.compute_optimal_costs(self.target_lower, self.target_upper, margin, 0)
        unshuffled_pred = preds[-1]
        unshuffled_cost = costs[-1]
        del moves, preds, costs

        for i in xrange(10):
            shuffler = np.arange(len(self.target_lower))
            np.random.shuffle(shuffler)
            moves, preds, costs = solver.compute_optimal_costs(self.target_lower[shuffler], self.target_upper[shuffler], margin, 0)
            assert preds[-1] == unshuffled_pred
            assert costs[-1] == unshuffled_cost

    def test_random_hinge(self):
        """
        Random testing with hinge loss
        """
        _random_testing(loss_degree=1)

    def test_random_squared_hinge(self):
        """
        Random testing with squared hinge loss
        """
        _random_testing(loss_degree=2)

    def test_real_data_1(self):
        """
        Neuroblastoma: Failing case #1
        """
        margin = 1.0
        target_lower = np.array([-1.7261263440487502, -0.528013735027211, -0.33134036358054303, -1.4050923044434198, -2.064918351765, 0.00015573501047014, 0.9155816763487209, -0.326021571684791, 0.6788199124179188])
        target_upper = np.array([inf, 1.6685363802387199, inf, inf, inf, inf, inf, 2.60996339512829, inf])

        moves, preds, costs = solver.compute_optimal_costs(target_lower, target_upper, margin, 1)  # squared hinge
        np.testing.assert_almost_equal(actual=costs[-1], desired=0.8772836396007255)

    def test_real_data_2(self):
        """
        Neuroblastoma: Failing case #2
        """
        margin = 1.0
        target_lower = np.array([-inf, -2.5951410258110204])[::-1].copy()
        target_upper = np.array([3.89204176769069, inf])[::-1].copy()

        moves, preds, costs = solver.compute_optimal_costs(target_lower, target_upper, margin, 1)  # squared hinge
        np.testing.assert_almost_equal(actual=costs[-1], desired=0.0)
        np.testing.assert_almost_equal(actual=preds[-1], desired=0.6484503149986267)
