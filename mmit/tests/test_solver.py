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
    estimator_tol_decimals = 6

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
            eprint("Random test #{0:d} -- Loss degree: {1:d}".format(i + 1, loss_degree))
            eprint("The margin is: {0:.4f}".format(margin))
            eprint("The lower and upper bounds are:")
            eprint("Lower: {0!s}".format(target_lower.tolist()))
            eprint("Upper: {0!s}".format(target_upper.tolist()))
            eprint("All bounds: {0!s}".format((
                target_lower[~np.isinf(target_lower)].tolist() + target_upper[~np.isinf(target_upper)].tolist())))
            eprint("All signs: {0!s}".format(((np.ones((~np.isinf(target_lower)).sum()) * -1).tolist() + (
            np.ones((~np.isinf(target_upper)).sum()).tolist()))))
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
        Margin=0 yields total cost 0

        """
        margin = 0.0
        moves, preds, costs = solver.compute_optimal_costs(self.target_lower, self.target_upper, margin, 0)
        np.testing.assert_allclose(preds, [-0.9999, 1, 0])  # solver uses an offset of 1e-4 in the \_ case
        np.testing.assert_allclose(costs, [0, 0, 0])

    def test_2(self):
        """
        Margin=0.5 yields total cost 0

        """
        margin = 0.5
        moves, preds, costs = solver.compute_optimal_costs(self.target_lower, self.target_upper, margin, 0)
        np.testing.assert_allclose(preds, [-0.4999, 1, 0])  # solver uses an offset of 1e-4 in the \_ case
        np.testing.assert_allclose(costs, [0, 0, 0])

    def test_3(self):
        """
        Margin=1 yields total cost 0

        """
        margin = 1.0
        moves, preds, costs = solver.compute_optimal_costs(self.target_lower, self.target_upper, margin, 0)
        np.testing.assert_allclose(preds, [1e-04, 1, 0])  # solver uses an offset of 1e-4 in the \_ case
        np.testing.assert_allclose(costs, [0, 0, 0])

    def test_4(self):
        """
        Margin=1.5 yields total cost 1

        """
        margin = 1.5
        moves, preds, costs = solver.compute_optimal_costs(self.target_lower, self.target_upper, margin, 0)
        np.testing.assert_allclose(preds, [0.5001, 1, 0])  # solver uses an offset of 1e-4 in the \_ case
        np.testing.assert_allclose(costs, [0, 0, 1])

    def test_5(self):
        """
        Margin=2 yields total cost 2

        """
        margin = 2
        moves, preds, costs = solver.compute_optimal_costs(self.target_lower, self.target_upper, margin, 0)
        np.testing.assert_allclose(preds, [1.0001, 1, 0.5])  # solver uses an offset of 1e-4 in the \_ case
        np.testing.assert_allclose(costs, [0, 0, 2])

    def test_6(self):
        """
        The zero challenge

        """
        margin = 0
        target_lower = np.zeros(1)
        target_upper = np.zeros(1)
        moves, preds, costs = solver.compute_optimal_costs(target_lower, target_upper, margin, 0)
        np.testing.assert_allclose(preds, [0])
        np.testing.assert_allclose(costs, [0])

    def test_7(self):
        """
        Repeated insertion of the same interval

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
        Solution is independent of the order of the intervals

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
        np.testing.assert_almost_equal(actual=costs[-1], desired=0.8772836396007255, decimal=6)

    def test_real_data_2(self):
        """
        Neuroblastoma: Failing case #2

        """
        margin = 1.0
        target_lower = np.array([-inf, -2.5951410258110204])[::-1].copy()
        target_upper = np.array([3.89204176769069, inf])[::-1].copy()

        moves, preds, costs = solver.compute_optimal_costs(target_lower, target_upper, margin, 1)  # squared hinge
        np.testing.assert_almost_equal(actual=costs[-1], desired=0.0, decimal=6)
        np.testing.assert_almost_equal(actual=preds[-1], desired=0.6484503149986267, decimal=6)

    def test_real_data_3(self):
        """
        Neuroblastoma: Failing case #3

        """
        margin = 1.77827941004
        target_lower = np.array([-2.5332945985918, -3.05095971846127, -1.4283767149598998, -3.14698378261289, -inf, -1.3439328116402098, -0.925521684710516, -2.44906873517344, -0.6702922465439661, -1.5696992592775902, -2.83108828303218, -inf, -2.06720066085223, -0.642426546060613, -2.66253921191746, -4.31207055103816, -inf, -0.523780897279553, -inf, -2.4887060208859397, -3.34992801358554, -2.1639095176731202, -2.82998766093003, -3.2463459566136, -0.925522100513634, -3.75775584278565, -inf, -3.16505292969599, -inf, -2.50332641020796, -1.1425157198608602, -inf])
        target_upper = np.array([inf, inf, inf, 3.1511999578044505, inf, inf, inf, inf, 2.0420300944892498, inf, inf, 2.48942330405145, inf, inf, inf, inf, inf, inf, 1.59408583560705, inf, inf, 2.5995190021667205, inf, inf, inf, inf, inf, inf, inf, 1.4237738979396901, inf, inf])

        moves, preds, costs = solver.compute_optimal_costs(target_lower, target_upper, margin, 0)  # linear hinge
        np.testing.assert_almost_equal(actual=costs[-1], desired=3.94641849103, decimal=6)
        np.testing.assert_almost_equal(actual=preds[-1], desired=0.836998462677, decimal=6)

        moves, preds, costs = solver.compute_optimal_costs(target_lower[::-1].copy(), target_upper[::-1].copy(), margin, 0)  # linear hinge
        np.testing.assert_almost_equal(actual=costs[-1], desired=3.94641849103, decimal=6)
        np.testing.assert_almost_equal(actual=preds[-1], desired=0.836998462677, decimal=6)

    def test_real_data_4(self):
        """
        Olympics linear: Failing case #1

        """
        margin = 0.000001
        target_lower = np.array([-inf, 54.578355, 54.037488])
        target_upper = np.array([54.6099, 55.178355, 54.637488])

        moves, preds, costs = solver.compute_optimal_costs(target_lower, target_upper, margin, 1)  # squared hinge
        np.testing.assert_almost_equal(actual=costs[-1], desired=0., decimal=6)
        np.testing.assert_almost_equal(actual=preds[-1], desired=54.5941275, decimal=6)

        moves, preds, costs = solver.compute_optimal_costs(target_lower[::-1].copy(), target_upper[::-1].copy(), margin,
                                                           1)  # squared hinge
        np.testing.assert_almost_equal(actual=costs[-1], desired=0., decimal=6)
        np.testing.assert_almost_equal(actual=preds[-1], desired=54.5941275, decimal=6)

    def test_real_data_5(self):
        """
        simulated.stepwise.log.0.010: Failing case #1

        """
        margin = 2.000000000
        target_lower = np.array([-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -0.10494, -inf, -0.11109000000000001, -0.097857, -0.093943, -0.09486, -0.111009, -inf, -0.105485, -0.116969, -0.093356, -0.118978, -0.10846700000000001, -0.09351, -0.111631, -inf, -inf, -0.12333499999999999, -0.09592, -0.116345, 0.605308, 0.581073, 0.589102, 0.598496, 0.582113, 0.594679, 0.59917, 0.596532, 0.583136, 0.5851, 0.5932930000000001, -inf, 0.590679, 0.5962569999999999, 0.602003, 0.588213, 0.5913430000000001, 0.598649, 0.591274, 0.601812, 0.9953629999999999, 1.000608, 1.0020950000000002, 1.001477, 1.00435, 1.006402, 1.007487, 0.997906, 1.002613, 0.9940129999999999, 1.0024620000000002, -inf, 1.016635, 1.00226, 1.003679, 1.001785, -inf, 1.004157, 0.9950559999999999, 1.0046389999999998, 1.2874940000000001, 1.294513, 1.273028, 1.277199, -inf, 1.298114, -inf, 1.302461, 1.280309, 1.284982, 1.285259, 1.279859, -inf, 1.291625, 1.291892, 1.271031, 1.292252, 1.280454, -inf, 1.528685, -inf, 1.50353, 1.504086, 1.5135459999999998, 1.5220639999999999, 1.519361, -inf, 1.513385, 1.5158129999999999, 1.509235, 1.504271, 1.484199, 1.529788, 1.523125, 1.491868, 1.5136049999999999, 1.509298, 1.50427, -inf, 1.508068, 1.6796650000000002, 1.717625, 1.6948130000000001, 1.676062, 1.7111919999999998, 1.694552, 1.702184, 1.698814, 1.6845700000000001, 1.688766, 1.681155, 1.6823919999999999, 1.6964169999999998, 1.688765, -inf, 1.6911580000000002, 1.70534, 1.7032450000000001, -inf, 1.6988029999999998, -inf, 1.852022, 1.8393700000000002, 1.844155, 1.852685, 1.8496169999999998, 1.8272599999999999, 1.829818, 1.8410990000000003, 1.8583060000000002, 1.8455970000000002, 1.850845, 1.83713, 1.8434669999999997, 1.8299729999999998, 1.846673, 1.847936, 1.843105, 1.8480919999999998, 1.854786, -inf, 1.972143, -inf, 1.9970849999999998, 1.965541, 1.9924240000000002, 1.9913349999999999, 1.993998, 1.974704, 1.972875, 1.980607, 1.9721240000000002, 1.9768990000000002, 1.979441, -inf, 1.961891, 1.9911849999999998, 1.981583, -inf, 1.990848, 2.103124, 2.112501, 2.091121, 2.110343, 2.104904, 2.101484, 2.0950159999999998, 2.109122, 2.1091439999999997, 2.103975, 2.095898, 2.099242, -inf, 2.099057, 2.113572, 2.088616, 2.1133040000000003, 2.1024700000000003, 2.1100380000000003, 2.106447])
        target_upper = np.array([-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, 0.09506, 0.100839, 0.08891, 0.102143, inf, 0.10514, 0.088991, 0.129059, 0.094515, 0.083031, inf, 0.081022, 0.091533, 0.10649, 0.088369, 0.096142, 0.103672, 0.076665, 0.10408, 0.083655, 0.805308, 0.781073, 0.789102, 0.798496, 0.782113, 0.794679, 0.79917, 0.796532, 0.783136, 0.7851, inf, 0.783219, 0.790679, 0.796257, 0.802003, 0.788213, 0.791343, 0.798649, 0.791274, 0.801812, 1.195363, 1.200608, 1.202095, 1.201477, inf, inf, 1.207487, 1.197906, 1.202613, 1.194013, 1.202462, 1.205496, 1.216635, 1.20226, 1.203679, 1.201785, 1.198553, 1.204157, inf, 1.204639, 1.487494, 1.494513, 1.473028, 1.477199, 1.480064, 1.498114, 1.48911, 1.502461, 1.480309, 1.484982, 1.485259, 1.479859, 1.49189, 1.491625, 1.491892, 1.471031, 1.492252, 1.480454, 1.477445, 1.728685, 1.481343, 1.70353, inf, 1.713546, 1.722064, 1.719361, 1.676259, inf, 1.715813, 1.709235, inf, 1.684199, 1.729788, 1.723125, 1.691868, 1.713605, 1.709298, 1.70427, 1.713183, 1.708068, inf, 1.917625, inf, 1.876062, 1.911192, 1.894552, 1.902184, 1.898814, 1.88457, 1.888766, 1.881155, inf, 1.896417, 1.888765, 1.882561, 1.891158, 1.90534, 1.903245, 1.874546, 1.898803, 2.063508, 2.052022, 2.03937, 2.044155, 2.052685, 2.049617, inf, 2.029818, 2.041099, inf, 2.045597, 2.050845, 2.03713, 2.043467, 2.029973, 2.046673, 2.047936, 2.043105, 2.048092, 2.054786, 2.16948, 2.172143, 2.171181, 2.197085, 2.165541, 2.192424, 2.191335, 2.193998, 2.174704, 2.172875, 2.180607, 2.172124, 2.176899, 2.179441, 2.166317, 2.161891, 2.191185, inf, 2.188154, 2.190848, 2.303124, 2.312501, 2.291121, 2.310343, inf, 2.301484, inf, 2.309122, 2.309144, 2.303975, 2.295898, 2.299242, 2.295398, 2.299057, 2.313572, 2.288616, 2.313304, 2.30247, 2.310038, 2.306447])

        moves, preds, costs = solver.compute_optimal_costs(target_lower, target_upper, margin, 0)  # linear hinge
        np.testing.assert_almost_equal(costs[-1], 606.979822, decimal=6)
        np.testing.assert_almost_equal(preds[-1], 0.309591, decimal=6)

        moves, preds, costs = solver.compute_optimal_costs(target_lower[::-1].copy(), target_upper[::-1].copy(), margin,
                                                           0)  # linear hinge
        np.testing.assert_almost_equal(costs[-1], 606.979822, decimal=6)
        np.testing.assert_almost_equal(preds[-1], 0.309591, decimal=6)