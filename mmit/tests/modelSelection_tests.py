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
import numpy as np

from unittest import TestCase

from ..core.modelSelection import modelSelection_interface
from ..pruning import cost_complexity

class ModelSelectionTests(TestCase):
    def setUp(self):
        ## From example(exactModelSelection),
        ## https://github.com/tdhock/PeakSegDP/blob/master/R/intervalRegression.R
        self.PoissonLoss=np.array([
            -2655355.59133828,
            -3061722.61406648,
            -3298325.54882865,
            -3345194.75421497,
            -3393017.89561113,
            -3407967.00267479,
        ])
        self.peaks=np.array([0.0, 1, 2, 3, 4, 5])
    def test_interface(self):
        i, breaks = modelSelection_interface(self.PoissonLoss, self.peaks)
        np.testing.assert_allclose(i, [4, 2, 1, 0, -1, -1])
        np.testing.assert_allclose(breaks, [
            1.49491071e+04,   4.73461734e+04,   2.36602935e+05,
            4.06367023e+05,  -1.00000000e+00,  -1.00000000e+00,
        ])

    def test_pruned_path(self):
        d = cost_complexity(self.PoissonLoss, self.peaks)
        np.testing.assert_allclose(d["cost"], [
            -3407967.00267479, -3393017.89561113, -3298325.54882865,
            -3061722.61406648, -2655355.59133828])
        np.testing.assert_allclose(d["complexity"], [
            5, 4, 2, 1, 0])
        np.testing.assert_allclose(d["min_lambda"], [
            0, 14949.1070636669, 47346.1733912372,
            236602.934762176, 406367.022728201])
        np.testing.assert_allclose(d["max_lambda"], [
            14949.1070636669, 47346.1733912372, 236602.934762176,
            406367.022728201, np.infty])

                
