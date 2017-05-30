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


def zero_one_loss(y_true, y_pred):
    """
    Proportion of the predicted values that are not in their target interval

    """
    if len(y_pred) != len(y_true):
        raise ValueError("The number of predictions must match the number of target interval values.")
    return float(1.0 * sum(0 if (t[0] <= p <= t[1]) else 1 for p, t in zip(y_pred, y_true)) / len(y_pred))


def mean_squared_error(y_true, y_pred):
    """
    The mean squared distance to the interval bounds

    """
    if len(y_pred) != len(y_true):
        raise ValueError("The number of predictions must match the number of target interval values.")
    return float(1.0 * sum((p - t[0])**2 if p < t[0] else ((p - t[1])**2 if t[1] < p else 0.0) for p, t in zip(y_pred, y_true)) / len(y_pred))
