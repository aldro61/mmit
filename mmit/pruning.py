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
from .core.modelSelection import modelSelection_interface

def cost_complexity(cost, complexity):
    """cost-complexity pruning

    Solves min_i cost[i] + lambda*complexity[i] for all lambda,
    returning a dictionary with four numpy arrays:
    cost: the cost of each model that is selected for at least one lambda,
    complexity: the selection

    """
    i, breaks = modelSelection_interface(cost, complexity)
    ok = 0 <= i
    selected_i = np.append([len(cost)-1], i[ok])
    min_lambda = np.append([0.0], breaks[ok])
    max_lambda = np.append(breaks[ok], [np.infty])
    return {
        "cost":cost[selected_i],
        "complexity":complexity[selected_i],
        "min_lambda":min_lambda,
        "max_lambda":max_lambda,
        }
        
