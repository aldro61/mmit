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
from sklearn.utils.validation import check_array, check_consistent_length


class BetweenDict(dict):
    """
    The BetweenDict
    By: Joshua Kugler
    Source: http://joshuakugler.com/archives/30-BetweenDict,-a-Python-dict-for-value-ranges.html
    """
    def __init__(self, d=None):
        if d is not None:
            for k,v in d.items():
                self[k] = v

    def __getitem__(self, key):
        for k, v in self.items():
            if k[0] <= key < k[1]:
                return v
        raise KeyError("Key '%s' is not between any values in the BetweenDict" % key)

    def __setitem__(self, key, value):
        try:
            if len(key) == 2:
                if key[0] < key[1]:
                    dict.__setitem__(self, (key[0], key[1]), value)
                else:
                    raise RuntimeError('First element of a BetweenDict key '
                                       'must be strictly less than the '
                                       'second element. Got [%.6f, %.6f]' % (key[0], key[1]))
            else:
                raise ValueError('Key of a BetweenDict must be an iterable '
                                 'with length two')
        except TypeError:
            raise TypeError('Key of a BetweenDict must be an iterable '
                             'with length two')

    def __contains__(self, key):
        try:
            return bool(self[key]) or True
        except KeyError:
            return False


def check_X_y(X, y):
    X = check_array(X, force_all_finite=True)
    y = check_array(y, 'csr', force_all_finite=False, ensure_2d=True, dtype=None)
    check_consistent_length(X, y)
    if y.shape[1] != 2:
        raise ValueError("y must contain lower and upper bounds for each interval.")
    return X, y