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


class DecisionStump(object):
    def __init__(self, feature_idx, threshold):
        """
        A decision stump rule of the form x[feature_idx] <= threshold

        Parameters:
        -----------
        feature_idx: int
            The index of the feature on which the threshold is applied
        threshold: float
            The threshold below which the rule returns True and over which it returns False.
        """
        self.feature_idx = feature_idx
        self.threshold = threshold

    def classify(self, X):
        return (X[:, self.feature_idx] <= self.threshold).reshape(-1, )

    def __str__(self):
        return "x[{0:d}] <= {1:.4f}".format(self.feature_idx, self.threshold)


class RegressionTreeNode(object):
    def __init__(self, depth, example_idx, rule=None, parent=None, left_child=None,
                 right_child=None, predicted_value=-1, cost_value=-1):
        self.rule = rule
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.example_idx = example_idx
        self.cost_value = cost_value
        self.predicted_value = predicted_value

    @property
    def is_leaf(self):
        return self.left_child is None and self.right_child is None

    @property
    def is_root(self):
        return self.parent is None

    @property
    def leaves(self):
        return _get_tree_leaves(self)

    @property
    def n_examples(self):
        """
        Returns the number of examples in the node

        """
        return len(self.example_idx)

    def predict(self, X):
        """
        Probabilistic class predictions using the current node's rule

        """
        predictions = np.zeros(X.shape[0])

        # Push each example down the tree (an example is a row of X)
        for i, x in enumerate(X):
            x = x.reshape(1, -1)
            current_node = self

            # While we are not in a leaf
            while not current_node.is_leaf:
                # If the rule of the current node returns TRUE, branch left
                if current_node.rule.classify(x):
                    current_node = current_node.left_child
                # Otherwise, branch right
                else:
                    current_node = current_node.right_child

            # A leaf has been reached. Use the leaf's predicted value
            predictions[i] = current_node.predicted_value

        return predictions

    @property
    def rules(self):
        return _get_tree_rules(self)

    def __iter__(self):
        for r in _get_tree_rules(self):
            yield r

    def __len__(self):
        """
        Returns the number of rules in the tree
        """
        return len(_get_tree_rules(self))

    def __str__(self):
        return "{0!s}".format((
            "Node({0!s}, {1!s}, {2!s})".format(self.rule, self.left_child, self.right_child) if not (self.left_child is None)
            else "Leaf({0:.4f})".format(self.predicted_value)))


class TreeExporter(object):
    def __init__(self, out_fmt="latex"):
        self.out_fmt = out_fmt

    def __call__(self, model):
        if self.out_fmt == "latex":
            _latex_export(model)
        else:
            raise ValueError("Unknown export format specified")


def _latex_export(model):
    def _rec_export(node, depth):
        if not node.is_leaf:
            indent = "\t" * depth
            return '\n{0!s} {1!s}[as={2!s}, nonterminal] -> {{{3!s}, {4!s}}}'.format(indent,
                    str(hash((node.rule, node.parent))),
                    str(node.rule).replace("<=", "$\leq$").replace("[", "(").replace("]", ")"),
                    _rec_export(node.left_child, depth + 1),
                    _rec_export(node.right_child, depth + 1))
        else:
            return "{0!s}[as=Ex: ${1:d}$\\\\Cost: ${2:.3f}$\\\\Pred: ${3:.3f}$, terminal]".format(str(hash(node)),
                                                                   node.n_examples,
                                                                   node.cost_value,
                                                                   node.predicted_value)
    exported = \
"""
% !TeX program = lualatex
\\documentclass[tikz,border=5]{{standalone}}
\\definecolor{{nonterminal}}{{RGB}}{{230,230,230}}
\\definecolor{{terminal}}{{RGB}}{{255,51,76}}
\\usetikzlibrary{{shapes.misc, positioning}}
\\usetikzlibrary{{graphs,graphdrawing,arrows.meta}}
\\usegdlibrary{{trees}}
\\begin{{document}}
\\begin{{tikzpicture}}[>=Stealth,
                     nonterminal/.style={{draw, fill=nonterminal, rounded rectangle, align=center}},
                     terminal/.style={{draw, fill=terminal, rectangle, align=left}}]
\\graph[binary tree layout]{{
{0!s}
}};
\\end{{tikzpicture}}
\\end{{document}}
""".format(_rec_export(model.tree_, 0))
    print exported


def _get_tree_leaves(root):
    def _get_leaves(node):
        leaves = []
        if not node.is_leaf:
            leaves += _get_leaves(node.left_child)
            leaves += _get_leaves(node.right_child)
        else:
            leaves.append(node)
        return leaves
    return _get_leaves(root)


def _get_tree_rules(root):
    def _get_rules(node):
        rules = []
        if node.rule is not None:
            rules.append(node.rule)
            rules += _get_rules(node.left_child)
            rules += _get_rules(node.right_child)
        return rules
    return _get_rules(root)
