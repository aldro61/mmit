from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import range

import json
import numpy as np
import pandas as pd

from collections import defaultdict
from functools import partial
from joblib import delayed, Parallel
from mmit import MaxMarginIntervalTree
from mmit.core.solver import compute_optimal_costs
from mmit.metrics import mean_squared_error, zero_one_loss
from mmit.model import TreeExporter
from mmit.model_selection import GridSearchCV
from os import listdir, mkdir, system
from os.path import abspath, basename, exists, join
from shutil import rmtree as rmdir
from sklearn.model_selection import KFold
from time import time


class Dataset(object):
    def __init__(self, path):
        self.path = path
        feature_data = pd.read_csv(join(path, "features.csv"))
        self.X = feature_data.values.astype(np.double)
        self.X.flags.writeable = False
        self.feature_names = feature_data.columns.values
        self.feature_names.flags.writeable = False
        del feature_data
        self.y = pd.read_csv(join(path, "targets.csv")).values.astype(np.double)
        self.y.flags.writeable = False
        self.folds = pd.read_csv(join(path, "folds.csv")).values.reshape(-1, )
        self.folds.flags.writeable = False
        self.name = basename(path)
        # We have to make the numpy arrays unwriteable to do multiprocessing
        # in cross-validation (relies on pickle)

    @property
    def n_examples(self):
        return self.X.shape[0]

    @property
    def n_features(self):
        return self.X.shape[1]

    def __hash__(self):
        return hash((self.X.data, self.y.data, self.feature_names.data, self.folds.data))


def find_datasets(path):
    for d in listdir(path):
        if exists(join(path, d, "features.csv")) and \
                exists(join(path, d, "targets.csv")) and \
                exists(join(path, d, "folds.csv")):
            yield Dataset(abspath(join(path, d)))


def evaluate_on_dataset(d, parameters, metric, result_dir, pruning=True,
                        n_margin_values=10, n_min_samples_split_values=10,
                        n_cpu=-1):
    ds_result_dir = join(result_dir, d.name)
    if not exists(ds_result_dir):
        mkdir(ds_result_dir)

    ds_uid_file = join(ds_result_dir, "dataset.uid")
    # if exists(ds_uid_file) and open(ds_uid_file, "r").next().strip() == str(hash(d)):
    if not exists(join(ds_result_dir, "predictions.csv")):
        start_time = time()
        fold_predictions = np.zeros(d.n_examples)
        fold_train_mse = []
        fold_cv_results = []
        for i, fold in enumerate(np.unique(d.folds)):
            fold_start = time()

            fold_train = d.folds != fold
            X_train = d.X[fold_train]
            y_train = d.y[fold_train]
            X_test = d.X[~fold_train]
            y_test = d.y[~fold_train]

            # Determine the margin grid
            sorted_limits = y_train.flatten()
            sorted_limits = sorted_limits[~np.isinf(sorted_limits)]
            sorted_limits.sort()
            range_max = sorted_limits.max() - sorted_limits.min()
            range_min = np.diff(sorted_limits)
            range_min = range_min[range_min > 0].min()
            parameters = dict(parameters)  # Make a copy
            parameters["margin"] = [0.] + np.logspace(np.log10(range_min), np.log10(range_max), n_margin_values).tolist()

            # Determine the min_samples_split grid
            if not pruning:
                range_min = 2
                range_max = X_train.shape[0]
                parameters["min_samples_split"] = np.logspace(np.log10(range_min), np.log10(range_max), n_min_samples_split_values).astype(np.uint).tolist()
            else:
                parameters["min_samples_split"] = [2]

            cv_protocol = KFold(n_splits=10, shuffle=True, random_state=42)
            cv = GridSearchCV(estimator=MaxMarginIntervalTree(), param_grid=parameters, cv=cv_protocol, n_jobs=n_cpu,
                              scoring=metric, pruning=pruning)
            cv.fit(X_train, y_train, d.feature_names)
            fold_predictions[~fold_train] = cv.predict(X_test)
            fold_cv_results.append({"best": cv.best_params_, "all": cv.cv_results_})
            fold_train_mse.append(mean_squared_error(y_train, cv.predict(X_train)))
            print("........fold {0:d} took {1:.2} seconds".format(i + 1, time() - fold_start))

            # Save the tree
            latex_exporter = TreeExporter("latex")
            open(join(ds_result_dir, "model_fold_{0:d}.tex".format(i + 1)), "w").write(
                latex_exporter(cv.best_estimator_))

        # Save the predictions
        open(join(ds_result_dir, "predictions.csv"), "w")\
            .write("pred.log.penalty\n" + "\n".join(str(x) for x in fold_predictions))

        # Save the cross-validation results for each fold
        json.dump(fold_cv_results, open(join(ds_result_dir, "parameters.json"), "w"))

        # Generate the PDF file for each tree
        # build_cmd = "cd {0!s}; for i in ./model_fold_*.tex; do lualatex $i > /dev/null; rm ./*.aux ./*.log;done".format(ds_result_dir)
        # !$build_cmd

        # Save a hash of the data to avoid re-running
        open(join(ds_uid_file), "w").write(str(hash(d)))


if __name__ == "__main__":
    n_cpu = 50

    run_algos = [
        "mmit.linear.hinge",
        "mmit.linear.hinge.pruning",
        "mmit.squared.hinge",
        "mmit.squared.hinge.pruning"
    ]

    def prep_result_dir(result_dir):
        if not exists(result_dir):
            mkdir(result_dir)

    def mse_metric(estimator, X, y):
        """
        Negative mean squared error, since GridSearchCV maximizes a metric
        """
        return -mean_squared_error(y_pred=estimator.predict(X), y_true=y)

    datasets = list(find_datasets("./data"))

    failed = []
    for method in run_algos:
        print(method)

        # Determine the values of the HPs based on the learning algorithm
        params = {"loss": ["linear_hinge" if method.split(".")[1] == "linear" else "squared_hinge"], "random_state": [np.random.RandomState(42)]}
        if "pruning" in method:
            params.update({"max_depth": [30]})
            pruning = True
        else:
            params.update({"max_depth": [1, 2, 3, 4, 6, 9, 12, 16, 22, 30]})
            pruning = False

        # Prepare the results directory
        result_dir = "./predictions/{0!s}".format(method)
        prep_result_dir(result_dir)

        # Run on all datasets
        for i, d in enumerate(datasets):
            print("....{0:d}/{1:d}: {2!s}".format(i, len(datasets), d.name))
            try:
                evaluate_on_dataset(d, params, mse_metric, result_dir, pruning,
                                    n_margin_values=20,
                                    n_min_samples_split_values=10,
                                    n_cpu=n_cpu)
            except Exception as e:
                print(e.message)
                failed.append((method, d.name))

    print("The following datasets failed to run:")
    for method, d_name in failed:
        print(method, d_name)