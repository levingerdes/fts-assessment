# MIT License
#
# Copyright (c) 2024 Space Robotics Lab at UMA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

__author__ = "Levin Gerdes"

import time
from typing import Optional

import numpy as np
import pandas as pd  # only for typing
import scipy.sparse as sp  # only for typing
from numpy.typing import ArrayLike  # only for typing
from sklearn import svm  # type: ignore
from sklearn.metrics import confusion_matrix  # type: ignore
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from .dataset import BaseprodProp

MatrixLike = np.ndarray | pd.DataFrame | sp.spmatrix


class TimedSVC(svm.SVC):
    """SVC with timing for training and inference"""

    def fit(
        self, X: MatrixLike, y: ArrayLike, sample_weight: Optional[ArrayLike] = None
    ) -> "TimedSVC":
        start_time = time.time()
        result = super().fit(X, y, sample_weight)
        end_time = time.time()
        self.training_time = end_time - start_time

        print(
            f"Training time: {self.training_time:.4f} sec,  C={self.C}, gamma={self.gamma}, kernel={self.kernel}"  # type: ignore # PyLance doesn't understand the inheritance from svm.SVC
        )
        return result


def main() -> None:
    # Set seed for reproducibility
    np.random.seed(42)

    # Parameters
    data_source: str = "all"
    test_split: float = 0.25
    shuffle: bool = True
    scale_features: bool = False
    do_print_training_time: bool = True

    match data_source:
        case "imu":
            csv_file = "training_data_imu.csv"
        case "fts":
            csv_file = "training_data_ft.csv"
        case _:
            csv_file = "training_data.csv"

    print(f"{data_source=}, {test_split=}, {shuffle=}, {scale_features=}")

    # Load data
    train_data = BaseprodProp(
        csv_file=csv_file,
        transform=None,
        training=True,
        test_split=test_split,
        shuffle=shuffle,
    )
    test_data = BaseprodProp(
        csv_file=csv_file,
        transform=None,
        training=False,
        test_split=test_split,
        shuffle=shuffle,
    )
    print(f"Train data: {len(train_data)} samples")
    print(f"Test data: {len(test_data)} samples")

    x = train_data.data.iloc[:, 1:]
    y = train_data.data.iloc[:, 0]
    x_test = test_data.data.iloc[:, 1:]
    y_test = test_data.data.iloc[:, 0]

    # Standardize the features
    if scale_features:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        x_test_scaled = scaler.transform(x_test)
    else:
        x_scaled = x
        x_test_scaled = x_test

    # Perform grid search for best SVM parameters
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": [1, 0.1, 0.01, 0.001],
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
    }
    classifier = (
        TimedSVC(decision_function_shape="ovr")
        if do_print_training_time
        else svm.SVC(decision_function_shape="ovr")
    )
    grid = GridSearchCV(classifier, param_grid, refit=True, verbose=1, cv=5)
    start_time = time.time()
    grid.fit(x_scaled, y)
    end_time = time.time()
    print(f"Grid search took {end_time - start_time:.4f} seconds")
    print("Best parameters found: ", grid.best_params_)
    clf = grid.best_estimator_

    # Evaluate
    score_train = clf.score(x_scaled, y)
    start_time = time.time()
    score_test = clf.score(x_test_scaled, y_test)
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time on test set: {inference_time:.4f} seconds")
    print(f"Train. Correct Pred.: {score_train*len(x):n} ({score_train*100:.2f}%)")
    print(
        f"Test Correct Pred.:   {score_test*len(x_test_scaled):n} ({score_test*100:.2f}%)"
    )

    confusion = confusion_matrix(y_test, clf.predict(x_test_scaled))
    print("\nConfusion Matrix:")
    print(confusion)

    # Calculate confusion matrix as percentages
    confusion_pct = confusion.astype("float") / confusion.sum(axis=1)[:, np.newaxis]
    print("\nConfusion Matrix (Percentages):")
    print(confusion_pct * 100)


if __name__ == "__main__":
    main()
