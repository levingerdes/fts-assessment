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

import argparse
import pickle
import statistics
import time
import timeit
from typing import Literal, Optional

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

        params = self.get_params()
        print(
            f"Training time: {self.training_time:.4f} sec,  C={params['C']}, gamma={params['gamma']}, kernel={params['kernel']}, degree={params['degree']}"
        )
        return result


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train an SVM model on Baseprod traverse data."
    )
    parser.add_argument(
        "--data_source",
        type=str,
        choices=["all", "imu", "fts", "1fts"],
        default="",
        help="Data source to use for training.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="training_data.csv",
        help="CSV file containing the training data. Ignored if --data_source is specified.",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.25,
        help="Fraction of data to use for testing.",
    )
    parser.add_argument(
        "--skip_shuffle",
        action="store_true",
        help="Skip shuffling the dataset before splitting.",
    )
    parser.add_argument(
        "--scale_features",
        action="store_true",
        help="Scale features using StandardScaler.",
    )
    parser.add_argument(
        "--print_training_time",
        action="store_true",
        help="Print training time for the SVM model.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Path to save the trained SVM model. Defaults to 'svm_{data_source}.pkl'.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and only evaluate a pre-trained model (see checkpoint_path).",
    )
    parser.add_argument(
        "-r",
        "--n_repeats",
        type=int,
        default=100,
        help="Number of repeats for measuring inference time (used for min, mean, std computation of averages over n_iterations).",
    )
    parser.add_argument(
        "-n",
        "--n_iterations",
        type=int,
        default=1000,
        help="Number of iterations for measuring inference time.",
    )
    return parser.parse_args()


def main() -> None:
    np.random.seed(42)

    args = get_args()

    # Parameters
    data_source: Literal["all", "imu", "fts", "1fts"] = args.data_source
    test_split: float = args.test_split
    shuffle: bool = not args.skip_shuffle
    scale_features: bool = args.scale_features
    do_print_training_time: bool = args.print_training_time
    eval_only: bool = args.eval_only
    csv_file: str = args.csv
    checkpoint_path: str = args.checkpoint_path or f"svm_{data_source}.pkl"

    print(f"{data_source=}, {test_split=}, {shuffle=}, {scale_features=}")

    # Load data
    train_data = BaseprodProp(
        csv_file=csv_file,
        transform=None,
        training=True,
        test_split=test_split,
        shuffle=shuffle,
        random_state=42,
    )
    test_data = BaseprodProp(
        csv_file=csv_file,
        transform=None,
        training=False,
        test_split=test_split,
        shuffle=shuffle,
        random_state=42,
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

    if eval_only:
        try:
            with open(checkpoint_path, "rb") as f:
                clf = pickle.load(f)
        except FileNotFoundError:
            print(f"Checkpoint file {checkpoint_path} not found. Exiting.")
            return
    else:
        # Perform grid search for best SVM parameters
        param_grid = {
            "C": [0.1, 1, 10, 100],
            "gamma": [1, 0.1, 0.01, 0.001],
            "kernel": ["linear", "rbf", "poly", "sigmoid"],
            "degree": [2, 3],
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

        # Print best classifier's cross-validation scores
        scores: list[float] = [
            grid.cv_results_[f"split{i}_test_score"][grid.best_index_]
            for i in range(grid.n_splits_)
        ]
        print(
            f"Cross val scores: {scores}. {np.mean(scores):0.4f} accuracy with standard deviation {np.std(scores):0.4f}"
        )

        # Save trained model to disk
        with open(f"svm_{data_source}.pkl", "wb") as f:
            pickle.dump(clf, f)

    # Evaluate
    score_train = clf.score(x_scaled, y)
    start_time = time.time()
    score_test = clf.score(x_test_scaled, y_test)
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time on test set: {inference_time:.4f} seconds")

    res = timeit.repeat(
        "clf.score(x_test_scaled, y_test)",
        repeat=args.n_repeats,
        number=args.n_iterations,
        globals=locals(),
    )
    res = [r / float(args.n_iterations) for r in res]

    minimum = min(res)
    mean = statistics.mean(res)
    std = statistics.stdev(res)

    print(
        f"Inference time on test set: {minimum=:.8f}, {mean=:.8f}, {std=:.8f} seconds (over {args.n_repeats} repeats and averaged over {args.n_iterations} iterations)"
    )

    print(f"Train. Correct Pred.: {score_train * len(x):n} ({score_train * 100:.4f}%)")
    print(
        f"Test Correct Pred.:   {score_test * len(x_test_scaled):n} ({score_test * 100:.4f}%)"
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
