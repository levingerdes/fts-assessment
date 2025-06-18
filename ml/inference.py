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
import os
import statistics
import timeit
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torcheval
import torcheval.metrics
from torch import Tensor
from torch.utils.data import DataLoader

from .dataset import BaseprodProp, ToTensor
from .model import Model


def get_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Inference timing script for terrain classification"
    )
    parser.add_argument(
        "-d",
        "--data_source",
        type=str,
        choices=["imu", "fts", "1fts", "all"],
        default="all",
        help="Data source to use for inference (default: all)",
    )
    parser.add_argument(
        "-p",
        "--checkpoint_path",
        type=str,
        default="./classify_all.pt",
        help="Path to the model checkpoint (default: ./classify_all.pt)",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for test data (default: 32)",
    )
    parser.add_argument(
        "--layer_sizes",
        type=int,
        nargs="+",
        default=[64, 64],
        help="Sizes of the hidden layers in the model (default: [64, 64])",
    )
    parser.add_argument(
        "-r",
        "--n_repeats",
        type=int,
        default=100,
        help="Number of repeats of n_iterations for measuring inference min, mean, std time (default: 100)",
    )
    parser.add_argument(
        "-n",
        "--n_iterations",
        type=int,
        default=1000,
        help="Number of iterations for measuring inference time (default: 1000)",
    )
    parser.add_argument(
        "-c",
        "--csv_file",
        type=str,
        default="data/training_data.csv",
        help="Path to the CSV file containing the dataset (default: data/training_data.csv)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "gpu", "mps"],
        default="cpu",
        help="Device to run the model on (default: cpu). 'auto' will use CUDA if available, otherwise MPS or CPU.",
    )
    return parser.parse_args()


def print_scores_and_confusion_matrix(
    model: nn.Module,
    device: torch.device,
    testloader: DataLoader,
    num_classes: int = 4,
) -> None:
    """
    Print performance metrics for the model on training and test data.
    Also print confusion matrix and accuracy for each class.

    :param model: Trained model
    :param device: Device to run the model on
    :param trainloader: DataLoader for training data
    :param testloader: DataLoader for test data
    :param num_classes: Number of terrain classes
    """
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        all_predictions: list[Tensor] = []
        all_targets: list[Tensor] = []

        for data in testloader:
            inputs, targets = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            predicted: Tensor = torch.max(outputs.data, 1).indices.squeeze()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            all_predictions.append(predicted)
            all_targets.append(targets)

        confusion_matrix: Tensor = (
            torcheval.metrics.functional.multiclass_confusion_matrix(
                input=torch.cat(all_predictions),
                target=torch.cat(all_targets),
                num_classes=num_classes,
            )
        )
        print(confusion_matrix)

    print(f"Accuracy of the network on test data: {100 * correct / total:.4f} %")

    # prepare to count predictions for each class
    classes: list[str] = ["soft soil", "compressed dirt", "pebbles", "rock"]
    correct_pred: dict[str, int] = {classname: 0 for classname in classes}
    total_pred: dict[str, int] = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            inputs, targets = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            predicted = torch.max(outputs.data, 1).indices.squeeze()

            for targets, prediction in zip(targets, predicted):
                if targets == prediction:
                    correct_pred[classes[targets]] += 1
                total_pred[classes[targets]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy: float = 100 * float(correct_count) / total_pred[classname]
        print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")


def evaluate_model(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    num_classes: int = 4,
) -> float:
    """
    Evaluate the model on the test data.

    :param model: Trained model
    :param device: Device to run the model on
    :param test_loader: DataLoader for test data
    :param num_classes: Number of terrain classes
    :return: Accuracy of the model on the test data
    """
    model.eval()
    running_acc = 0.0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            predicted: Tensor = torch.argmax(outputs, dim=1)
            score_accuracy: Tensor = torcheval.metrics.functional.multiclass_accuracy(
                predicted, labels, average="macro", num_classes=num_classes
            )
            running_acc += score_accuracy.item()
    return running_acc / len(test_loader)


def measure_inference_time(
    device: torch.device, model: nn.Module, testloader: DataLoader, r: int, n: int
) -> tuple[float, float, float]:
    """
    Runs inference on the test data `r` times `n` times and returns the minimum, mean, and standard deviation wrt. `r`.

    I.e., the returned `std` is the std of `r` values, each representing the average time of `n` inferences.

    :param device: Device to run the model on
    :param model: Trained model
    :param testloader: DataLoader for test data
    :param r: Number of repeats
    :param n: Number of iterations to run for timing
    :return: Min, mean, std of `r` repeats of `n` inferences [s]
    """
    model.eval()

    # Move to device before starting the timer
    model.to(device)
    inputs_on_device: list[Tensor] = [data[0].to(device) for data in testloader]

    res = timeit.repeat(
        "[model(input) for input in inputs_on_device]",
        repeat=r,
        number=n,
        globals=locals(),
    )

    res = [r / float(n) for r in res]  # average over n iterations

    minimum = min(res)
    mean = statistics.mean(res)
    std = statistics.stdev(res)

    return minimum, mean, std


def main() -> None:
    args = get_args()
    data_source: str = args.data_source
    checkpoint_path: str = args.checkpoint_path
    batch_size_test = args.batch_size
    layer_sizes: list[int] = args.layer_sizes
    n_repeats: int = args.n_repeats
    n_iterations: int = args.n_iterations
    csv_file: str = args.csv_file

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    match data_source:
        case "imu":
            input_size = 1 * 3 * 5
        case "fts":
            input_size = 6 * 7 * 5
        case "1fts":
            input_size = 1 * 7 * 5
        case _:
            input_size = 6 * 7 * 5 + 1 * 3 * 5

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    match args.device:
        case "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        case "cpu" | "mps":
            device = torch.device(args.device)
        case "cuda" | "gpu":
            device = torch.device("cuda:0")
        case _:
            raise ValueError(f"Unknown device: {args.device}")

    print(device)

    model = Model(input_size, layer_sizes)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    testloader: DataLoader[BaseprodProp] = DataLoader(
        dataset=BaseprodProp(
            csv_file=csv_file,
            transform=ToTensor(),
            training=False,
            test_split=0.25,
            random_state=42,
            shuffle=True,
        ),
        batch_size=batch_size_test,
        shuffle=True,
        num_workers=0,
    )
    print(f"Test data: {len(cast(BaseprodProp, testloader.dataset))} samples")

    minimum, mean, std = measure_inference_time(
        device, model, testloader, r=n_repeats, n=n_iterations
    )
    print(f"Inference time: {minimum=:.8f}, {mean=:.8f}, {std=:.8f} sec")

    print_scores_and_confusion_matrix(model, device, testloader)


if __name__ == "__main__":
    main()
