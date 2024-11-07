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
    device: str, model: nn.Module, testloader: DataLoader
) -> float:
    """
    Run and time inference on the test data.

    :param device: Device to run the model on
    :param model: Trained model
    :param testloader: DataLoader for test data
    :return: Inference time [s]
    """
    model.eval()

    # Move to device before starting the timer
    model.to(device)
    inputs_on_device: list[Tensor] = [data[0].to(device) for data in testloader]

    start_time: float = time.time()
    with torch.no_grad():
        for input in inputs_on_device:
            _ = model(input)
    end_time: float = time.time()
    return end_time - start_time


def main() -> None:
    data_source: str = "all"
    checkpoint_path: str = f"./classify_{data_source}.pt"
    batch_size_test = 32

    match data_source:
        case "imu":
            csv_file = "training_data_imu.csv"
            input_size = 3 * 5
        case "fts":
            csv_file = "training_data_ft.csv"
            input_size = 6 * 7 * 5
        case _:
            csv_file = "training_data.csv"
            input_size = 6 * 7 * 5 + 3 * 5

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = Model(input_size)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    # optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)
    testloader: DataLoader[BaseprodProp] = DataLoader(
        dataset=BaseprodProp(
            csv_file=csv_file,
            transform=ToTensor(),
            training=False,
            test_split=1.0,
            shuffle=False,
        ),
        batch_size=batch_size_test,
        shuffle=True,
        num_workers=0,
    )
    print(f"Test data: {len(cast(BaseprodProp, testloader.dataset))} samples")

    inference_time: float = measure_inference_time("cpu", model, testloader)
    print(f"Inference time: {inference_time:.4} sec")

    print_scores_and_confusion_matrix(model, device, testloader)


if __name__ == "__main__":
    main()
