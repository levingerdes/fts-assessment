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
import torch.optim as optim
import torcheval
import torcheval.metrics
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader

from .dataset import BaseprodProp, ToTensor
from .model import Model


def validate(
    criterion: nn.Module, model: nn.Module, device: torch.device, testloader: DataLoader
) -> float:
    """
    Compute validation loss

    :param criterion: Loss function
    :param model: Trained model
    :param device: Device to run the model on
    :param testloader: DataLoader for test data
    :return: Validation loss
    """
    running_loss: float = 0.0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            inputs, target = data[0].to(device), data[1].to(device)
            outputs: Tensor = model(inputs)
            running_loss += criterion(outputs, target).item()
    loss: float = running_loss
    return loss


def print_scores_and_confusion_matrix(
    model: nn.Module,
    device: torch.device,
    trainloader: DataLoader,
    testloader: DataLoader,
    checkpoint_path: str,
) -> None:
    """
    Print performance metrics for the model on training and test data.
    Also print confusion matrix and accuracy for each class.

    :param model: Trained model
    :param device: Device to run the model on
    :param trainloader: DataLoader for training data
    :param testloader: DataLoader for test data
    :param checkpoint_path: Path to the checkpoint file
    """
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    predictions: list[Tensor] = []
    labels: list[Tensor] = []
    with torch.no_grad():
        for data in trainloader:
            inputs, targets = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            predicted = torch.max(outputs.data, 1).indices.squeeze()
            total += targets.size(0)
            predictions.append(predicted)
            labels.append(targets)
            correct += (predicted == targets).sum().item()

    print(
        f"Accuracy of the network on the training data points: {100 * correct / total} %"
    )

    model.load_state_dict(torch.load(checkpoint_path))
    correct = 0
    total = 0

    with torch.no_grad():
        all_predictions: list[Tensor] = []
        all_targets: list[Tensor] = []

        for data in testloader:
            inputs, targets = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            predicted = torch.max(outputs.data, 1).indices.squeeze()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            all_predictions.append(predicted)
            all_targets.append(targets)

        confusion_matrix: Tensor = (
            torcheval.metrics.functional.multiclass_confusion_matrix(
                input=torch.cat(all_predictions),
                target=torch.cat(all_targets),
                num_classes=NUM_CLASSES,
            )
        )
        print(confusion_matrix)

    print(f"Accuracy of the network on test data: {100 * correct / total:.4f} %")

    # Count predictions for each class
    classes: list[str] = ["soft soil", "compressed dirt", "pebbles", "rock"]
    correct_pred: dict[str, int] = {classname: 0 for classname in classes}
    total_pred: dict[str, int] = {classname: 0 for classname in classes}
    with torch.no_grad():
        for data in testloader:
            inputs, targets = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            predicted = torch.max(outputs.data, 1).indices.squeeze()
            # collect the correct predictions for each class
            for targets, prediction in zip(targets, predicted):
                if targets == prediction:
                    correct_pred[classes[targets]] += 1
                total_pred[classes[targets]] += 1

    # Print accuracy per class
    for classname, correct_count in correct_pred.items():
        accuracy: float = 100 * float(correct_count) / total_pred[classname]
        print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")


def evaluate_model(model, device, test_loader) -> float:
    """
    Evaluate the model on the test data.

    :param model: Trained model
    :param device: Device to run the model on
    :param test_loader: DataLoader for test data
    :return: Accuracy of the model on the test data
    """
    model.eval()
    running_acc = 0.0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs: Tensor = model(inputs)
            predicted: Tensor = torch.argmax(outputs, dim=1)
            score_accuracy: Tensor = torcheval.metrics.functional.multiclass_accuracy(
                predicted, labels, average="macro", num_classes=NUM_CLASSES
            )
            running_acc += score_accuracy.item()
    return running_acc / len(test_loader)


def train_one_epoch(
    epoch_idx: int,
    criterion: nn.Module,
    model: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    trainloader: DataLoader,
) -> tuple[float, float, float, float]:
    """
    Train the model for one epoch.

    :param epoch_idx: Index of the current epoch
    :param criterion: Loss function
    :param model: Model to train
    :param optimizer: Optimizer to use
    :param device: Device to run the model on
    :param trainloader: DataLoader for training data
    :return: Tuple of loss, f1, accuracy, and precision
    """
    model.train()
    running_loss = 0.0
    running_f1 = 0.0
    running_acc = 0.0
    running_prec = 0.0
    for i, data in enumerate(trainloader):
        inputs, target = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs: Tensor = model(inputs)
        loss: Tensor = criterion(outputs, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)

        score_f1 = torcheval.metrics.functional.multiclass_f1_score(
            predicted, target, num_classes=NUM_CLASSES
        )
        score_precision = torcheval.metrics.functional.multiclass_precision(
            input=predicted,
            target=target,
            num_classes=NUM_CLASSES,
            average="macro",
        )
        score_accuracy = torcheval.metrics.functional.multiclass_accuracy(
            predicted, target, average="macro", num_classes=NUM_CLASSES
        )
        running_loss += loss.item()
        running_f1 += score_f1.item()
        running_acc += score_accuracy.item()
        running_prec += score_precision.item()

        if (i + 1) % 100 == 0:
            print(
                f"[{epoch_idx + 1}, {i + 1:5d}] loss: {loss:.3f} accuracy: {score_accuracy:.3f} precision: {score_precision:.3f} f1: {score_f1:.3f}"
            )
    return (running_loss, running_f1, running_acc, running_prec)


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
    inputs_on_device = [data[0].to(device) for data in testloader]

    start_time = time.time()
    with torch.no_grad():
        for input in inputs_on_device:
            _ = model(input)
    end_time = time.time()
    return end_time - start_time


def main() -> None:
    num_epochs = 50
    batch_size = 32
    batch_size_test = int(0.15 * 1361)
    learning_rate = 0.001
    data_source: str = "all"
    checkpoint_path: str = f"./classify_{data_source}.pt"
    shuffle: bool = True
    test_split: float = 0.25

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

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = Model(input_size)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    trainloader: DataLoader[BaseprodProp] = DataLoader(
        BaseprodProp(
            csv_file=csv_file,
            transform=ToTensor(),
            training=True,
            test_split=test_split,
            shuffle=shuffle,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    testloader: DataLoader[BaseprodProp] = DataLoader(
        BaseprodProp(
            csv_file=csv_file,
            transform=ToTensor(),
            training=False,
            test_split=test_split,
            shuffle=shuffle,
        ),
        batch_size=batch_size_test,
        shuffle=True,
        num_workers=0,
    )
    print(f"Train data: {len(cast(BaseprodProp, trainloader.dataset))} samples")
    print(f"Test data: {len(cast(BaseprodProp, testloader.dataset))} samples")

    loss_values: list[float] = []
    f1_values: list[float] = []
    acc_values: list[float] = []
    prec_values: list[float] = []
    val_loss_values: list[float] = []
    len_trainloader: int = len(trainloader)
    len_testloader: int = len(testloader)

    start_time: float = time.time()
    for epoch in range(num_epochs):
        res: tuple[float, float, float, float] = train_one_epoch(
            epoch, criterion, model, optimizer, device, trainloader
        )
        loss_values.append(res[0] / len_trainloader)
        f1_values.append(res[1] / len_trainloader)
        acc_values.append(res[2] / len_trainloader)
        prec_values.append(res[3] / len_trainloader)
        val_res: float = validate(criterion, model, device, testloader)
        val_loss_values.append(val_res / len_testloader)
    end_time: float = time.time()
    print(f"Training time: {end_time - start_time:.4f} sec")

    print("Finished Training")
    torch.save(model.state_dict(), checkpoint_path)

    inference_time: float = measure_inference_time("cpu", model, testloader)
    print(f"Inference time: {inference_time:.4} sec")

    print_scores_and_confusion_matrix(
        model, device, trainloader, testloader, checkpoint_path
    )

    plt.plot(loss_values, label="train_loss")
    plt.plot(f1_values, label="f1")
    plt.plot(acc_values, label="multiclass_accuracy")
    plt.plot(prec_values, label="multiclass_precision")
    plt.plot(val_loss_values, label="validation_loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    NUM_CLASSES = 4
    main()
