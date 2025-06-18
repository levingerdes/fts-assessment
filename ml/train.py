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
import time
from typing import Literal, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torcheval
import torcheval.metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from .dataset import BaseprodProp, ToTensor
from .model import Model


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a model on the Baseprod dataset."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to train the model.",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.25,
        help="Fraction of data to use for testing.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        choices=["all", "imu", "fts", "1fts"],
        default="all",
        help="Data source to use for training.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="CSV file containing the training data.",
    )
    parser.add_argument(
        "--skip_shuffle",
        action="store_true",
        help="Whether to skip shuffling the dataset before training.",
    )
    parser.add_argument(
        "--skip_cross_val",
        action="store_true",
        help="Whether to skip cross-validation.",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of folds for (stratified) cross-validation.",
    )

    return parser.parse_args()


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
    num_classes: int = 4,
) -> None:
    """
    Print performance metrics for the model on training and test data.
    Also print confusion matrix and accuracy for each class.

    :param model: Trained model
    :param device: Device to run the model on
    :param trainloader: DataLoader for training data
    :param testloader: DataLoader for test data
    :param checkpoint_path: Path to the checkpoint file
    :param num_classes: Number of terrain classes in the dataset
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")

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
                input=torch.cat(all_predictions).to(
                    "cpu" if device.type == "mps" else device
                ),
                target=torch.cat(all_targets).to(
                    "cpu" if device.type == "mps" else device
                ),
                num_classes=num_classes,
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


def evaluate_model(
    model: Model,
    device: torch.device,
    test_loader: DataLoader,
    num_classes: int = 4,
) -> float:
    """
    Evaluate the model on the test data.

    :param model: Trained model
    :param device: Device to run the model on
    :param test_loader: DataLoader for test data
    :param num_classes: Number of terrain classes in the dataset
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
                predicted, labels, average="macro", num_classes=num_classes
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
    metrics: bool,
    num_classes: int = 4,
) -> tuple[float, float, float, float]:
    """
    Train the model for one epoch.

    If `metrics` is False, returns tuple of loss followed by 0.0 for rest.

    :param epoch_idx: Index of the current epoch
    :param criterion: Loss function
    :param model: Model to train
    :param optimizer: Optimizer to use
    :param device: Device to run the model on
    :param trainloader: DataLoader for training data
    :param metrics: Whether to compute metrics during training
    :param num_classes: Number of terrain classes in the dataset
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

        running_loss += loss.item()
        if metrics:
            _, predicted = torch.max(outputs.data, 1)

            score_f1 = torcheval.metrics.functional.multiclass_f1_score(
                predicted, target, num_classes=num_classes
            )
            score_precision = torcheval.metrics.functional.multiclass_precision(
                input=predicted,
                target=target,
                num_classes=num_classes,
                average="macro",
            )
            score_accuracy = torcheval.metrics.functional.multiclass_accuracy(
                predicted, target, average="macro", num_classes=num_classes
            )
            running_f1 += score_f1.item()
            running_acc += score_accuracy.item()
            running_prec += score_precision.item()

            if (i + 1) % 100 == 0:
                print(
                    f"[{epoch_idx + 1}, {i + 1:5d}] loss: {loss:.3f} accuracy: {score_accuracy:.3f} precision: {score_precision:.3f} f1: {score_f1:.3f}"
                )

    if metrics:
        res = (running_loss, running_f1, running_acc, running_prec)
    else:
        res = (running_loss, 0.0, 0.0, 0.0) if not metrics else res

    return res


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


def cross_validate(
    device: torch.device,
    criterion: nn.Module,
    input_size: int,
    data_source: Literal["imu", "fts", "1fts", "all"],
    dataset: Dataset[BaseprodProp],
    num_folds: int,
    epochs: int,
    batch_size: int,
    random_state: int,
    num_classes: int = 4,
) -> tuple[list[float], str]:
    """
    Perform stratified k-fold cross-validation on the dataset.

    :param device: Device to run the model on
    :param criterion: Loss function
    :param input_size: Input size of the model
    :param data_source: Data source to use for training
    :param dataset: Dataset to use for training
    :param num_folds: Number of folds for cross-validation
    :param epochs: Number of epochs to train each model
    :param batch_size: Batch size for training
    :param random_state: Random state for reproducibility
    :param num_classes: Number of terrain classes in the dataset

    :return: Tuple of validation accuracies for each fold and the path to the best fold's checkpoint
    """
    val_losses = [0.0] * num_folds
    val_accuracies = [0.0] * num_folds
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)

    # Dataset is list of tuples (input vector, target)
    X = [data[0] for data in iter(dataset)]
    y = [data[1] for data in iter(dataset)]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):  # type: ignore
        print(f"Fold {fold + 1}/{num_folds}")

        # Dataloaders for this fold
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Init
        model = Model(input_dim=input_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        val_loss: float = 0.0
        running_accuracy: float = 0.0
        for epoch in range(epochs):
            model.train()
            train_one_epoch(
                epoch, criterion, model, optimizer, device, train_loader, metrics=False
            )

            model.eval()
            val_loss = 0.0
            running_accuracy = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
                    accuracy = torcheval.metrics.functional.multiclass_accuracy(
                        outputs, targets, average="macro", num_classes=num_classes
                    ).item()
                    running_accuracy += accuracy

        val_losses[fold] = val_loss / len(val_loader)
        val_accuracies[fold] = running_accuracy / len(val_loader)

        torch.save(model.state_dict(), f"model_{data_source}_fold_{fold}.pth")
    print(
        f"Cross-validation results: {val_accuracies}, avg: {np.mean(val_accuracies)}, std: {np.std(val_accuracies)}"
    )

    best_fold = np.argmax(val_accuracies)
    best_checkpoint_path: str = f"model_{data_source}_fold_{best_fold}.pth"

    return val_accuracies, best_checkpoint_path


def main() -> None:
    args = get_args()
    print(args)
    epochs: int = args.epochs
    test_split: float = args.test_split
    batch_size: int = args.batch_size
    learning_rate: float = args.learning_rate
    data_source: Literal["all", "imu", "fts", "1fts"] = args.data_source
    checkpoint_path: str = f"./classify_{data_source}.pt"
    shuffle: bool = not args.skip_shuffle
    do_cross_val: bool = not args.skip_cross_val
    k_folds: int = args.k_folds

    match data_source:
        case "imu":
            csv_file = "training_data_imu.csv"
            input_size = 3 * 5
        case "fts":
            csv_file = "training_data_ft.csv"
            input_size = 6 * 7 * 5
        case "1fts":
            csv_file = "training_data_1fts.csv"
            input_size = 1 * 7 * 5
        case _:
            csv_file = "training_data.csv"
            input_size = 6 * 7 * 5 + 3 * 5
    if args.csv != "":
        csv_file = args.csv

    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(device)

    criterion = nn.CrossEntropyLoss()
    trainloader: DataLoader[BaseprodProp] = DataLoader(
        BaseprodProp(
            csv_file=csv_file,
            transform=ToTensor(),
            training=True,
            test_split=test_split,
            shuffle=shuffle,
            random_state=42,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    testloader: DataLoader[BaseprodProp] = DataLoader(
        ds := BaseprodProp(
            csv_file=csv_file,
            transform=ToTensor(),
            training=False,
            test_split=test_split,
            shuffle=shuffle,
            random_state=42,
        ),
        batch_size=len(ds),
        shuffle=True,
        num_workers=0,
    )
    print(f"Train data: {len(cast(BaseprodProp, trainloader.dataset))} samples")
    print(f"Test data: {len(cast(BaseprodProp, testloader.dataset))} samples")

    start_time: float = time.time()

    canditate_layer_sizes: list[list[int]] = [
        [32],
        [32, 16],
        [64, 32],
        [64, 64],
        [32, 16, 8],
        [64, 32, 8],
        [64, 8, 8, 8],
    ]
    best_hyper_params = canditate_layer_sizes[0]

    if do_cross_val:
        print("Looking for best layer sizes with cross-validation")
        best_accuracy = -1.0
        for layer_sizes in canditate_layer_sizes:
            print(f"Starting cross-validation for {layer_sizes=}")
            cross_val_accuracies, best_cross_val_ckpt = cross_validate(
                device=device,
                criterion=criterion,
                data_source=data_source,
                input_size=input_size,
                dataset=trainloader.dataset,
                num_folds=k_folds,
                epochs=epochs,
                batch_size=batch_size,
                random_state=42,
            )
            accuracy = np.mean(cross_val_accuracies)
            print(f"Finished cross-validation: {accuracy=}")
            if accuracy > best_accuracy or best_accuracy < 0:
                best_accuracy = accuracy.item()
                best_hyper_params = layer_sizes
                print(
                    f"New best hyper-parameters: {best_hyper_params} with {best_accuracy=}"
                )
        print(
            f"Best hyper-parameters: {best_hyper_params}. \nHyperparameter search took {time.time() - start_time:.4f} sec with {best_accuracy=:.4f}"
        )

    # Reset seeds to be independent of whether or not we do cross-validation
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Setting up model with best hyper-parameters: {best_hyper_params}")
    layer_sizes = best_hyper_params
    model = Model(input_dim=input_size, layer_sizes=layer_sizes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_values: list[float] = []
    f1_values: list[float] = []
    acc_values: list[float] = []
    prec_values: list[float] = []
    val_loss_values: list[float] = []
    len_trainloader: int = len(trainloader)
    len_testloader: int = len(testloader)

    start_time_final_training: float = time.time()

    print(f"Starting model training for {epochs} epochs")
    for epoch in range(epochs):
        res: tuple[float, float, float, float] = train_one_epoch(
            epoch, criterion, model, optimizer, device, trainloader, metrics=True
        )
        loss_values.append(res[0] / len_trainloader)
        f1_values.append(res[1] / len_trainloader)
        acc_values.append(res[2] / len_trainloader)
        prec_values.append(res[3] / len_trainloader)
        val_res: float = validate(criterion, model, device, testloader)
        val_loss_values.append(val_res / len_testloader)
    end_time: float = time.time()
    print(
        f"Finished final training. Time: {end_time - start_time_final_training:.4f} sec\nFinal train loss: {loss_values[-1]:.4f}\nFinal val loss: {val_loss_values[-1]:.4f}"
    )

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
    main()
