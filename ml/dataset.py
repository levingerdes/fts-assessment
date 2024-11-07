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

from typing import Sequence, Sized

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class BaseprodProp(Dataset, Sequence, Sized):
    def __init__(
        self,
        csv_file: str,
        transform=None,
        training: bool = True,
        test_split: float = 0.15,
        shuffle: bool = True,
    ) -> None:
        """
        :param csv_file: Path to the CSV file containing the dataset
        :param transform: Optional transform to be applied on a sample
        :param training: If True, return `training` data, otherwise `test` data
        :param test_split: Fraction of data to use for testing
        :param shuffle: If True, shuffle the data
        """
        if test_split == 0 and not training:
            raise ValueError("Test data is requested but test_split is 0")
        if test_split < 0 or test_split > 1:
            raise ValueError("test_split must be between 0 and 1")

        self.data = pd.read_csv(csv_file, skiprows=1)
        self.transform = transform

        if shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)

        if test_split != 0:
            num_samples = len(self.data)
            num_test_samples = int(num_samples * test_split)
            if training:
                self.data = self.data[:-num_test_samples]
            else:
                self.data = self.data[-num_test_samples:]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[Tensor | np.ndarray, Tensor | int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Subtracting one because classes in file are [1..4] instead of [0..3]
        original_label: int = int(
            pd.to_numeric(self.data.iloc[idx, 0].item(), errors="raise")
        )
        label: int = original_label - 1

        # FTS and IMU have 5 metrics (min,max,mean,median,std) per original entry, eg. Fx.
        inputs: np.ndarray = np.array(self.data.iloc[idx, 1:], dtype=float)
        sample: dict[str, np.ndarray | int] = {"inputs": inputs, "label": label}

        if self.transform:
            sample = self.transform(sample)

        res_inputs = sample["inputs"]
        res_label = sample["label"]
        assert isinstance(res_inputs, np.ndarray) or isinstance(res_inputs, Tensor)
        assert isinstance(res_label, int) or isinstance(res_label, Tensor)

        return res_inputs, res_label


class ToTensor(object):
    def __call__(self, sample) -> dict[str, torch.Tensor]:
        return {
            "inputs": torch.from_numpy(sample["inputs"]).float(),
            "label": torch.tensor(sample["label"], dtype=torch.long),
        }
