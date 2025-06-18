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

import unittest

import numpy as np
import pytest
from torch import Tensor

from ml.dataset import BaseprodProp, ToTensor


class TestDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.data_path: str = "tests/testdata/training_data.csv"
        return super().setUp()

    def test_baseprodprop_init_raises(self) -> None:
        with pytest.raises(ValueError):
            BaseprodProp(self.data_path, training=False, test_split=0)
        with pytest.raises(ValueError):
            BaseprodProp(self.data_path, test_split=-1)
        with pytest.raises(ValueError):
            BaseprodProp(self.data_path, test_split=1.1)

    def test_baseprodprop_init(self) -> None:
        dataset = BaseprodProp(self.data_path, transform=None)
        print(len(dataset))
        for data, label in dataset:
            self.assertTrue(isinstance(data, np.ndarray))
            self.assertTrue(isinstance(label, int))

    def test_baseprodprop_init_tensor(self) -> None:
        dataset = BaseprodProp(self.data_path, transform=ToTensor())
        print(len(dataset))
        for data, label in dataset:
            self.assertTrue(isinstance(data, Tensor))
            self.assertTrue(isinstance(label, Tensor))

    def test_baseprod_shuffle(self) -> None:
        # Shuffling should change order compared to order in CSV
        ds1 = BaseprodProp(self.data_path, shuffle=True, random_state=42)
        ds2 = BaseprodProp(self.data_path, shuffle=False)
        self.assertNotEqual(ds1.data.values.tolist(), ds2.data.values.tolist())

        # Test reproducibility to avoid mixing training and test data.
        ds2 = BaseprodProp(self.data_path, shuffle=True, random_state=42)
        self.assertEqual(ds1.data.values.tolist(), ds2.data.values.tolist())
