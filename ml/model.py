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

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def init_weights(self, m) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_dim: int, layer_sizes: list[int] = [64, 64]) -> None:
        """
        :param input_dim: Input dimensions, e.g. 6 for one FTS
        :param layer_sizes: List of FC layer sizes, e.g. [64, 64] for two hidden layers with 64 neurons each
        """
        super(Model, self).__init__()
        num_terrains = 4

        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.fc_first = nn.Linear(input_dim, layer_sizes[0])
        self.fcs = nn.ModuleList(
            [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                for i in range(len(layer_sizes) - 1)
            ]
        )
        self.fc_last = nn.Linear(layer_sizes[-1], num_terrains)

        self.apply(self.init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc_first(x)
        x = F.relu(x)
        for fc in self.fcs:
            x = fc(x)
            x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc_last(x)
        output = F.log_softmax(x, dim=-1)

        return output
