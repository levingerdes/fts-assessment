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

from typing import Sequence, TypeAlias

import matplotlib.pyplot as plt
import numpy as np

AxesArray: TypeAlias = "np.ndarray[Sequence[Sequence[plt.Axes]], np.dtype[np.object_]]"
"""
Type hint for a 2D array of matplotlib axes.

E.g.:
```python
from typing import cast
fig, _axs = plt.subplots(2, 2)
axs: AxesArray = cast(AxesArray, _axs)
```

Taken from [here](https://stackoverflow.com/questions/72649220/precise-type-annotating-array-numpy-ndarray-of-matplotlib-axes-from-plt-subplo)
"""


def copy_axis_content(source_ax: plt.Axes, target_ax: plt.Axes) -> None:
    """
    Copy content from source figure axis to target figure axis

    :param source_ax: The source axis.
    :param target_ax: The target axis.
    """
    for line in source_ax.get_lines():
        target_ax.plot(
            line.get_xdata(),
            line.get_ydata(),
            label=line.get_label(),
            color=line.get_color(),
        )
    target_ax.set_title(source_ax.get_title())
    target_ax.set_xlabel(source_ax.get_xlabel())
    target_ax.set_ylabel(source_ax.get_ylabel())

    x_min, x_max = source_ax.get_xlim()
    target_ax.set_xlim(xmin=x_min, xmax=x_max)

    y_min, y_max = source_ax.get_ylim()
    target_ax.set_ylim(ymin=y_min, ymax=y_max)
