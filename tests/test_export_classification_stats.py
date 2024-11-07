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

from preprocessing.baseprod_helpers import TraverseList
from preprocessing.export_classification_stats import process_baseprod


class TestExportClassificationStats(unittest.TestCase):
    def test_process_baseprod(self) -> None:
        traverses: TraverseList = [("1234", "1234", "1235", "Comment", "0")]

        self.assertRaises(FileNotFoundError, process_baseprod, "", traverses)
        self.assertRaises(ValueError, process_baseprod, "/", [])
        self.assertRaises(
            ValueError, process_baseprod, "/", traverses, rolling_window_ns=0
        )
        self.assertRaises(
            ValueError,
            process_baseprod,
            "/",
            traverses,
            cut_to_lever=True,
            lever_min=0,
            lever_max=0,
        )
        self.assertRaises(
            ValueError,
            process_baseprod,
            "/",
            traverses,
            cut_to_lever=True,
            lever_min=3,
            lever_max=2,
        )

        # self.assertEqual(
        #     process_baseprod(
        #         "/", traverses, cut_to_lever=False, lever_min=0, lever_max=0
        #     ),
        #     None,
        # )
