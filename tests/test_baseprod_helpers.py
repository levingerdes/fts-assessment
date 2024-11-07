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

import math
import unittest

from preprocessing.baseprod_helpers import (
    get_angular_velocity,
    get_drv_name,
)


class TestBaseprodHelpers(unittest.TestCase):
    def test_get_drv_name_correct(self) -> None:
        self.assertEqual(get_drv_name("FL"), "link_DRV_LF")
        self.assertEqual(get_drv_name("FR"), "link_DRV_RF")
        self.assertEqual(get_drv_name("CL"), "link_DRV_LM")
        self.assertEqual(get_drv_name("CR"), "link_DRV_RM")
        self.assertEqual(get_drv_name("BL"), "link_DRV_LR")
        self.assertEqual(get_drv_name("BR"), "link_DRV_RR")

    def test_get_drv_name_raises(self) -> None:
        # Too short
        self.assertRaises(ValueError, get_drv_name, "")
        # Too long
        self.assertRaises(ValueError, get_drv_name, "CLF")
        # Wrong first character
        self.assertRaises(ValueError, get_drv_name, "XL")
        # Wrong second character
        self.assertRaises(ValueError, get_drv_name, "FX")

    def test_get_angular_velocity_raises(self) -> None:
        self.assertRaises(ValueError, get_angular_velocity, 0, 0)
        self.assertRaises(ValueError, get_angular_velocity, 0, 1)
        self.assertRaises(ValueError, get_angular_velocity, 0, -1)

    def test_get_angular_velocity(self) -> None:
        self.assertAlmostEqual(0, get_angular_velocity(1e9, 0))
        self.assertAlmostEqual(math.pi, get_angular_velocity(1e9, math.pi))
        self.assertAlmostEqual(math.pi / 2, get_angular_velocity(2e9, math.pi))
        self.assertAlmostEqual(math.pi * 2, get_angular_velocity(0.5e9, math.pi))
        self.assertAlmostEqual(-math.pi, get_angular_velocity(1e9, -math.pi))
