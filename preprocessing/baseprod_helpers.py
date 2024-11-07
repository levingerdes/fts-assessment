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

"""
Traverse names, timestamps, and terrain types.
"""

__author__ = "Levin Gerdes"

import os
from enum import Enum, auto
from typing import TypeAlias

import pandas as pd
from scipy.spatial.transform import Rotation as R


class Fts(Enum):
    """Force torque sensor names"""

    FL = auto()
    FR = auto()
    CL = auto()
    CR = auto()
    BL = auto()
    BR = auto()


class Terrain(Enum):
    LOOSE_SOIL = auto()
    COMPRESSED_SAND = auto()
    PEBBLES = auto()
    ROCK = auto()


data_path: str = os.environ.get("BASEPROD_TRAVERSE_PATH", "/mnt/baseprod/rover_sensors")

TraverseList: TypeAlias = list[tuple[str, str, str, str, str]]
"""[(Traverse Name, Start Timestamp, End Timestamp, Comment, Terrain Type)]"""

# fmt:off
traverse_names: list[str] = ["2023-07-20_18-12-05", "2023-07-21_13-59-14",
"2023-07-21_17-45-42", "2023-07-22_17-31-58", "2023-07-20_19-12-27",
"2023-07-21_14-08-29", "2023-07-22_13-00-57", "2023-07-22_17-38-50",
"2023-07-20_20-01-38", "2023-07-21_14-44-56", "2023-07-22_13-28-55",
"2023-07-23_11-23-18", "2023-07-21_12-38-15", "2023-07-21_14-51-07",
"2023-07-22_14-18-23", "2023-07-23_11-52-09", "2023-07-21_12-58-11",
"2023-07-21_17-07-00", "2023-07-22_16-24-27", "2023-07-23_12-52-39",
"2023-07-21_13-43-00", "2023-07-21_17-34-18", "2023-07-22_17-18-36",
"2023-07-23_13-05-11"]
"""Names of traverses (YYYY-MM-DD_HH-mm-ss)"""

# Lists of labelled traverses with name of traverse, start timestamp, end timestamp, comment, and terrain type
terrain_example_traverses: TraverseList = [
    ("2023-07-21_17-34-18", "1689953714661234688", "1689953790316479488", 'loose soil', Terrain.LOOSE_SOIL.name),
    ("2023-07-20_18-12-05", "1689869949240817152", "1689869979196407296", 'compressed sand', Terrain.COMPRESSED_SAND.name),
    ("2023-07-20_18-12-05", "1689870258000023808", "1689870339792390912", 'right side in compressed riverbed with pebbles', Terrain.PEBBLES.name),
    ("2023-07-21_14-08-29", "1689942140028759808", "1689942204442091264", 'rock uphill. diagonal.',Terrain.ROCK.name),
]

labelled_traverses: TraverseList = [
    ("2023-07-20_18-12-05", "1689869949240817152", "1689869979196407296", 'compressed sand', Terrain.COMPRESSED_SAND.name),
    ("2023-07-20_18-12-05", "1689870061457053952", "1689870147052582912", 'compressed sand', Terrain.COMPRESSED_SAND.name),
    ("2023-07-21_17-34-18", "1689953847491407872", "1689953924046949632", 'compressed sand', Terrain.COMPRESSED_SAND.name),
    ("2023-07-21_17-34-18", "1689953994664815360", "1689954158683838720", 'compressed sand', Terrain.COMPRESSED_SAND.name),
    ("2023-07-20_18-12-05", "1689870258000023808", "1689870339792390912", 'right side in compressed riverbed with pebbles', Terrain.PEBBLES.name),
    ("2023-07-20_18-12-05", "1689870617825619712", "1689870675467329536", 'shaky riverbed, mainly right', Terrain.PEBBLES.name),
    ("2023-07-20_19-12-27", "1689873200913701376", "1689873283070410240", 'loose soil', Terrain.LOOSE_SOIL.name),
    ("2023-07-21_17-34-18", "1689953714661234688", "1689953790316479488", 'loose soil', Terrain.LOOSE_SOIL.name),
    ("2023-07-21_12-38-15", "1689936246126183424", "1689936387894778880", 'rock with turn', Terrain.ROCK.name),
    ("2023-07-21_12-58-11", "1689937212387096320", "1689937281936816384", 'rock straight', Terrain.ROCK.name),
    ("2023-07-21_14-08-29", "1689942975867127040", "1689943123473719808", 'rock straight', Terrain.ROCK.name),
    ("2023-07-21_12-58-11", "1689937414564903424", "1689937542721212288", 'rock uphill', Terrain.ROCK.name),
    ("2023-07-21_12-58-11", "1689937691162692864", "1689937765383283968", 'pebbles downhill', Terrain.PEBBLES.name),
    ("2023-07-21_12-58-11", "1689939418613311488", "1689939507579242752", 'rock uphill. diagonal.',Terrain.ROCK.name),
    ("2023-07-21_14-08-29", "1689942140028759808", "1689942204442091264", 'rock uphill. diagonal.',Terrain.ROCK.name),
]
# fmt:on

# Full, unlabelled traverses in the same format as above
full_traverses: TraverseList = list(
    zip(
        traverse_names,
        ["0"] * len(traverse_names),
        ["9999999999999999999"] * len(traverse_names),
        ["unlabelled"] * len(traverse_names),
        ["UNLABELLED"] * len(traverse_names),
    )
)


def rel_time_sec(timestamp_ns: int, first_timestamp_ns: int) -> float:
    """
    Get relative time to first timestamp in seconds.

    :param timestamp_ns: The current timestamp in nanoseconds
    :param first_timestamp_ns: The first timestamp in nanoseconds
    :return: The relative time in seconds
    """
    return (timestamp_ns - first_timestamp_ns) / 1e9


def rel_time_series(series: pd.Series, t0: int | None = None) -> pd.Series:
    """
    Returns series with relative time in seconds

    :param timestamps: Data Series containing UTC timestamps [ns]
    :param t0: first timestamp [ns], if provided, otherwise using first value of series
    :returns: Data Series containing seconds since first row of ds
    """
    first_timestamp: int = t0 if t0 is not None else series.iloc[0]
    return (series - first_timestamp) / 1e9


def rel_time(df: pd.DataFrame, t0: int | None = None) -> pd.Series:
    """
    Returns series with relative time in seconds

    :param df: DataFrame containing UTC timestamps [ns] in column "Timestamp"
    :param t0: first timestamp [ns], if provided, otherwise using first value of series
    :returns: Data Series containing seconds since first row of df
    """
    return rel_time_series(df["Timestamp"], t0)


def get_drv_name(fts: str) -> str:
    """
    Infer the DRV name from the FTS name.

    The drive names have a different convention than the FT sensors.

    :param fts: The FTS name.
    :return: The DRV name.
    """
    mapping: dict[str, str] = {
        "FL": "LF",
        "FR": "RF",
        "CL": "LM",
        "CR": "RM",
        "BL": "LR",
        "BR": "RR",
    }
    drv_short: str | None = mapping.get(fts, None)

    if drv_short is None:
        raise ValueError("Invalid FTS name")

    return "link_DRV_" + drv_short


def wheel_drive_quat_to_euler_angle(
    qx: float, qy: float, qz: float, qw: float
) -> float:
    """
    Convert the quaternion representation of wheel drive orientation to euler angle.

    :param qx: The x component of the quaternion.
    :param qy: The y component of the quaternion.
    :param qz: The z component of the quaternion.
    :param qw: The w component of the quaternion.
    :return: The euler angle in radians.
    """
    return R.from_quat([qx, qy, qz, qw]).as_euler("ZYX", degrees=False)[1]


def get_angular_velocity(delta_t_ns: float, delta_angle_rad: float) -> float:
    """
    Calculate the angular velocity given the time difference and angle difference.

    :param delta_t_ns: The time difference in nanoseconds.
    :param delta_angle_rad: The angle difference in radians.
    :return: The angular velocity in radians per second.
    """
    if 0 == delta_t_ns:
        raise ValueError("Time difference must not be zero")

    return (delta_angle_rad) / (delta_t_ns / 1e9)
