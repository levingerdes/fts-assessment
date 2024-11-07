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
Analyzes F/T data and determines the usability of the data based on the computed
distance between Torque_Y and Force_X (Ty/Fx).

Note: Assumes the presence of specific columns ("Force_X", "Torque_Y",
"Timestamp") in the input DataFrame.
"""

__author__ = "Levin Gerdes"

import os
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series

from . import baseprod_helpers
from .baseprod_helpers import Fts, rel_time_series
from .plt_helpers import AxesArray


def filter_by_limits(
    traverse_name: str, df: DataFrame, field: str, lim: float
) -> DataFrame:
    """
    Filters the input DataFrame based on the given field and limit.

    Prints the traverse name if any entries in df[field] fall outside [-lim,lim].

    :param traverse_name: The name of the traverse
    :param df: The input DataFrame
    :param field: The field to check
    :param lim: The limit to check against
    :return: The filtered DataFrame
    """
    exceeding_entries: DataFrame = df[(df[field] < -lim) | (df[field] > lim)]
    if not exceeding_entries.empty:
        percent_exceeding = len(exceeding_entries) / len(df) * 100
        print(f"{traverse_name}: {field} exceeding limits: {percent_exceeding:.2f}%")

    df["valid"] = df.get("valid", True) & (-lim <= df[field]) & (df[field] <= lim)

    return df[df["valid"]].copy()


def filter_lever(df: DataFrame, dist_min: float, dist_max: float) -> DataFrame:
    """
    Filter the input DataFrame based on the computed distance between Torque_Y/Force_X.

    :param df: The input DataFrame
    :param lever_min: The minimum lever length [m]
    :param lever_max: The maximum lever length [m]
    :return: The filtered DataFrame
    """

    dist_computed: Series[float] = abs(df["Torque_Y"] / df["Force_X"])
    df["valid"] = (
        df.get("valid", True) & (dist_min < dist_computed) & (dist_computed < dist_max)
    )

    return df[df["valid"]].copy()


class DataStats:
    """
    Store statistics about the data of multiple DataFrames before vs
    after filtering.

    Does not modify the input DataFrames.

    Usage:
    data_stats = DataStats()
    data_stats.update_stats(df_1, df_filtered_1, fts)
    data_stats.update_stats(df_2, df_filtered_2, fts)
    data_stats.print_stats()
    """

    def __init__(self) -> None:
        self.num_total: int = 0
        self.num_dict_total: dict[Fts, float] = {}
        self.num_ok: int = 0
        self.num_dict_ok: dict[Fts, float] = {}
        self.valid_percentage: float = 0.0

    def update_stats(
        self, df: DataFrame, df_filtered: DataFrame, fts: baseprod_helpers.Fts
    ) -> None:
        """
        Update the statistics based on the const input DataFrames.
        """
        len_df: int = len(df)
        len_df_filtered: int = len(df_filtered)

        self.num_total += len_df
        if fts not in self.num_dict_total:
            self.num_dict_total[fts] = 0
        self.num_dict_total[fts] += len_df

        self.num_ok += len_df_filtered
        if fts not in self.num_dict_ok:
            self.num_dict_ok[fts] = 0
        self.num_dict_ok[fts] += len_df_filtered

        self.valid_percentage = round(len_df_filtered / len_df * 100, 2)

    def print_stats(self) -> None:
        print("Total data points:", self.num_total)
        print("'Valid' per sensor [%]:")
        for fts in Fts:
            print(fts, (self.num_dict_ok[fts] / self.num_dict_total[fts]) * 100)
        print(f"Total: {(self.num_ok / self.num_total)*100}")


def main() -> None:
    tolerance: float = 0.01  # Tolerance for the computed distance [m]
    lever_min_no_tol: float = 0.1  # Minimum lever length before tolerance [m]
    lever_max_no_tol: float = 0.175  # Maximum lever length before tolerance [m]
    lever_min: float = lever_min_no_tol - tolerance
    lever_max: float = lever_max_no_tol + tolerance
    # force_limit: float = 205.2  # [N] without considering gear efficiency
    # torque_limit: float = 35.39  # [Nm] without considering gear efficiency
    force_limit: float = 164.16  # [N] considering the gearhead efficiency of 0.8
    torque_limit: float = 28.728  # [Nm] considering the gearhead efficiency of 0.8
    # force_limit: float = 129.276  # [N] considering a total efficiency of 0.63
    # torque_limit: float = 22.6233  # [Nm] considering a total efficiency of 0.63

    show_plots: bool = True

    data_stats: DataStats = DataStats()
    for fts in Fts:
        for traverse in set([item[0] for item in baseprod_helpers.full_traverses]):
            file_path: str = os.path.join(
                baseprod_helpers.data_path, traverse, f"FTS_{fts.name}_CORRECTED.csv"
            )
            df: DataFrame = pd.read_csv(file_path)

            df_filtered: DataFrame
            df_filtered = filter_by_limits(traverse, df, "Force_X", force_limit)
            df_filtered = filter_by_limits(traverse, df, "Torque_Y", torque_limit)
            df_filtered = filter_lever(df=df, dist_min=lever_min, dist_max=lever_max)

            data_stats.update_stats(df, df_filtered, fts)

            fig, _axs = plt.subplots(nrows=2, ncols=1)
            axs: AxesArray = cast(AxesArray, _axs)
            first_timestamp: int = df_filtered["Timestamp"].iloc[0]

            plt.suptitle(
                f"{traverse} {fts}\n{lever_min:.4} ≤ lever ≤ {lever_max:.4} m, {data_stats.valid_percentage}% valid"
            )

            axs[0].scatter(
                rel_time_series(df["Timestamp"][~df["valid"]], first_timestamp),
                df["Force_X"][~df["valid"]],
                c="orange",
                alpha=0.5,
                label="Invalid points",
                s=0.5,
            )
            axs[0].scatter(
                rel_time_series(df["Timestamp"][df["valid"]], first_timestamp),
                df["Force_X"][df["valid"]],
                c="tab:blue",
                alpha=0.5,
                label="Valid points",
                s=0.5,
            )

            axs[1].scatter(
                rel_time_series(df["Timestamp"][~df["valid"]], first_timestamp),
                df["Torque_Y"][~df["valid"]],
                c="orange",
                alpha=0.5,
                label="Invalid points",
                s=0.5,
            )
            axs[1].scatter(
                rel_time_series(df["Timestamp"][df["valid"]], first_timestamp),
                df["Torque_Y"][df["valid"]],
                c="tab:blue",
                alpha=0.5,
                label="Valid points",
                s=0.5,
            )
            # plt.plot(
            #     plt_common.rel_time_series(df_filtered["Timestamp"], first_timestamp),
            #     df_filtered[plot_field],
            #     label="Original Signal",
            # )

            axs[0].set_xlabel("Time [s]")
            axs[1].set_xlabel("Time [s]")
            axs[0].set_ylabel("Force [N]", rotation=90)
            axs[1].set_ylabel("Torque [Nm]", rotation=90)
            axs[0].legend()
            axs[1].legend()

            if show_plots:
                plt.show()

            plt.close()

    data_stats.print_stats()


if __name__ == "__main__":
    main()
