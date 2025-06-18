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
Visualize forces, torques, and IMU orientation to see whether we can find clear
terrain signatures in the F/T data.
"""

__author__ = "Levin Gerdes"

import os
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.core.window.rolling import Rolling

from . import baseprod_helpers
from .baseprod_helpers import (
    get_angular_velocity,
    get_drv_name,
    rel_time,
    wheel_drive_quat_to_euler_angle,
)
from .baseprod_helpers import labelled_traverses as traverses
from .plt_helpers import AxesArray, copy_axis_content


def plot_fts_data(
    fts: str,
    plot_stats: bool,
    rolling_window_sec: float,
    marker_size: float,
    marker_style: str,
    linewidth: float,
    axs: AxesArray,
    df: DataFrame,
) -> None:
    """
    Plot the FTS data on axes axs[1:2, 0:2].

    :param fts: The FTS name.
    :param plot_stats: Whether to plot the mean and standard deviation.
    :param rolling_window_sec: The rolling window size in seconds.
    :param marker_size: The size of the markers.
    :param marker_style: The style of the markers.
    :param linewidth: The width of the lines.
    :param axs: The axes to plot on.
    :param df: The dataframe containing FTS data.
    """
    fields: list[str] = [
        "Force_X",
        "Force_Y",
        "Force_Z",
        "Torque_X",
        "Torque_Y",
        "Torque_Z",
    ]
    for i, field in enumerate(fields):
        ax_row: int = 1 + i // 3
        ax_col: int = i % 3
        ax: plt.Axes = axs[ax_row, ax_col]
        ax.plot(
            rel_time(df),
            df[field],
            marker=marker_style,
            markersize=marker_size,
            label=f"{fts} {field}",
            linewidth=linewidth,
        )
        axs[ax_row, ax_col].set_title(f"FTS {fts} {field.split('_')[1]} Axis")
        if field.find("Force") != -1:
            axs[ax_row, ax_col].set_ylabel("Force [N]", rotation=90)
        else:
            axs[ax_row, ax_col].set_ylabel("Torque [Nm]", rotation=90)

        if plot_stats:
            # Create rolling window. Needs a Dataframe with Datetime index
            df["Datetime"] = pd.to_datetime(df["Timestamp"]).dt.tz_localize(None)
            df = df.set_index("Datetime")
            rolling: Rolling[Series] = df[field].rolling(
                pd.to_timedelta(rolling_window_sec, unit="seconds"),
                min_periods=1,
            )
            ax.plot(
                rel_time(df),
                rolling.mean(),
                marker=marker_style,
                markersize=marker_size,
                label=f"{fts} {field}",
                linewidth=linewidth,
            )
            ax.plot(
                rel_time(df),
                rolling.std(),
                marker=marker_style,
                markersize=marker_size,
                label=f"{fts} {field}",
                linewidth=linewidth,
            )


def plot_wheel_data(
    fts: str,
    marker_size: float,
    marker_style: str,
    linewidth: float,
    axs: AxesArray,
    t_start: int,
    t_end: int,
    traverse_dir_abs: str,
) -> None:
    """
    Plot the wheel angular velocity and drive angle

    :param fts: The FTS name.
    :param marker_size: The size of the markers.
    :param marker_style: The style of the markers.
    :param linewidth: The width of the lines.
    :param axs: The axes to plot on.
    :param t_start: The start time.
    :param t_end: The end time.
    :param traverse_dir_abs: The absolute path to the traverse directory.
    """
    file_path: str = os.path.join(traverse_dir_abs, "TF.csv")
    df_tf: DataFrame = pd.read_csv(file_path)
    df_tf = df_tf[
        (t_start <= df_tf["Timestamp"]) & (df_tf["Timestamp"] <= t_end)
    ].dropna()
    df_tf = df_tf[df_tf["Child_Frame_ID"] == get_drv_name(fts)].dropna()
    df_tf["Wheel_Drive_Angle"] = df_tf.apply(
        lambda row: wheel_drive_quat_to_euler_angle(
            row["QX"], row["QY"], row["QZ"], row["QW"]
        ),
        axis=1,
    )
    df_tf["Delta_T"] = df_tf["Timestamp"] - df_tf["Timestamp"].shift(1)
    df_tf["Delta_Angle"] = df_tf["Wheel_Drive_Angle"] - df_tf[
        "Wheel_Drive_Angle"
    ].shift(1)
    df_tf = df_tf.where(0 < df_tf["Delta_T"]).dropna()
    df_tf["Angular_Velocity"] = df_tf.apply(
        lambda row: abs(get_angular_velocity(row["Delta_T"], row["Delta_Angle"])),
        axis=1,
    )
    axs[0, 0].plot(
        rel_time(df_tf),
        df_tf["Angular_Velocity"],
        marker=marker_style,
        markersize=marker_size,
        label="Wheel rotation",
        linewidth=linewidth,
    )
    axs[0, 0].set_title("Wheel Angular Velocity")
    axs[0, 0].set_ylabel("Angular velocity [rad/s]", rotation=90)
    axs[0, 1].plot(
        rel_time(df_tf),
        df_tf["Wheel_Drive_Angle"],
        marker=marker_style,
        markersize=marker_size,
        label="Wheel_Drive_Angle",
        linewidth=linewidth,
    )
    axs[0, 1].set_title("Wheel Drive Angle")
    axs[0, 1].set_ylabel("Angle [rad]", rotation=90)


def plot_lever(
    cut_to_lever: bool,
    lever_min: float,
    lever_max: float,
    marker_size: float,
    marker_style: str,
    linewidth: float,
    axs: AxesArray,
    df: DataFrame,
) -> None:
    """
    Plot the lever length Ty/Fx

    :param cut_to_lever: Whether to cut to lever.
    :param lever_min: The minimum lever length.
    :param lever_max: The maximum lever length.
    :param marker_size: The size of the markers.
    :param marker_style: The style of the markers.
    :param linewidth: The width of the lines.
    :param axs: The axes to plot on.
    :param df: The dataframe.
    """
    lever: Series[float] = abs(df["Torque_Y"] / df["Force_X"])
    if cut_to_lever:
        df.where(lever_min < lever, inplace=True)
        df.where(lever < lever_max, inplace=True)
        df = df.dropna()
        lever = abs(df["Torque_Y"] / df["Force_X"])
    axs[0, 2].plot(
        rel_time(df),
        lever,
        marker=marker_style,
        markersize=marker_size,
        linewidth=linewidth,
        label="Lever [m]",
    )
    axs[0, 2].set_title("Lever")
    axs[0, 2].set_ylabel("Length [m]", rotation=90)


def plot_imu(
    marker_size: float,
    marker_style: str,
    linewidth: float,
    axs: AxesArray,
    t_start: int,
    t_end: int,
    traverse_dir_abs: str,
    t0: int,
) -> None:
    """
    Read and plot IMU acceleration and orientation data

    :param marker_size: The size of the markers.
    :param marker_style: The style of the markers.
    :param linewidth: The width of the lines.
    :param axs: The axes to plot on.
    :param t_start: The traverse start time.
    :param t_end: The traverse end time.
    :param traverse_dir_abs: The absolute path to the traverse directory.
    :param t0: Reference start time for relative time, e.g. df["Timestamp"].iloc[0].
    """
    file_path: str = os.path.join(traverse_dir_abs, "IMU.csv")
    df_imu: DataFrame = pd.read_csv(file_path)
    df_imu = df_imu[
        (t_start <= df_imu["Timestamp"]) & (df_imu["Timestamp"] <= t_end)
    ].dropna()

    ax_row = 3
    for i, field in enumerate(
        [
            "Linear_Acceleration_X",
            "Linear_Acceleration_Y",
            "Linear_Acceleration_Z",
        ]
    ):
        axs[ax_row, i % 3].plot(
            rel_time(df_imu, t0),
            df_imu[field],
            marker=marker_style,
            markersize=marker_size,
            label=f"IMU {field}",
            linewidth=linewidth,
        )
        axs[ax_row, i % 3].set_title(f"IMU {field.split('_')[2]} Axis")
        axs[ax_row, i % 3].set_ylabel("Acceleration [m/s²]", rotation=90)

    ax_row = 4
    for i, field in enumerate(["Angle_X", "Angle_Y", "Angle_Z"]):
        axs[ax_row, i % 3].plot(
            rel_time(df_imu, t0),
            df_imu[field] * 180 / np.pi,
            marker=marker_style,
            markersize=marker_size,
            label=f"IMU {field}",
            linewidth=linewidth,
        )
        axs[ax_row, i % 3].set_title(f"IMU {field.split('_')[1]} Axis")
        axs[ax_row, i % 3].set_ylabel("Angle [deg]", rotation=90)


def plot_extra_fx_ty_lever(axs: AxesArray) -> None:
    """
    Plot new figure containing Fx, Ty, and lever length.

    :param axs: The axes to copy data from.
    """
    _, axs_new = plt.subplots(nrows=3, ncols=1)
    copy_axis_content(axs[1, 0], axs_new[0])
    copy_axis_content(axs[2, 1], axs_new[1])
    copy_axis_content(axs[0, 2], axs_new[2])
    axs_new[1].sharex(axs_new[0])
    axs_new[2].sharex(axs_new[0])


def plot_extra_fts_and_acceleration(axs: AxesArray) -> None:
    """
    Plot new figure FTS and IMU acceleration data.

    :param axs: The axes to copy data from.
    """
    _, axs_new = plt.subplots(nrows=3, ncols=3)
    for i in range(3):
        for j in range(3):
            copy_axis_content(axs[i + 1, j], axs_new[i, j])
            axs_new[i, j].sharex(axs_new[0, 0])

    # Limit y axis for better comparison between different traverse plots

    axs_new[0, 0].set_ylim([-60, 60])
    axs_new[0, 1].set_ylim([-60, 60])
    axs_new[0, 2].set_ylim([-60, 60])

    axs_new[1, 0].set_ylim([-10, 10])
    axs_new[1, 1].set_ylim([-10, 10])
    axs_new[1, 2].set_ylim([-10, 10])

    axs_new[2, 0].set_ylim([-10, 20])
    axs_new[2, 1].set_ylim([-10, 20])
    axs_new[2, 2].set_ylim([-10, 20])


def read_fts_data(
    fts: str, t_start: int, t_end: int, traverse_dir_abs: str
) -> DataFrame:
    """
    Read the FTS data from the CSV file and takes care of the FTS orientation.

    :param fts: The FTS name.
    :param t_start: The start time.
    :param t_end: The end time.
    :param traverse_dir_abs: The absolute path to the traverse directory.
    :return: The dataframe containing the FTS data.
    """
    file_path: str = os.path.join(traverse_dir_abs, f"FTS_{fts}_CORRECTED.csv")
    df: DataFrame = pd.read_csv(file_path)
    df = df[(t_start <= df["Timestamp"]) & (df["Timestamp"] <= t_end)].dropna()

    # for field in ["Force_X", "Force_Y", "Force_Z", "Torque_X", "Torque_Y", "Torque_Z"]:
    #     df[field] = [
    #         float(np.format_float_positional(x, 4, trim="-")) for x in df[field]
    #     ]

    # FTS on the right is mounted backwards
    if fts[1] == "R":
        df["Force_X"] *= -1
        df["Force_Y"] *= -1

    return df


def main() -> None:
    # matplotlib.use("Agg")  # headless plotting
    baseprod_path: str = baseprod_helpers.data_path
    fts = "BL"  # FL, FR, CL, CR, BL, BR

    # Lever settings
    cut_to_lever = False
    lever_min = 0.08
    lever_max = 0.195

    # Plot mean and standard deviation?
    plot_stats = False
    rolling_window_sec = 1

    # Plot settings
    marker_size = 2
    marker_style = "."
    linewidth = 0.1
    interactive_plots = True
    use_tight_layout = True
    do_plot_extra_fx_ty_lever = True
    do_plot_extra_fts_and_acceleration = False

    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams.update({"font.size": 14})
    if not interactive_plots:
        plt.rcParams["figure.figsize"] = (19.20, 10.80)
        plt.rcParams["figure.dpi"] = 300

    for traverse_number, traverse in enumerate(traverses):
        _, _axs = plt.subplots(nrows=5, ncols=3)
        fig_title = f"{traverse_number}: {traverse[0]} {traverse[3]}"
        plt.suptitle(fig_title)

        # Typechecking
        axs: AxesArray = cast(AxesArray, _axs)

        # Cut to start/end timestamps
        traverse_directory, t_start_str, t_end_str, t_comment, t_type = traverse
        t_start = int(t_start_str)
        t_end = int(t_end_str)
        print(
            f"Traverse {traverse_number}: {(t_end - t_start) // 1e9} s, {traverse[3]}"
        )
        traverse_dir_abs = os.path.join(baseprod_path, traverse_directory)

        plot_wheel_data(
            fts,
            marker_size,
            marker_style,
            linewidth,
            axs,
            t_start,
            t_end,
            traverse_dir_abs,
        )

        df = read_fts_data(fts, t_start, t_end, traverse_dir_abs)

        plot_lever(
            cut_to_lever,
            lever_min,
            lever_max,
            marker_size,
            marker_style,
            linewidth,
            axs,
            df,
        )

        plot_fts_data(
            fts,
            plot_stats,
            rolling_window_sec,
            marker_size,
            marker_style,
            linewidth,
            axs,
            df,
        )

        plot_imu(
            marker_size,
            marker_style,
            linewidth,
            axs,
            t_start,
            t_end,
            traverse_dir_abs,
            df["Timestamp"].iloc[0],
        )

        axs[0, 2].set_ylim([0, 0.5])
        axs[1, 1].sharey(axs[1, 0])
        axs[1, 2].sharey(axs[1, 1])
        axs[2, 1].sharey(axs[2, 0])
        axs[2, 2].sharey(axs[2, 1])
        axs[1, 1].set_ylim([-60, 60])
        axs[2, 1].set_ylim([-10, 10])

        for i in range(5):
            for j in range(3):
                axs[i, j].sharex(axs[0, 0])
                axs[i, j].set_xlabel("Time [s]")

        if do_plot_extra_fx_ty_lever:
            plot_extra_fx_ty_lever(axs)

        if do_plot_extra_fts_and_acceleration:
            plot_extra_fts_and_acceleration(axs)

        if use_tight_layout:
            plt.tight_layout()
        else:
            plt.subplots_adjust(
                left=0.025, bottom=0.03, right=1, top=0.93, wspace=0.0, hspace=0.2
            )

        plt.savefig(fig_title)
        if interactive_plots:
            plt.show()


if __name__ == "__main__":
    main()
