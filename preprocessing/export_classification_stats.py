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
Compute Truncated SVD for BASEPROD

Collects FTS and IMU data from different BASEPROD traverses. The script then
reduces the dimensionality via Truncated SVD (Singular Value Decomposition) and
saves the transformed data to CSV files for further use in classification.
"""

__author__ = "Levin Gerdes"

import argparse
import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore
from pandas import DataFrame
from sklearn.decomposition import PCA, TruncatedSVD  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from . import baseprod_helpers
from .baseprod_helpers import Terrain, TraverseList
from .baseprod_helpers import labelled_traverses as traverses


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process BASEPROD FTS and IMU data.")
    parser.add_argument(
        "--baseprod_path",
        type=str,
        default=baseprod_helpers.data_path,
        help="Path to the BASEPROD directory",
    )
    parser.add_argument(
        "--rolling_window_sec",
        type=float,
        default=0.5,
        help="Size of the rolling window in seconds",
    )
    parser.add_argument(
        "--fts_names",
        nargs="*",
        default=["FL", "FR", "CL", "CR", "BL", "BR"],
        help="List of FTS sensor names to export",
    )
    parser.add_argument(
        "--only_export",
        action="store_true",
        help="If set, only export traverse data without further processing or visualizations",
    )
    return parser.parse_args()


def count_raw_samples(
    baseprod_path: str,
    traverses: TraverseList,
    fts_names: list[str],
) -> None:
    """
    Counts and prints number of raw data points for each terrain type and sensor.

    :param baseprod_path: Path to the BASEPROD directory
    :param traverses: List of traverses with as given in traverse_overview.py
    :param fts_names: List of FTS sensor names
    """
    all_sensors: list[str] = ["IMU"] + fts_names
    samples_per_terrain: dict[Terrain, int] = {terrain: 0 for terrain in Terrain}
    samples_per_terrain_and_sensor: dict[tuple[str, Terrain], int] = {
        (sensor, terrain): 0 for sensor in all_sensors for terrain in Terrain
    }

    for traverse in traverses:
        traverse_directory, t_start_str, t_end_str, t_comment, terrain_type = traverse
        traverse_dir_abs = os.path.join(baseprod_path, traverse_directory)

        df_imu: DataFrame = pd.read_csv(os.path.join(traverse_dir_abs, "IMU.csv"))
        df_imu = df_imu[df_imu["Timestamp"].between(int(t_start_str), int(t_end_str))]

        samples_per_terrain[Terrain[terrain_type]] += df_imu.shape[0]
        samples_per_terrain_and_sensor[("IMU", Terrain[terrain_type])] += df_imu.shape[
            0
        ]

        for fts in fts_names:
            df_fts: DataFrame = pd.read_csv(
                os.path.join(traverse_dir_abs, f"FTS_{fts}_CORRECTED.csv")
            )
            df_fts = df_fts[
                df_fts["Timestamp"].between(int(t_start_str), int(t_end_str))
            ]

            samples_per_terrain[Terrain[terrain_type]] += df_fts.shape[0]
            samples_per_terrain_and_sensor[(fts, Terrain[terrain_type])] += (
                df_fts.shape[0]
            )

    print("Raw data points: ", samples_per_terrain, sum(samples_per_terrain.values()))

    for sensor, terrain in samples_per_terrain_and_sensor.keys():
        print(
            f"Raw data points for {sensor} on {terrain}: {samples_per_terrain_and_sensor[(sensor, terrain)]}"
        )


def process_baseprod(
    baseprod_path: str,
    traverses: TraverseList,
    rolling_window_ns: int = int(1e9),
    fts_names: list[str] = ["FL", "FR", "CL", "CR", "BL", "BR"],
    add_lever: bool = True,
    cut_to_lever: bool = False,
    lever_min: float = 0.05,
    lever_max: float = 0.225,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reads FTS and IMU data from BASEPROD traverses and processes it for further use.
    Adding mean, std etc. for each window of size rolling_window_ns.

    :param baseprod_path: Path to the BASEPROD directory
    :param traverses: List of traverses with as given in traverse_overview.py
    :param rolling_window_ns: Size of the rolling window in nanoseconds
    :param fts_names: List of FTS sensor names
    :param add_lever: If True, add lever length to the FTS data
    :param cut_to_lever: If True, only use data points with lever length between lever_min and lever_max
    :param lever_min: Minimum lever length [m]
    :param lever_max: Maximum lever length [m]
    :return (df_all, df_ft, df_imu): DataFrames containing the processed data
    """
    if rolling_window_ns < 1:
        raise ValueError("Rolling window size must be at least 1 ns")
    if os.path.isdir(baseprod_path) is False:
        raise FileNotFoundError("BASEPROD directory not found")
    if len(traverses) == 0:
        raise ValueError("No traverses given")
    if cut_to_lever:
        if lever_min >= lever_max:
            raise ValueError("Lever minimum must be smaller than lever maximum")
        if lever_min < 0 or lever_max < 0:
            raise ValueError("Lever limits must be positive")

    imu_datapoints: dict[str, list[list[float]]] = {"IMU": []}
    ft_datapoints: dict[str, list[list[float]]] = {
        "FL": [],
        "FR": [],
        "CL": [],
        "CR": [],
        "BL": [],
        "BR": [],
    }

    for traverse_number, traverse in enumerate(traverses):
        traverse_directory, t_start_str, t_end_str, t_comment, terrain_type = traverse
        t_start = int(t_start_str)
        t_end = int(t_end_str)
        print(
            f"Traverse {traverse_number}: {(t_end - t_start) // 1e9} s, {traverse[3]}"
        )
        traverse_dir_abs = os.path.join(baseprod_path, traverse_directory)

        df_imu = pd.read_csv(os.path.join(traverse_dir_abs, "IMU.csv"))
        for fts in fts_names:
            df_ft = pd.read_csv(
                os.path.join(traverse_dir_abs, f"FTS_{fts}_CORRECTED.csv")
            )

            # FTS on the right is mounted backwards
            if fts[1] == "R":
                df_ft["Force_X"] *= -1
                df_ft["Force_Y"] *= -1

            if add_lever:
                df_ft["Lever"] = df_ft["Torque_Y"] / df_ft["Force_X"]

            # iterate with window of size rolling_window_ns
            for i in range(int((t_end - t_start) // rolling_window_ns)):
                current_start = t_start + i * rolling_window_ns
                current_end = t_start + (i + 1) * rolling_window_ns
                current_df = df_ft[
                    df_ft["Timestamp"].between(current_start, current_end)
                ]

                if cut_to_lever:
                    current_df = current_df.loc[
                        (abs(df_ft["Torque_Y"] / df_ft["Force_X"]) > lever_min)
                        & (abs(df_ft["Torque_Y"] / df_ft["Force_X"]) < lever_max)
                    ]

                    current_df.dropna()
                    if (
                        current_df.notna().shape[0] < current_df.shape[0]
                        or current_df.notna().shape[0] < 2
                    ):
                        # No applicable values found after filtering. Only NaN.
                        continue

                # for later PCA etc
                # if fts == "FR":
                if True:
                    ft_datapoints[fts].append(
                        [
                            Terrain[traverse[4]].value,
                            current_df["Force_X"].min(),
                            current_df["Force_X"].max(),
                            current_df["Force_X"].mean(),
                            current_df["Force_X"].median(),
                            current_df["Force_X"].std(),
                            current_df["Force_Y"].min(),
                            current_df["Force_Y"].max(),
                            current_df["Force_Y"].mean(),
                            current_df["Force_Y"].median(),
                            current_df["Force_Y"].std(),
                            current_df["Force_Z"].min(),
                            current_df["Force_Z"].max(),
                            current_df["Force_Z"].mean(),
                            current_df["Force_Z"].median(),
                            current_df["Force_Z"].std(),
                            current_df["Torque_X"].min(),
                            current_df["Torque_X"].max(),
                            current_df["Torque_X"].mean(),
                            current_df["Torque_X"].median(),
                            current_df["Torque_X"].std(),
                            current_df["Torque_Y"].min(),
                            current_df["Torque_Y"].max(),
                            current_df["Torque_Y"].mean(),
                            current_df["Torque_Y"].median(),
                            current_df["Torque_Y"].std(),
                            current_df["Torque_Z"].min(),
                            current_df["Torque_Z"].max(),
                            current_df["Torque_Z"].mean(),
                            current_df["Torque_Z"].median(),
                            current_df["Torque_Z"].std(),
                        ]
                    )
                    if add_lever:
                        ft_datapoints[fts][-1].extend(
                            [
                                current_df["Lever"].min(),
                                current_df["Lever"].max(),
                                current_df["Lever"].mean(),
                                current_df["Lever"].median(),
                                current_df["Lever"].std(),
                            ]
                        )

        # Repeat for IMU
        for i in range(int((t_end - t_start) // rolling_window_ns)):
            current_start = t_start + i * rolling_window_ns
            current_end = t_start + (i + 1) * rolling_window_ns
            current_idx = df_imu["Timestamp"].between(current_start, current_end)
            current_df_imu = df_imu[current_idx]

            imu_datapoints["IMU"].append(
                [
                    Terrain[traverse[4]].value,
                    current_df_imu["Linear_Acceleration_X"].min(),
                    current_df_imu["Linear_Acceleration_X"].max(),
                    current_df_imu["Linear_Acceleration_X"].mean(),
                    current_df_imu["Linear_Acceleration_X"].median(),
                    current_df_imu["Linear_Acceleration_X"].std(),
                    current_df_imu["Linear_Acceleration_Y"].min(),
                    current_df_imu["Linear_Acceleration_Y"].max(),
                    current_df_imu["Linear_Acceleration_Y"].mean(),
                    current_df_imu["Linear_Acceleration_Y"].median(),
                    current_df_imu["Linear_Acceleration_Y"].std(),
                    current_df_imu["Linear_Acceleration_Z"].min(),
                    current_df_imu["Linear_Acceleration_Z"].max(),
                    current_df_imu["Linear_Acceleration_Z"].mean(),
                    current_df_imu["Linear_Acceleration_Z"].median(),
                    current_df_imu["Linear_Acceleration_Z"].std(),
                ]
            )

    # Combine all
    fts = fts_names[0]
    combined_ft: list[list[float]] = copy.deepcopy(ft_datapoints[fts])
    if len(fts_names) > 1:
        for i, x in enumerate(combined_ft):
            for fts in fts_names[1:]:
                assert x[0] == ft_datapoints[fts][i][0]
                x.extend(ft_datapoints[fts][i][1:])
    combined_ft_and_imu = copy.deepcopy(combined_ft)
    for i, x in enumerate(combined_ft_and_imu):
        assert x[0] == imu_datapoints["IMU"][i][0]
        x.extend(imu_datapoints["IMU"][i][1:])

    # Save to file
    df_all = pd.DataFrame(combined_ft_and_imu).dropna()
    df_ft = pd.DataFrame(combined_ft).dropna()
    df_imu = pd.DataFrame(imu_datapoints["IMU"]).dropna()

    return (df_all, df_ft, df_imu)


def check_for_errors(
    df_all: pd.DataFrame, df_fts: pd.DataFrame, df_imu: pd.DataFrame
) -> None:
    """
    Check data for any erroneous entries.
    """
    print(
        df_all.shape,
        df_all.notna().shape,
        df_all.notnull().shape,
        df_fts.shape,
        df_fts.notna().shape,
        df_fts.notnull().shape,
        df_imu.shape,
        df_imu.notna().shape,
        df_imu.notnull().shape,
    )

    if df_all.isnull().values.any():
        raise ValueError("There are NaNs in the DataFrame")

    if np.isinf(df_all.values).any():
        raise ValueError("There are infinite values in the DataFrame")

    if not all(np.issubdtype(dtype, np.number) for dtype in df_all.dtypes):
        raise TypeError("Non-numeric entries")


def main() -> None:
    args = get_args()

    # matplotlib.use("Agg")  # headless plotting
    baseprod_path: str = args.baseprod_path
    add_lever = True
    cut_to_lever = False
    rolling_window_sec: float = args.rolling_window_sec
    rolling_window_ns: int = int(rolling_window_sec * 1e9)
    process_bardenas: bool = True
    only_export_traverse_data: bool = args.only_export
    fts_names: list[str] = args.fts_names

    # Collect datapoints to compute PCA, SVD etc

    count_raw_samples(
        baseprod_path=baseprod_path, traverses=traverses, fts_names=fts_names
    )

    if process_bardenas:
        df_all, df_ft, df_imu = process_baseprod(
            baseprod_path=baseprod_path,
            traverses=traverses,
            rolling_window_ns=rolling_window_ns,
            fts_names=fts_names,
            add_lever=add_lever,
            cut_to_lever=cut_to_lever,
        )

        print(
            "Stats samples per terrain type:",
            df_all[df_all[0] == 1].shape[0],
            df_all[df_all[0] == 2].shape[0],
            df_all[df_all[0] == 3].shape[0],
            df_all[df_all[0] == 4].shape[0],
            df_all.shape[0],
        )

        df_all.to_csv("training_data.csv", index=False)
        df_ft.to_csv("training_data_ft.csv", index=False)
        df_imu.to_csv("training_data_imu.csv", index=False)

        if only_export_traverse_data:
            exit()

    else:
        df_all = pd.read_csv("training_data.csv")
        df_ft = pd.read_csv("training_data_ft.csv")
        df_imu = pd.read_csv("training_data_imu.csv")
    df_ft = df_ft.dropna(axis=0)
    df_imu = df_imu.dropna(axis=0)
    df_all = df_all.dropna(axis=0)

    check_for_errors(df_all, df_ft, df_imu)

    # scale data
    X_ft = np.array([x[1:] for x in df_ft.to_numpy()])
    Y_ft = np.array([int(x[0]) for x in df_ft.to_numpy()]).reshape((-1, 1))
    X_imu = np.array([x[1:] for x in df_imu.to_numpy()])
    Y_imu = np.array([int(x[0]) for x in df_imu.to_numpy()]).reshape((-1, 1))
    X_all = np.array([x[1:] for x in df_all.to_numpy()])
    Y_all = np.array([int(x[0]) for x in df_all.to_numpy()]).reshape((-1, 1))
    print(X_ft.shape, Y_ft.shape)
    print(X_imu.shape, Y_imu.shape)

    scaler = StandardScaler()
    X_ft_scaled = scaler.fit_transform(X_ft)
    X_imu_scaled = scaler.fit_transform(X_imu)
    X_all_scaled = scaler.fit_transform(X_all)

    # PCA
    pca = PCA(n_components=2)
    pca_res_ft = pca.fit_transform(X_ft_scaled)

    data_ft = pd.DataFrame(pca_res_ft, columns=["PCA Component 1", "PCA Component 2"])
    data_ft["Target"] = Y_ft

    plt.figure("PCA of FTSs")
    sns.scatterplot(
        data=data_ft, x="PCA Component 1", y="PCA Component 2", hue="Target"
    )
    plt.show()

    # compute truncated SVD
    tsvd = TruncatedSVD(n_components=2)
    X_ft_svd = tsvd.fit_transform(X_ft_scaled)
    X_imu_svd = tsvd.fit_transform(X_imu_scaled)
    X_all_svd = tsvd.fit_transform(X_all_scaled)

    data_ft = pd.DataFrame(X_ft_svd, columns=["SVD Component 1", "SVD Component 2"])
    data_ft["Target"] = Y_ft

    data_imu = pd.DataFrame(X_imu_svd, columns=["SVD Component 1", "SVD Component 2"])
    data_imu["Target"] = Y_imu

    data_all = pd.DataFrame(X_all_svd, columns=["SVD Component 1", "SVD Component 2"])
    data_all["Target"] = Y_all

    plt.figure("SVD results for FTSs")
    sns.scatterplot(
        data=data_ft, x="SVD Component 1", y="SVD Component 2", hue="Target"
    )
    plt.show()

    plt.figure("SVD results for IMU")
    sns.scatterplot(
        data=data_imu, x="SVD Component 1", y="SVD Component 2", hue="Target"
    )
    plt.show()

    plt.figure("SVD results for all data")
    sns.scatterplot(
        data=data_all, x="SVD Component 1", y="SVD Component 2", hue="Target"
    )
    plt.show()

    # tsne
    ft_x_and_y_scaled = np.concatenate(
        (np.array(Y_ft).reshape((len(Y_ft), 1)), X_ft_scaled), axis=1
    )
    # all_scaled_imu = np.concatenate((np.array(Y), X_imu_scaled), axis=1)

    columns: list[str] = [f"col_{i}" for i in range(ft_x_and_y_scaled.shape[1])]
    columns[0] = "y"
    df_ft_scaled = pd.DataFrame(
        columns=columns,
        # data=datapoints["FR"],
        data=ft_x_and_y_scaled,
    )

    imu_x_and_y_scaled = np.concatenate(
        (np.array(Y_imu).reshape((len(Y_imu), 1)), X_imu_scaled), axis=1
    )
    columns = [f"col_{i}" for i in range(imu_x_and_y_scaled.shape[1])]
    columns[0] = "y"
    df_imu_scaled = pd.DataFrame(
        columns=columns,
        # data=datapoints["FR"],
        data=imu_x_and_y_scaled,
    )

    # use pca features too
    print(pca_res_ft.shape)
    for i in range(pca_res_ft.shape[1]):
        df_ft_scaled[f"pca_fatures_{i}"] = pca_res_ft[:, i]

    print(df_ft_scaled, df_imu_scaled)

    time_start: float = time.time()
    tsne = TSNE(n_components=2, init="random", verbose=1, perplexity=20, n_iter=2000)
    tsne_res = tsne.fit_transform(df_ft_scaled)
    # tsne_res = tsne.fit_transform(np.array(X))
    print(
        "t-SNE of FTS done! Time elapsed: {} seconds".format(time.time() - time_start),
        tsne_res.shape,
        tsne_res,
    )

    plt.figure("t-SNE of FTSs")
    sns.scatterplot(
        x=tsne_res[:, 0],
        y=tsne_res[:, 1],
        # hue="y",
        hue="y",
        palette=sns.color_palette(palette="hls", n_colors=4),
        data=df_ft_scaled,
        legend="full",
        alpha=0.3,
    )
    plt.show()

    tsne_res_imu = tsne.fit_transform(df_imu_scaled)
    plt.figure("t-SNE of IMU")
    sns.scatterplot(
        x=tsne_res_imu[:, 0],
        y=tsne_res_imu[:, 1],
        hue="y",
        palette=sns.color_palette(palette="hls", n_colors=4),
        data=df_imu_scaled,
        legend="full",
        alpha=0.3,
    )
    plt.show()

    # 3D
    tsne = TSNE(n_components=3, init="random", verbose=1, perplexity=50, n_iter=5000)
    tsne_res = tsne.fit_transform(df_ft_scaled)

    fig = plt.figure("3D t-SNE of FTSs")
    ax: plt.Axes = fig.add_subplot(projection="3d")
    ax.scatter(tsne_res[:, 0], tsne_res[:, 1], tsne_res[:, 2], c=df_ft_scaled["y"])
    plt.show()


if __name__ == "__main__":
    main()
