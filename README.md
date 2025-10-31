# FTS Assessment

[![Paper](https://img.shields.io/badge/Paper-JIRS-blue)](https://doi.org/10.1007/s10846-025-02324-2)
[![Data](https://img.shields.io/badge/Data-DOI-green)](https://doi.org/10.57780/esa-xxd1ysw)
[![Citation](https://img.shields.io/badge/Cite-This%20Work-orange)](#citation)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![CI-Pytest](https://github.com/spaceuma/fts-assessment/actions/workflows/pytest.yml/badge.svg)](https://github.com/spaceuma/fts-assessment/actions/workflows/pytest.yml)
[![CI-Ruff](https://github.com/spaceuma/fts-assessment/actions/workflows/ruff.yml/badge.svg)](https://github.com/spaceuma/fts-assessment/actions/workflows/ruff.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

These scripts were used to prepare the paper ["Field Assessment of Force Torque Sensors for Planetary Rover Navigation"](https://doi.org/10.1007/s10846-025-02324-2).
The source code is licensed under the [MIT License](LICENSE).

## Links

- Paper (Oct. 2025): <https://doi.org/10.1007/s10846-025-02324-2>
- Data: <https://doi.org/10.57780/esa-xxd1ysw>
- Dataset paper: <https://doi.org/10.1038/s41597-024-03881-1>

## Reproducibility

Results in the [paper](https://doi.org/10.1007/s10846-025-02324-2) were generated using code release [v0.2-revision](https://github.com/spaceuma/fts-assessment/tree/v0.2-revision).
The [preprint](https://doi.org/10.48550/arXiv.2411.04700) was prepared with code release
[v0.1-preprint](https://github.com/spaceuma/fts-assessment/tree/v0.1-preprint).

## Dependencies

This repository uses Python 3.10.
All requirements are listed in [pyproject.toml](pyproject.toml).

Install them via `uv`

```bash
uv sync
uv pip install -e .
```

or `pip`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Data path

Point the scripts to the [Baseprod](https://doi.org/10.1038/s41597-024-03881-1) traverse data. You can use your own path like so:

1. Set the environment variable for your path, e.g.

    ```bash
    export BASEPROD_TRAVERSE_PATH="/mnt/baseprod/sensor_data"
    ```

2. Alternatively, modify the default path (if no correct environment variable is found) in
[preprocessing/traverse_overview](preprocessing/traverse_overview.py).

## Preprocessing

- Check how much data is usable according to the distance computed by Fy/Tx with [preprocessing/find_usable.py](preprocessing/find_usable.py).

    ```bash
    python -m preprocessing.find_usable
    ```

- Generate the dataset for the later machine learning with [preprocessing/export_classification_stats.py](preprocessing/export_classification_stats.py).

    ```bash
    python -m preprocessing.export_classification_stats
    ```

  Creates three output files (`training_data.csv`, `training_data_ft.csv`,
  `training_data_imu.csv`) containing the data for all sensors, only the force
  torque sensors, and only the IMU data, respectively.

  If only a subset of FTSs should be included, `--fts_names FL FR` can be passed
  to only include the data from FTSs `FL` and `FR` (plus IMU) for example.

- Plot FTS and IMU data with [preprocessing/plot_fts_imu.py](preprocessing/plot_fts_imu.py)

    ```bash
    python -m preprocessing.plot_fts_imu
    ```

## Run classification training

From the project's root, invoke

```bash
python -m ml.svm --csv training_data_ft.csv --data_source fts
```

or

```bash
python -m ml.train --csv training_data.csv --data_source all
```

See all possible arguments by passing `--help`.

## Citation

If you find this work helpful, please cite it as:

Gerdes, L., Pérez del Pulgar, C., Castilla Arquillo, R., Azkarate, M. Field Assessment of Force Torque Sensors for Planetary Rover Navigation. *J Intell Robot Syst* **111**, 122 (2025). <https://doi.org/10.1007/s10846-025-02324-2>

```bibtex
@article{Gerdes2025FTS,
  author={Gerdes, Levin and P{\'e}rez del Pulgar, Carlos and Castilla Arquillo, Ra{\'u}l and Azkarate, Martin},
  title={Field Assessment of Force Torque Sensors for Planetary Rover Navigation},
  journal={Journal of Intelligent {\&} Robotic Systems},
  year={2025},
  month={Oct},
  day={30},
  volume={111},
  number={4},
  pages={122},
  doi={10.1007/s10846-025-02324-2},
  url={https://doi.org/10.1007/s10846-025-02324-2}
}
```
