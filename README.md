# FTS Assessment

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![CI-Pytest](https://github.com/spaceuma/fts-assessment/actions/workflows/pytest.yml/badge.svg)](https://github.com/spaceuma/fts-assessment/actions/workflows/pytest.yml)
[![CI-Ruff](https://github.com/spaceuma/fts-assessment/actions/workflows/ruff.yml/badge.svg)](https://github.com/spaceuma/fts-assessment/actions/workflows/ruff.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the scripts we used to assess force-torque sensor data
for a paper that is currently under review.

- Author: Levin Gerdes [![orcid](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0001-7648-8928)
- Supervisor: Carlos J. Pérez del Pulgar [![orcid](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0001-5819-8310)

## Dependencies

Requirements are listed in [pyproject.toml](pyproject.toml).
One way to install them:

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
- Generate the dataset for the later Neural Network training with [preprocessing/export_classification_stats.py](preprocessing/export_classification_stats.py).
- Plot FTS and IMU data with [preprocessing/plot_fts_imu.py](preprocessing/plot_fts_imu.py)

## Run classification training

From the project's root, invoke

```bash
python -m ml.svm
```

or

```bash
python -m ml.train
```
