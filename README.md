# A Robust Attention-based Cardinality Estimator with Domain Adaptation

A PyTorch implementation of AHCE and DACE. This repository provides the source code for data import, PostgreSQL cardinality estimation, and histogram extraction.

## Requirements

- Python 3.9.7
- PyTorch 1.10.1

## Dataset

The project is designed for PostgreSQL 11. If PostgreSQL is not installed, please follow the official installation guide: [PostgreSQL Download](https://www.postgresql.org/download/).

Our experiments are conducted on the [IMDB](https://www.imdb.com/interfaces/) and [Forest](https://archive.ics.uci.edu/ml/datasets/Covertype) datasets. The dataset can be imported into PostgreSQL using the provided script:

```shell
cd importForest
python forest.py
```

Additionally, the Attention-based Cardinality Estimator integrates histogram information. The histogram data can be extracted using the following example:

```shell
cd histogramGenerator
jupyter notebook getForestHistogram.ipynb
```

This will generate `histogram_forest.csv`.

## Running Experiments

### AHCE

AHCE supports single-table and multi-table cardinality estimation. The model can be evaluated using different workloads. You can specify the workload name and execute the model accordingly.

### DACE

DACE mitigates query workload drift using domain adaptation techniques, including:

- **MMD-based adaptation**
- **GRL-based adaptation**

Both approaches help improve estimation accuracy under workload shifts.

### PostgreSQL Baseline

To compare AHCE with PostgreSQL’s native estimator, run the following script:

```shell
cd get-postgresql-cardinality
python get_cardinality_estimate_actual.py
```

This will generate a CSV file containing estimated cardinalities, actual values, and q-errors.

## Repository Structure

```
AHCE-DACE/
│── AHCE/                      # Attention-based Cardinality Estimator implementation
│── DACE/                      # Domain Adaptation for Cardinality Estimation
│── get-postgresql-cardinality # PostgreSQL estimation baseline
│── histogramGenerator/        # Histogram extraction scripts
│── importForest/              # Data import scripts
│── README.md                  # Project documentation
```

## Code References

- [LW-NN/XGB](https://github.com/jt-zhang/CardinalityEstimationTestbed)
- [MSCN](https://github.com/andreaskipf/learnedcardinalities)
- [NNGP](https://github.com/Kangfei/NNGP-src)
- [DeepDB](https://github.com/DataManagementLab/deepdb-public)
- [NeuroCard](https://github.com/neurocard/neurocard)
- [UAE](https://github.com/pagegitss/UAE)
- [DANN](https://github.com/zengjichuan/DANN)