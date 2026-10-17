#!/usr/bin/env bash
# Reproduce every number and figure of the project (CPU only, ~2.5 h on 4 cores).
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
bash scripts/00_download_data.sh
$PY -u scripts/01_preprocess.py --dataset replogle_k562_essential
$PY -u scripts/01_preprocess.py --dataset adamson
$PY -u scripts/02_build_embeddings.py
$PY -u scripts/03_benchmark.py --dataset replogle_k562_essential --models all --n_ens 3
$PY -u scripts/03_benchmark.py --dataset adamson --models all --n_ens 3
$PY -u scripts/03b_dev_sweep.py
$PY -u scripts/04_noise_ceiling.py --dataset replogle_k562_essential
$PY -u scripts/04_noise_ceiling.py --dataset adamson
$PY -u scripts/05_flow.py
$PY -u scripts/06_external_validation.py
$PY -u scripts/07_interpretation.py
$PY -u scripts/09_figures.py
$PY -u scripts/08_train_final.py
$PY -u scripts/predict.py --gene GATA1
