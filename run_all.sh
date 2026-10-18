#!/usr/bin/env bash
# Reproduce every number and figure of the project (CPU only, about 4 h on 4 cores).
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
$PY -m pytest -q tests                                   # data-free unit tests (seconds)
bash scripts/00_download_data.sh
$PY -u scripts/01_preprocess.py --dataset replogle_k562_essential
$PY -u scripts/01_preprocess.py --dataset adamson
$PY -u scripts/02_build_embeddings.py
$PY -u scripts/04_noise_ceiling.py --dataset replogle_k562_essential
$PY -u scripts/04_noise_ceiling.py --dataset adamson
$PY -u scripts/03b_dev_sweep.py                          # model selection on the dev split only
$PY -u scripts/03_benchmark.py --dataset replogle_k562_essential --models all --n_ens 3
for SEED in 1 2; do                                      # robustness to the random CV split
  $PY -u scripts/03_benchmark.py --dataset replogle_k562_essential --models Linear,kNN,PerturbFormer,PerturbNet \
      --n_ens 3 --seed $SEED --tag _seed$SEED
done
$PY -u scripts/03c_split_robustness.py
$PY -u scripts/03d_baseline_tuning.py                    # baselines tuned on the same dev split
$PY -u scripts/03_benchmark.py --dataset adamson --models all --n_ens 3
$PY -u scripts/05_flow.py                                # fold-specific PCA, leak-free; includes borrowed-cells baseline
# optional, ~1 h each, needs the GEARS extras from requirements.txt; run alone (needs ~4.5 GB RAM):
# $PY -u scripts/12_gears_baseline.py --cells_per_pert 15 --n_ctrl 1500 --epochs 10
$PY -u scripts/06_external_validation.py
$PY -u scripts/07_interpretation.py
$PY -u scripts/11_representation.py
$PY -u scripts/09_figures.py
$PY -u scripts/08_train_final.py
$PY -u scripts/predict.py --gene MYC
$PY -u scripts/10_export_explorer.py
$PY -u docs/build_slides.py
