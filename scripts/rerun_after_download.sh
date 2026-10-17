#!/usr/bin/env bash
# Re-run the full analysis (data already downloaded). Logs: results/logs/<step>.log
set -e
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
step() { name=$1; shift; echo "[$(date +%H:%M:%S)] start $name"; $PY -u "$@" > results/logs/$name.log 2>&1; echo "[$(date +%H:%M:%S)] done  $name"; }
step 01_preprocess_k562   scripts/01_preprocess.py --dataset replogle_k562_essential
step 01_preprocess_adamson scripts/01_preprocess.py --dataset adamson
step 02_embeddings        scripts/02_build_embeddings.py
step 04_noise_k562        scripts/04_noise_ceiling.py --dataset replogle_k562_essential
step 04_noise_adamson     scripts/04_noise_ceiling.py --dataset adamson
step 03_benchmark_k562    scripts/03_benchmark.py --dataset replogle_k562_essential --models all --n_ens 3
step 05_flow              scripts/05_flow.py
step 06_external          scripts/06_external_validation.py
step 07_interpretation    scripts/07_interpretation.py
step 09_figures           scripts/09_figures.py
step 08_train_final       scripts/08_train_final.py
step predict_GATA1        scripts/predict.py --gene GATA1
step predict_TP53         scripts/predict.py --gene TP53
step 10_explorer          scripts/10_export_explorer.py
step 03_benchmark_adamson scripts/03_benchmark.py --dataset adamson --models all --n_ens 3
step 03b_dev_sweep        scripts/03b_dev_sweep.py
echo "[$(date +%H:%M:%S)] ALL DONE"
