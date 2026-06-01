#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 3.25 GB
# Estimated disk usage after selected extraction: 3.26 GB for the default
# small tasks, up to about 10 GB if all released tensors are extracted.
#
# This script is intentionally workspace-local: it writes only under
# data/downloads/ so a fresh clone can reproduce the benchmark inputs.

set -e

cd /home/user

mkdir -p data/downloads

if [ ! -e data/downloads/coda_data.zip ]; then
  gdown '1H8zXwAGkkAQP5L1gofpeF69jeIZqzPaW' -O data/downloads/coda_data.zip
fi

python data/extract_coda_release.py --tasks glue_rte glue_sst2 cifar10_5592
python data/prepare_modelselector.py --tasks glue_rte glue_sst2 cifar10_5592
