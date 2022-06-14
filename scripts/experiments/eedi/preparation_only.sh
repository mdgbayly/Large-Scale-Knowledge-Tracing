#!/bin/bash
###############################################################################
## Eedi: prepare data and extract all features                               ##
###############################################################################

# parameter
export PYTHONPATH="."
DATASET="eedi"
NTHREADS=31
SPLITS=5


# prepare data file
echo "starting data preparation"
python ./src/preparation/prepare_data.py \
    --dataset=$DATASET \
    --n_splits=$SPLITS
