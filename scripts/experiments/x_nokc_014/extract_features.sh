#!/bin/bash
###############################################################################
## x_nokc: prepare data and extract all features                          ##
###############################################################################

# parameter
export PYTHONPATH="."
DATASET="x_nokc"
NTHREADS=7
SPLITS=5

# available features classes
OH_FEATURES="-i -bundle -c"
COUNT_FEATURES="-tcA -tcW -icA -icW"
TW_FEATURES="-tcA_TW -tcW_TW -icA_TW -icW_TW"
DATE_FEATURES="-month -week -day -hour -weekend -part_of_day"
AVERAGE_CORRRECT="-user_avg_correct"
NGRAM="-n_gram"

# select feature classes for extraction
FS=(
    "$OH_FEATURES"
    "$COUNT_FEATURES"
    "$TW_FEATURES"
    "$DATE_FEATURES"
    "$AVERAGE_CORRRECT"
    "$NGRAM"
)


# extract features
echo "starting feature extraction"
for f in "${FS[@]}"; do
    python ./src/preprocessing/extract_features.py \
        --dataset=$DATASET \
        --num_threads=$NTHREADS \
        -recompute \
        $f
done
