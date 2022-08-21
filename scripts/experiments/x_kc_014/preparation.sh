#!/bin/bash
###############################################################################
## EdNet_KT3: prepare data and extract all features                          ##
###############################################################################

# parameter
export PYTHONPATH="."
DATASET="x_kc_014"
NTHREADS=7
SPLITS=5


# prepare data file
echo "starting data preparation"
python ./src/preparation/prepare_data.py \
    --dataset=$DATASET \
    --n_splits=$SPLITS


# available features classes
OH_FEATURES="-i -s -bundle -c"
COUNT_FEATURES="-tcA -tcW -scA -scW -icA -icW"
TW_FEATURES="-tcA_TW -tcW_TW -scA_TW -scW_TW -icA_TW -icW_TW"
DATE_FEATURES="-month -week -day -hour -weekend -part_of_day"
AVERAGE_CORRRECT="-user_avg_correct"
NGRAM="-n_gram"
RPFA="-rpfa_F -rpfa_R"
PPE="-ppe"

# select feature classes for extraction
FS=(
    "$OH_FEATURES"
    "$COUNT_FEATURES"
    "$TW_FEATURES"
    "$DATE_FEATURES"
    "$AVERAGE_CORRRECT"
    "$NGRAM"
    "$RPFA"
    "$PPE"
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
