#!/bin/bash
###############################################################################
## Experiment: Evaluation of individual features                             ##
###############################################################################

# parameters
export PYTHONPATH="."
DATASET="x_nokc_014"
EXPNAME="single_feature_evaluation"
NPROCESSES=1
NTHREADS=4
SPLITS=5

OH_FEATURES="-i -bundle -c"
COUNT_FEATURES="-tcA -tcW -icA -icW"
TW_FEATURES="-tcA_TW -tcW_TW -icA_TW -icW_TW"
DATE_FEATURES="-month -week -day -hour -weekend -part_of_day"
AVERAGE_CORRRECT="-user_avg_correct"
NGRAM="-n_gram"

IT_FEATURES=""
GRAPH_FEATURE=""
VIDEO_FEATURES=""
READING_FEATURE=""
SM_FEATURES=""
RPFA=""
PPE=""

cmd=""
for (( i=0; i<$SPLITS; i++ )); do
#---------------------------------------------------------#
# One-hot features                                        #
#---------------------------------------------------------#
for f in $OH_FEATURES; do
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $f\n"
done

#---------------------------------------------------------#
# Count features                                          #
#---------------------------------------------------------#
# Comes in pairs and combined
for f in "-tcA -tcW" "-icA -icW"; do
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $f\n"
done

cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $COUNT_FEATURES\n"

#---------------------------------------------------------#
# Time window features                                    #
#---------------------------------------------------------#
# Comes in pairs and combined
for f in "-tcA_TW -tcW_TW" "-icA_TW -icW_TW"; do
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $f\n"
done

cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $TW_FEATURES\n"

#---------------------------------------------------------#
# Interaction time features                               #
#---------------------------------------------------------#
for f in $IT_FEATURES; do
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $f\n"
done

#---------------------------------------------------------#
# Datetime features                                       #
#---------------------------------------------------------#
# month, week, day, hour, weekend, part of day
for f in $DATE_FEATURES; do
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $f\n"
done

#---------------------------------------------------------#
# Video features                                          #
#---------------------------------------------------------#
for f in $VIDEO_FEATURES; do
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $f\n"
done

#---------------------------------------------------------#
# Reading features                                        #
#---------------------------------------------------------#
for f in $READING_FEATURE; do
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $f\n"
done

#---------------------------------------------------------#
# study module/part specific counts                       #
# NOTE: COMBINED!!!! JUST LOOPING TO FILTER OUT FOR NOW   #
#---------------------------------------------------------#

#---------------------------------------------------------#
# Student average correct                                 #
#---------------------------------------------------------#
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $AVERAGE_CORRRECT\n"

#---------------------------------------------------------#
# n-gram feature                                          #
#---------------------------------------------------------#
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $NGRAM\n"

#---------------------------------------------------------#
# PPE feature                                             #
#---------------------------------------------------------#

#---------------------------------------------------------#
# RPFA features                                           #
#---------------------------------------------------------#
# Comes in pairs and combined

# --------------------------------------------------------#
done


echo -e $cmd
echo "============================="
echo "Waiting 10s before starting jobs...."
sleep 10s
echo -e $cmd | xargs -n1 -P$NPROCESSES -I{} -- bash -c '{}'
