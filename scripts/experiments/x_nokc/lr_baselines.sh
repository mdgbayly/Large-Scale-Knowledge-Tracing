#!/bin/bash
###############################################################################
## Experiment: Evaluate various logistic regression baselines                ##
###############################################################################

# parameters
export PYTHONPATH="."
EXPNAME="lr_baselines"
DATASET="x_nokc"
NPROCESSES=1
NTHREADS=4
SPLITS=5

# Adjusted to represent the features currently supported by x_nokc
IRT="-i"
BESTLR="-i -tcA -tcW"
BESTLR_PLUS="-i -icA_TW -icW_TW -tcA_TW -tcW_TW -user_avg_correct -n_gram"

# Comment out models which we can't support at all
FS=(
    "$IRT"
    "$BESTLR"
    "$BESTLR_PLUS"
)

cmd=""

for (( i=0; i<$SPLITS; i++ )); do
#-----------------------------------------------------------
for f in "${FS[@]}"; do
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $f \n"
done
#-----------------------------------------------------------
done


echo -e $cmd
echo "============================="
echo "Waiting 10s before starting jobs...."
sleep 10s
echo -e $cmd | xargs -n1 -P$NPROCESSES -I{} -- bash -c '{}'
