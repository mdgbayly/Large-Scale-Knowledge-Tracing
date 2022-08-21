#!/bin/bash
###############################################################################
## Experiment: Evaluate rich logistic regression                             ##
###############################################################################

# parameters
export PYTHONPATH="."
EXPNAME="lr_baselines"
DATASET="x_nokc_hard"
NPROCESSES=1
NTHREADS=4
SPLITS=5

cmd=""

XNLRp="-i
      -icA -icW -tcA -tcW \
      -icA_TW -icW_TW -tcA_TW -tcW_TW \
      -n_gram \
      -bundle -user_avg_correct"

for (( i=0; i<$SPLITS; i++ )); do
#-----------------------------------------------------------
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $XNLRp \n"
#-----------------------------------------------------------
done


echo -e $cmd
echo "============================="
echo "Waiting 10s before starting jobs...."
sleep 10s
echo -e $cmd | xargs -n1 -P$NPROCESSES -I{} -- bash -c '{}'
