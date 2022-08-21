#!/bin/bash
###############################################################################
## Experiment: Evaluate augmented logistic regression                        ##
###############################################################################

# parameters
export PYTHONPATH="."
EXPNAME="lr_baselines"
DATASET="x_kc_014"
NPROCESSES=1
NTHREADS=4
SPLITS=5

cmd=""

#XKLR="-i -s \
#      -icA -icW -scA -scW -tcA -tcW \
#      -icA_TW -icW_TW -scA_TW -scW_TW -tcA_TW -tcW_TW \
#      -rpfa_F -rpfa_R -ppe"

XKLR="-i \
      -tcA -tcW \
      -icA_TW -icW_TW -tcA_TW -tcW_TW"

for (( i=0; i<$SPLITS; i++ )); do
#-----------------------------------------------------------
cmd+="python ./src/training/compute_lr.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=$i \
    --exp_name=$EXPNAME $XKLR \n"
#-----------------------------------------------------------
done


echo -e $cmd
echo "============================="
echo "Waiting 10s before starting jobs...."
sleep 10s
echo -e $cmd | xargs -n1 -P$NPROCESSES -I{} -- bash -c '{}'
