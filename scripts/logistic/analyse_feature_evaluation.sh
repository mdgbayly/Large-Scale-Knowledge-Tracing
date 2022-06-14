###############################################################################
## Analysis of feature evaluation experiments                                ##
###############################################################################
# Args:                                                                       #
#    dataset (string): dataset name, available datasets:                      #
#      - elemmath_2021                                                        #
#      - ednet_kt3                                                            #
#      - junyi_15                                                             #
#      - junyi_20                                                             #
#      - eedi                                                                 #
#    exp_name (string): name of evaluation experiment:                        #
#      - single_feature_evaluation                                            #
#      - bestlr_feature_evaluation                                            #
#      - lr_baselines                                                         #
#    n_splits (int): number of cross-validation splits                        #
#    csv: Output CSV format                                                   #
#    baseline_acc (float): Baseline accuracy to compare evaluations against   #
#    baseline_auc (float): Baseline AUC to compare evaluations against        #
###############################################################################

export PYTHONPATH="."
DATASET="x_nokc"
EXPNAME="lr_baselines"
NSPLITS=5
BASELINE_ACC="0.843749"
BASELINE_AUC="0.5"


python ./src/analysis/feature_evaluation.py \
    --dataset=$DATASET \
    --exp_name=$EXPNAME \
    --n_splits=$NSPLITS \
    --csv \
    --baseline_acc=$BASELINE_ACC \
    --baseline_auc=$BASELINE_AUC
