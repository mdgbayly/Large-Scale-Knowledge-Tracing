###############################################################################
## Train time specialized logistic regresssion models                        ##
###############################################################################
# Args:                                                                       #
#    dataset (string): dataset name, available datasets:                      #
#      - elemmath_2021                                                        #
#      - ednet_kt3                                                            #
#      - junyi_15                                                             #
#      - junyi_20                                                             #
#      - eedi                                                                 #
#    split_id (int): train/test split to use                                  #
#    num_threads (int): number of threads available for feature preparation   #
#    generalist (string): "" to train time specialized, "1" for generalist    #
#                         generalist assumes model has been computed before   #
#                                                                             #
# One-hot features:                                                           #
#    -u: user ID one-hot                                                      #
#    -i: item/question ID one-hot                                             #
#    -s: skill ID one-hot                                                     #
#    -sch: school one-hot                                                     #
#    -tea: teacher/group one-hot                                              #
#    -sm: study module one-hot                                                #
#    -c: course id one-hot                                                    #
#    -d: difficulty one-hot                                                   #
#    -at: app-type one-hot                                                    #
#    -t: topic one-hot                                                        #
#    -bundle: bundle/quiz one-hot                                             #
#    -part: TOIEC part one-hot                                                #
#    -age: user age one-hot                                                   #
#    -gender: user gender one-hot                                             #
#    -ss: user social support one-hot                                         #
#                                                                             #
# User history features:                                                      #
#    -tcA: total count of previous attemts                                    #
#    -tcW: total count of previous wins                                       #
#    -scA: skill count of previous attemts                                    #
#    -scW: skill count of previous wins                                       #
#    -icA: item/question count of previous attemts                            #
#    -icW: item/question count of previous wins                               #
#    -tcA_TW: total count of previous attemts by time window                  #
#    -tcW_TW: total count of previous wins by time window                     #
#    -scA_TW: skill count of previous attemts by time window                  #
#    -scW_TW: skill count of previous wins by time window                     #
#    -icA_TW: item/question count of previous attemts by time window          #
#    -icW_TW: item/question count of previous wins by time window             #
#    -smA: count previous attempts in this study module                       #
#    -smW: count previous wins in this study module                           #
#    -partcA: count previous attempts in this part                            #
#    -partcW: count previous wins in this part                                #
#                                                                             #
# Graph features:                                                             #
#    -pre: pre-req skill one-hot                                              #
#    -post: pre-req skill one-hot                                             #
#    -precA: pre-req skill count of previous attemts                          #
#    -precW: pre-req skill count of previous wins                             #
#    -postcA: post-req skill count of previous attemts                        #
#    -postcW: post-req skill count of previous wins                           #
#                                                                             #
# Time features:                                                              #
#    -resp_time: user response time in seconds                                #
#    -resp_time_cat: response time phi and categories                         #
#    -prev_resp_time_cat: previous response time phi and categories           #
#    -lag_time: user lag time in minutes                                      #
#    -lag_time_cat: lag time phi and categories                               #
#    -prev_lag_time_cat: previous lag time phi and categories                 #
#                                                                             #
# RPFA features:                                                              #
#    -rpfa_F: recency-weighted failure count                                  #
#    -rpfa_R: recency-weighted proportion of past successes                   #
#                                                                             #
# PPE feature  :                                                              #
#    -ppe: spacing time weighted attempt count                                #
#                                                                             #
# Datetime features:                                                          #
#    -month: month one-hot                                                    #
#    -week: week one-hot                                                      #
#    -day: day one-hot                                                        #
#    -hour: hour one-hot                                                      #
#    -weekend: weekend one-hot                                                #
#    -part_of_day: part of day one-hot                                        #
#                                                                             #
# Reading features:                                                           #
#    -rc: count readings on skill and total level (hints for Junyi15)         #
#    -rt: total reading time on skill and total level (hints for Junyi15)     #
#                                                                             #
# Video features:                                                             #
#    -vw: count watched videos                                                #
#    -vs: count skipped videos                                                #
#    -vt: watching time on skill and total level                              #
#                                                                             #
# Misc features:                                                              #
#    -user_avg_correct: average user correctness over time                    #
#    -n_gram: correctness patterns in sequence                                #
#                                                                             #
###############################################################################
export PYTHONPATH="."
DATASET="elemmath_2021"
NTHREADS=23

# "" to train time specialized, "1" for generalist
# NOTE: generalist assumes model has been computed before
GENERALIST=""

# Select features using the flags above
features="-i -s -icA -icW -scA -scW -n_gram -user_avg_correct"

# Item Response Theory (IRT):
# features="-i"

# PFA:
# features="-s -scA -scW"

# Best-LR:
# features="-i -s -scA -scW -tcA -tcW"

# Best-LR+
# features="-i -s \
#    -icA_TW -icW_TW -scA_TW -scW_TW -tcA_TW -tcW_TW \
#    -user_avg_correct -n_gram -rpfa_F -rpfa_R -ppe"

python ./src/training/compute_time_specialized.py \
    --dataset=$DATASET \
    --num_threads=$NTHREADS \
    --split_id=5 \
    --generalist=$GENERALIST $features
