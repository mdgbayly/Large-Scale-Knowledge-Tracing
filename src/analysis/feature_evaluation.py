"""Summarize experimental results and create result file."""
import os
import csv
from pathlib import Path
import json
import argparse
import numpy as np
from config.constants import SEED
from src.utils.misc import set_random_seeds

PREFIX = "s" + str(SEED) + "-"

SQLR = "i_icA_TW_icW_TW_lag_time_cat_n_gram_postcA_postcW_precA_precW" + \
    "_rc_s_scA_TW_scW_TW_sm_t_tcA_TW_tcW_TW_user_avg_correct"

SQLRp = "i_icA_TW_icW_TW_lag_time_cat_n_gram_postcA_postcW_precA_precW_prev_" \
    + "resp_time_cat_rc_s_scA_TW_scW_TW_sm_t_tcA_TW_tcW_TW_user_avg_correct_vw"

SQLRpp = "i_icA_TW_icW_TW_lag_time_cat_n_gram_postcA_postcW_ppe_precA_precW_" \
    + "prev_resp_time_cat_rc_rpfa_F_rpfa_R_s_scA_TW_scW_TW_sm_t_tcA_TW_" \
    + "tcW_TW_user_avg_correct_vw"

EDLR = "i_icA_TW_icW_TW_lag_time_cat_n_gram_partcA_partcW_s_scA_TW" + \
    "_scW_TW_sm_tcA_TW_tcW_TW_user_avg_correct_vw"

EDLRp = "i_icA_TW_icW_TW_lag_time_cat_n_gram_partcA_partcW_" \
    "prev_resp_time_cat_s_scA_TW_scW_TW_sm_tcA_TW_tcW_TW_user_avg_correct_vw"

EDLRpp = "i_icA_TW_icW_TW_lag_time_cat_n_gram_partcA_partcW_ppe_" \
    + "prev_resp_time_cat_rpfa_F_rpfa_R_s_scA_TW_scW_TW_sm_tcA_TW_tcW_TW_" \
    + "user_avg_correct_vs_vw"

EDLRp_nlag = "i_icA_TW_icW_TW_n_gram_partcA_partcW_prev_resp_time_cat_s_" \
    "scA_TW_scW_TW_sm_tcA_TW_tcW_TW_user_avg_correct_vw"

EELR = "bundle_i_icA_TW_icW_TW_n_gram_precA_precW_s_scA_TW_scW_TW_sm" + \
    "_tcA_TW_tcW_TW_tea_user_avg_correct"

EELRpp = "bundle_i_icA_TW_icW_TW_n_gram_ppe_precA_precW_rpfa_F_rpfa_R_s_" + \
    "scA_TW_scW_TW_sm_tcA_TW_tcW_TW_tea_user_avg_correct"

JULR = "hour_i_icA_TW_icW_TW_n_gram_postcA_postcW_precA_precW_rc_s_scA_TW" + \
    "_scW_TW_sm_tcA_TW_tcW_TW_user_avg_correct"

JULRp = "hour_i_icA_TW_icW_TW_lag_time_cat_n_gram_postcA_postcW_precA_" \
    + "precW_prev_resp_time_cat_rc_s_scA_TW_scW_TW_sm_tcA_TW_tcW_TW_" \
    + "user_avg_correct"

JULRpp = "hour_i_icA_TW_icW_TW_lag_time_cat_n_gram_postcA_postcW_ppe_precA_" \
    + "precW_prev_resp_time_cat_rc_rpfa_F_rpfa_R_rt_s_scA_TW_scW_TW_sm_tcA_TW_" \
    + "tcW_TW_user_avg_correct"

LRP = "i_icA_TW_icW_TW_n_gram_s_scA_TW_scW_TW_tcA_TW_tcW_TW_user_avg_correct"

TWC = "TW combined counts"
EXP_NAMES = {
    # Single feature experiments
    "i": "item ID",
    "s": "skill ID",
    "tcA_tcW": "total counts",
    "scA_scW": "skill counts",
    "icA_icW": "item counts",
    "icA_icW_scA_scW_tcA_tcW": "combined counts",
    "icA_icW_tcA_tcW": "non-skill combined counts",
    "tcA_TW_tcW_TW": "TW total counts",
    "scA_TW_scW_TW": "TW skill counts",
    "icA_TW_icW_TW": "TW item counts",
    "icA_TW_icW_TW_scA_TW_scW_TW_tcA_TW_tcW_TW": "TW combined counts",
    "icA_TW_icW_TW_tcA_TW_tcW_TW": "non-skill TW combined counts",
    "resp_time_cat": "response time",
    "prev_resp_time_cat": "previous response time",
    "lag_time_cat": "lag time",
    "prev_lag_time_cat": "previous lag time",
    "month": "month OH",
    "week": "week OH",
    "day": "day OH",
    "hour": "hour OH",
    "weekend": "weekend OH",
    "part_of_day": "part of day OH",
    "at": "platform",
    "sm": "study module",
    "tea": "teacher",
    "sch": "school",
    "c": "course",
    "t": "topic",
    "d": "difficulty",
    "bundle": "bundle",
    "part": "part id",
    "pre": "prereq OH",
    "precA_precW": "prereq Count",
    "post": "postreq OH",
    "postcA_postcW": "postreq Count",
    "partcA_partcW": "part count",
    "smA_smW": "study module count",
    "vw": "count videos watched",
    "vs": "count videos skipped",
    "vt": "time videos watched",
    "rc": "count readings",
    "rt": "reading time",
    "user_avg_correct": "user average correct",
    "n_gram": "response pattern",
    "age": "age",
    "gender": "gender",
    "ss": "social support",
    "ones": "one vector",
    "rpfa_R": "RPFA Ratio",
    "rpfa_F": "RPFA Failure",
    "rpfa_F_rpfa_R": "RPFA Combined",
    "ppe": "PPE Count",
    # BestLR experiments
    "i_s_scA_scW_tcA_tcA_TW_tcW_tcW_TW": "TW total counts",
    "i_tcA_tcA_TW_tcW_tcW_TW": "TW total counts",
    "i_s_scA_scA_TW_scW_scW_TW_tcA_tcW": "TW skill counts",
    "i_icA_TW_icW_TW_s_scA_scW_tcA_tcW": "TW item counts",
    "i_icA_TW_icW_TW_tcA_tcW": "TW item counts",
    "i_icA_TW_icW_TW_s_scA_scA_TW_scW_scW_TW_tcA_tcA_TW_tcW_tcW_TW": TWC,
    "i_icA_TW_icW_TW_tcA_tcA_TW_tcW_tcW_TW": TWC,
    "i_resp_time_cat_s_scA_scW_tcA_tcW": "response time",
    "i_lag_time_cat_s_scA_scW_tcA_tcW": "lag time",
    "i_prev_resp_time_cat_s_scA_scW_tcA_tcW": "prior response time",
    "i_prev_lag_time_cat_s_scA_scW_tcA_tcW": "prior lag time",
    "i_month_s_scA_scW_tcA_tcW": "month OH",
    "i_month_tcA_tcW": "month OH",
    "i_s_scA_scW_tcA_tcW_week": "week OH",
    "i_tcA_tcW_week": "week OH",
    "day_i_s_scA_scW_tcA_tcW": "day OH",
    "day_i_tcA_tcW": "day OH",
    "hour_i_s_scA_scW_tcA_tcW": "hour OH",
    "hour_i_tcA_tcW": "hour OH",
    "i_s_scA_scW_tcA_tcW_weekend": "weekend OH",
    "i_tcA_tcW_weekend": "weekend OH",
    "i_part_of_day_s_scA_scW_tcA_tcW": "part of day OH",
    "i_part_of_day_tcA_tcW": "part of day OH",
    "at_i_s_scA_scW_tcA_tcW": "platform",
    "i_s_scA_scW_sm_tcA_tcW": "study module",
    "i_s_scA_scW_tcA_tcW_tea": "teacher",
    "i_s_scA_scW_sch_tcA_tcW": "school",
    "c_i_s_scA_scW_tcA_tcW": "course",
    "c_i_tcA_tcW": "course",
    "i_s_scA_scW_t_tcA_tcW": "topic",
    "d_i_s_scA_scW_tcA_tcW": "difficulty",
    "bundle_i_s_scA_scW_tcA_tcW": "bundle",
    "bundle_i_tcA_tcW": "bundle",
    "i_part_s_scA_scW_tcA_tcW": "part id",
    "i_pre_s_scA_scW_tcA_tcW": "prereq OH",
    "i_precA_precW_s_scA_scW_tcA_tcW": "prereq Count",
    "i_post_s_scA_scW_tcA_tcW": "postreq OH",
    "i_postcA_postcW_s_scA_scW_tcA_tcW": "postreq Count",
    "i_partcA_partcW_s_scA_scW_tcA_tcW": "part count",
    "i_s_scA_scW_smA_smW_tcA_tcW": "study module count",
    "i_s_scA_scW_tcA_tcW_vw": "count videos watched",
    "i_s_scA_scW_tcA_tcW_vs": "count videos skipped",
    "i_s_scA_scW_tcA_tcW_vt": "time videos watched",
    "i_rc_s_scA_scW_tcA_tcW": "count readings",
    "i_rt_s_scA_scW_tcA_tcW": "reading time",
    "i_s_scA_scW_tcA_tcW_user_avg_correct": "user average correct",
    "i_tcA_tcW_user_avg_correct": "user average correct",
    "i_n_gram_s_scA_scW_tcA_tcW": "response pattern",
    "i_n_gram_tcA_tcW": "response pattern",
    "age_i_s_scA_scW_tcA_tcW": "age",
    "gender_i_s_scA_scW_tcA_tcW": "gender",
    "i_s_scA_scW_ss_tcA_tcW": "social support",
    "i_rpfa_R_s_scA_scW_tcA_tcW": "RPFA Ratio",
    "i_rpfa_F_s_scA_scW_tcA_tcW": "RPFA Failure",
    "i_rpfa_F_rpfa_R_s_scA_scW_tcA_tcW": "RPFA Combined",
    "i_ppe_s_scA_scW_tcA_tcW": "PPE Count",
    # Baselines
    "s_scA_scW": "PFA",
    "i_icA_icW_s_scA_scW_tcA_tcW": "item counts",
    "i_icA_icW_tcA_tcW": "item counts",
    "rpfa_F_rpfa_R_s": "RPFA",
    "ppe_s": "PPE",
    "i_s_scA_TW_scW_TW": "DAS3H",
    "i_s_scA_scW_tcA_tcW": "Best-LR",
    "i_tcA_tcW": "non-skill Best-LR",
    "i_icA_icW_n_gram_s_scA_scW_sm_tcA_tcW_user_avg_correct": "BestLR+",
    "i_icA_TW_icW_TW_n_gram_tcA_TW_tcW_TW_user_avg_correct": "non-skill BestLR+",
    "i_icA_TW_icW_TW_n_gram_rpfa_F_rpfa_R_s_scA_TW_scW_TW_tcA_TW_tcW_TW_user_avg_correct": "LRP+RPFA",
    "i_icA_TW_icW_TW_n_gram_ppe_s_scA_TW_scW_TW_tcA_TW_tcW_TW_user_avg_correct": "LRP+PPE",
    "i_icA_TW_icW_TW_n_gram_ppe_rpfa_F_rpfa_R_s_scA_TW_scW_TW_tcA_TW_tcW_TW_user_avg_correct": "LRP+PPE+RPFA",
    # AugmentedLR
    SQLR: "SQLR",
    SQLRp: "SQLR+",
    SQLRpp: "SQLR+New",
    EDLR: "EDLR",
    EELR: "EELR",
    EELRpp: "EELR+New",
    JULR: "JULR",
    JULRp: "JULR+",
    JULRpp: "JULR+New",
    LRP: "LRP",
    EDLRp: "EDLR+",
    EDLRp_nlag: "EDLR+ no lag",
    EDLRpp: "EDLR+New"
}


def evaluate_artifacts(file_paths):
    acc_vals, auc_vals = [], []
    for f in file_paths:
        with open(f) as json_file:
            data = json.load(json_file)
            acc_vals.append(data["metrics_test"]["acc"])
            auc_vals.append(data["metrics_test"]["auc"])

    acc_avg = np.round(np.mean(acc_vals), decimals=6)
    acc_std = np.round(np.std(acc_vals), decimals=6)
    auc_avg = np.round(np.mean(auc_vals), decimals=6)
    auc_std = np.round(np.std(auc_vals), decimals=6)

    return acc_avg, acc_std, auc_avg, auc_std


def write_text(dataset, files, n_splits, res_path, out_path):
    output = ""
    for f in files:
        fps = [res_path + str(i) + "_" + f + ".json" for i in range(n_splits)]
        acc_avg, acc_std, auc_avg, auc_std = evaluate_artifacts(fps)
        out = EXP_NAMES[f] + " ACC / AUC:"
        out += (" " * (35 - len(out)))
        out += "\\avgvar{" + str(acc_avg) + "}{" + str(acc_std) + "} &"
        out += " \\avgvar{" + str(auc_avg) + "}{" + str(auc_std) + "} \n\n"
        print(out)
        output += out

    out_file = open(out_path + dataset + ".txt", "w")
    out_file.write(output)
    out_file.close()


def write_csv(dataset, files, n_splits, res_path, out_path, baseline_acc, baseline_auc):
    with open(out_path + dataset + ".csv", "w", encoding='UTF8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['expname', 'acc_avg', 'acc_diff', 'acc_std', 'auc_avg', 'auc_diff', 'auc_std', 'sig'])
        writer.writerow(['Baseline', baseline_acc, '-', '-', baseline_auc, '-', '-', '-'])

        for f in files:
            fps = [res_path + str(i) + "_" + f + ".json" for i in range(n_splits)]
            acc_avg, acc_std, auc_avg, auc_std = evaluate_artifacts(fps)
            writer.writerow([EXP_NAMES[f], acc_avg, acc_avg - baseline_acc, acc_std, auc_avg, auc_avg - baseline_auc,
                             auc_std, (auc_avg - baseline_auc) > 0.0005])

            out = EXP_NAMES[f] + " ACC / AUC:"
            out += (" " * (35 - len(out)))
            out += "\\avgvar{" + str(acc_avg) + "}{" + str(acc_std) + "} &"
            out += " \\avgvar{" + str(auc_avg) + "}{" + str(auc_std) + "} \n\n"
            print(out)


def analyse_feature_evaluation(dataset, exp_name, n_splits, out_path, output_csv, baseline_acc, baseline_auc):
    res_path = "artifacts/" + exp_name + "/" + dataset + "/"
    files = os.listdir(res_path)
    files = set([f[7:-5] for f in files if f != "models"])
    res_path += PREFIX

    # Check if we have results for all seeds
    for f in files:
        for i in range(n_splits):
            f_path = res_path + str(i) + "_" + f + ".json"
            if not os.path.exists(f_path):
                print('Missing: ' + f_path)
                return
            # assert os.path.exists(f_path), "Missing file: " + f_path
    if output_csv:
        write_csv(dataset, files, n_splits, res_path, out_path, baseline_acc, baseline_auc)
    else:
        write_text(dataset, files, n_splits, res_path, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyse feature evaluation.')
    parser.add_argument('--dataset', type=str, help="Relevant dataset.")
    parser.add_argument('--exp_name', type=str, help="Experiment name.")
    parser.add_argument('--n_splits', type=int, help="Number of splits.",
                        default=5)
    parser.add_argument('--csv', action='store_true',
                        help='Output in CSV format')

    parser.add_argument('--baseline_acc', type=float, help='Baseline Accuracy for comparison')
    parser.add_argument('--baseline_auc', type=float, help='Baseline AUC for comparison')

    args = parser.parse_args()
    set_random_seeds(SEED)

    dataset = args.dataset
    exp_name = args.exp_name
    n_splits = args.n_splits
    output_csv = args.csv
    baseline_auc = args.baseline_auc
    baseline_acc = args.baseline_acc

    print("\nDataset: " + dataset)
    print("Experiment name: " + exp_name)
    print("Number of splits: " + str(n_splits) + "\n")

    # prepare result folder
    res_path = "./results/" + args.exp_name + "/"
    # if not os.path.isdir(res_path):
        # os.mkdir(res_path)
    Path(res_path).mkdir(parents=True, exist_ok=True)
    analyse_feature_evaluation(dataset, exp_name, n_splits, res_path, output_csv, baseline_acc, baseline_auc)

    print("----------------------------------------")
    print("Completed feature evaluation analysis\n")
