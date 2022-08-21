import os
import json
import time
import pickle
import datetime
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from config.constants import ALL_FEATURES, DATASET_PATH, SEED
from src.utils.metrics import compute_metrics
import src.utils.prepare_parser as prepare_parser
from src.utils.data_loader import get_combined_features_and_split


def train_func(X_train, y_train, X_test, y_test, args):
    print("\nPerforming logistic regression:")
    print("----------------------------------------")
    print("Fitting LR model...")
    start = time.perf_counter()
    model = LogisticRegression(solver="liblinear",  # "saga" "lbfgs"
                               max_iter=args.num_iterations,
                               n_jobs=args.num_threads,
                               verbose=1)

    model.fit(X_train, y_train)
    now = time.perf_counter()
    print("Fitted LR model in " + str(round(now - start, 2)) + " seconds")

    print("Evaluating LR model...")
    start = time.perf_counter()
    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1]

    acc_train, auc_train, nll_train, mse_train, f1_train, \
        fpr_train, tpr_train, J_best_i_train, roc_opt_threshold_train, gmean_train, \
        pr_auc_train, cm_train, cm_roc_opt_train, cm_pr_opt_train, precision_train, recall_train, \
        fscore_best_i_train, pr_opt_threshold_train = \
        compute_metrics(y_pred_train, y_train)

    acc_test, auc_test, nll_test, mse_test, f1_test, \
        fpr_test, tpr_test, J_best_i_test, roc_opt_threshold_test, gmean_test, \
        pr_auc_test, cm_test, cm_roc_opt_test, cm_pr_opt_test, precision_test, recall_test, \
        fscore_best_i_test, pr_opt_threshold_test = \
        compute_metrics(y_pred_test, y_test)

    metrics_train = {
        "acc": acc_train,
        "auc": auc_train,
        "pr_auc": pr_auc_train,
        "nll": nll_train,
        "mse": mse_train,
        "rmse": np.sqrt(mse_train),
        "f1": f1_train,
        "gmean": gmean_train,
        "cm": cm_train.tolist(),
        "cm_roc_opt": cm_roc_opt_train.tolist(),
        "cm_pr_opt": cm_pr_opt_train.tolist(),
        "roc_opt_t": roc_opt_threshold_train,
        "pr_opt_t": pr_opt_threshold_train,
        "J_best_i": int(J_best_i_train),
        "fpr_curve": fpr_train.tolist(),
        "tpr_curve": tpr_train.tolist(),
        "fscore_best_i": int(fscore_best_i_train),
        "precision_curve": precision_train.tolist(),
        "recall_curve": recall_train.tolist()
    }

    metrics_test = {
        "acc": acc_test,
        "auc": auc_test,
        "pr_auc": pr_auc_test,
        "nll": nll_test,
        "mse": mse_test,
        "rmse": np.sqrt(mse_test),
        "f1": f1_test,
        "gmean": gmean_test,
        "cm": cm_test.tolist(),
        "cm_roc_opt": cm_roc_opt_test.tolist(),
        "cm_pr_opt": cm_pr_opt_test.tolist(),
        "roc_opt_t": roc_opt_threshold_test,
        "pr_opt_t": pr_opt_threshold_test,
        "J_best_i": int(J_best_i_test),
        "fpr_curve": fpr_test.tolist(),
        "tpr_curve": tpr_test.tolist(),
        "fscore_best_i": int(fscore_best_i_test),
        "precision_curve": precision_test.tolist(),
        "recall_curve": recall_test.tolist()
    }

    now = time.perf_counter()
    print("Evaluated LR model in " + str(round(now - start, 2)) + " seconds")

    print("----------------------------------------")
    print("Completed logistic regression\n")
    return metrics_train, metrics_test, model


def store_results(met_train, met_test, features, spl_id, model, path):
    print("\nStoring results...")
    suf = '_'.join(features)

    res_path = path + "s" + str(SEED) + "-" + str(spl_id) + "_" + suf + ".json"
    res_dict = {
        "time": datetime.datetime.now().isoformat(),
        "features": features,
        "seed": SEED,
        "split_id": spl_id,
        "metrics_test": met_test,
        "metrics_train": met_train
    }
    with open(res_path, "w") as f:
        json.dump(res_dict, f)

    mod_path = path + "/models/" + "s" + str(SEED) + "-" \
        + str(spl_id) + "_" + suf + ".pkl"
    with open(mod_path, "w") as f:
        pickle.dump(model, open(mod_path, 'wb'))
    print("Stored results under: " + res_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LR model.')
    prepare_parser.add_feature_arguments(parser)
    parser.add_argument('--num_threads', type=int, default=1,
                        help='Number of threads for feature preparation.')
    parser.add_argument('--num_iterations', type=int, default=5000)
    parser.add_argument('--exp_name', type=str,
                        help="Experiment name", default="other")

    args = parser.parse_args()
    split_id = args.split_id
    selected_features = [features for features in ALL_FEATURES
                         if vars(args)[features]]
    selected_features.sort()

    print("Selected features: ", selected_features)
    print("Experiment name:", args.exp_name)
    print("Num Threads: " + str(args.num_threads))
    print("Cross-validation split: " + str(split_id))
    print("Iterations: " + str(args.num_iterations))
    print("Dataset name:", args.dataset)
    assert args.dataset in DATASET_PATH, "The specified dataset not supported"

    res_path = "./artifacts/" + args.exp_name + "/"
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    res_path += args.dataset + "/"
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    if not os.path.isdir(res_path + "models/"):
        os.mkdir(res_path + "models/")

    # retrieve combined data and train/test-split
    X, y, split = get_combined_features_and_split(selected_features,
                                                  split_id, args.dataset)
    X_train = X[split["selector_train"]]
    y_train = y[split["selector_train"]]
    X_test = X[split["selector_test"]]
    y_test = y[split["selector_test"]]

    m_train, m_test, lr_model = \
        train_func(X_train, y_train, X_test, y_test, args)

    print(f"\nfeatures = {selected_features}, "
          f"\nacc_train = {m_train['acc']}, acc_test = {m_test['acc']}, "
          f"\nauc_train = {m_train['auc']}, auc_test = {m_test['auc']}, "
          f"\nf1_train = {m_train['f1']}, f1_test = {m_test['f1']}, "
          f"\nrmse_train = {m_train['rmse']}, rmse_test = {m_test['rmse']}, ")

    store_results(m_train, m_test, selected_features,
                  split_id, lr_model, res_path)
    print("\n----------------------------------------")
    print("Completed logistic regression")

