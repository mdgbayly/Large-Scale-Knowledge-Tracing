import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss, f1_score

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from numpy import argmax, sqrt

def compute_metrics(y_pred, y, protection=1e-8):
    """
    Compute accuracy, AUC score, Negative Log Loss, MSE and F1 score
    """
    # print(y_pred.min(), y_pred.max(), y_pred.shape)
    y_pred = np.array([i if np.isfinite(i) else 0.5 for i in y_pred])
    y_hat = y_pred >= 0.5
    acc = accuracy_score(y, y_hat)
    mse = brier_score_loss(y, y_pred)
    # certain metrics can only be computed if both classes are present
    if (0 in y) and (1 in y):
        roc_auc = roc_auc_score(y, y_pred)
        nll = log_loss(y, y_pred)
        f1 = f1_score(y, y_hat)

        fpr, tpr, roc_thresholds = roc_curve(y, y_pred)
        J = tpr - fpr
        J_best_i = argmax(J)
        roc_opt_threshold = roc_thresholds[J_best_i]
        gmeans = sqrt(tpr * (1 - fpr))
        gmean = gmeans[J_best_i]
        g_best_i = argmax(gmeans)

        lr_precision, lr_recall, pr_thresholds = precision_recall_curve(y, y_pred)
        # convert to f score
        fscore = (2 * lr_precision * lr_recall) / (lr_precision + lr_recall)
        # locate the index of the largest f score
        fscore_best_i = argmax(fscore)
        pr_opt_threshold = pr_thresholds[fscore_best_i]

        cm = confusion_matrix(y, y_hat)
        cm_roc_opt = confusion_matrix(y, y_pred > roc_opt_threshold)
        cm_pr_opt = confusion_matrix(y, y_pred > pr_opt_threshold)
        pr_auc = auc(lr_recall, lr_precision)
    else:
        roc_auc = -1
        nll = -1
        f1 = -1
        fpr = []
        tpr = []
        J_best_i = -1
        roc_opt_threshold = -1
        gmean = -1
        cm = []
        cm_roc_opt = []
        cm_pr_opt = []
        lr_precision = []
        lr_recall = []
        pr_auc = -1
        fscore_best_i = -1
        pr_opt_threshold = -1
    return acc, roc_auc, nll, mse, f1, fpr, tpr, J_best_i, roc_opt_threshold, gmean, pr_auc, cm, cm_roc_opt, cm_pr_opt, lr_precision, lr_recall, fscore_best_i, pr_opt_threshold


class Metrics:
    """
    Keep track of metrics over time in a dictionary.
    """
    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def store(self, new_metrics):
        for key in new_metrics:
            if key in self.metrics:
                self.metrics[key] += new_metrics[key]
                self.counts[key] += 1
            else:
                self.metrics[key] = new_metrics[key]
                self.counts[key] = 1

    def average(self):
        average = {k: v / self.counts[k] for k, v in self.metrics.items()}
        self.metrics, self.counts = {}, {}
        return average
