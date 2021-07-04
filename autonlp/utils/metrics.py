import numpy as np
from sklearn.metrics import *
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)

#####################
# Metrics
#####################


def calcul_metric_binary(y_true_, y_pred, thr_1=0.5, print_score=True):
    """ Compute for binary variable (y_true_, y_pred) roc auc, accuracy, recall, precision and f1
    Args:
        y_true_ (Dataframe or array)
        y_pred (array)
        thr_1 (float) threshold for label 1
        print_score (Boolean)
    Returns:
        acc, f1, recall, precision, roc_auc (float)
    """
    if isinstance(y_true_, pd.DataFrame):  # pass y_true_ to array type
        y_true = y_true_.values.copy()
    else:
        y_true = y_true_.copy()

    report = classification_report(y_true.reshape(-1), np.where(y_pred > thr_1, 1, 0).reshape(-1), digits = 4, output_dict = True)
    acc = np.round(report['accuracy'], 4)
    f1 = np.round(report['1']['f1-score'], 4)
    recall = np.round(report['1']['recall'], 4)
    precision = np.round(report['1']['precision'], 4)
    # roc_auc = np.round(roc_auc_score(y_true.values, np.where(y_pred<0.5,0,1)),4)
    fp_rate, tp_rate, thresholds = roc_curve(y_true.reshape(-1), y_pred.reshape(-1))
    roc_auc = np.round(auc(fp_rate, tp_rate), 4)

    if print_score:
        logger.info('\nScores :')
        logger.info('roc_auc = {}'.format(roc_auc))
        logger.info('precision 1 = {}'.format(precision))
        logger.info('recall 1 = {}'.format(recall))
        logger.info('f1 score 1 = {}'.format(f1))
        logger.info('\n')
        logger.info(classification_report(y_true.reshape(-1), np.where(y_pred > thr_1, 1, 0).reshape(-1), digits=3))

    return acc, f1, recall, precision, roc_auc


def roc(y_true, y_proba):
    """ Compute Receiver operating characteristic for binary proba variable (y_true, y_proba) """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return fpr, tpr


def calcul_metric_classification(y_true, y_pred, average="weighted", print_score=True):
    """ Compute for multi-class variable (y_true, y_pred) accuracy, recall_weighted, precision_weighted and f1_weighted
    Args:
        y_true (Dataframe or array)
        y_pred (Dataframe or array)
        average (str) 'micro', 'macro' or 'weighted'
        print_score (Boolean)
    Returns:
        acc, f1, recall, precision (float)
    """
    acc = np.round(accuracy_score(y_true, y_pred), 4)
    f1 = np.round(f1_score(y_true, y_pred, average=average), 4)
    recall = np.round(recall_score(y_true, y_pred, average=average), 4)
    precision = np.round(precision_score(y_true, y_pred, average=average), 4)

    if print_score:
        logger.info('\nScores :')
        logger.info('precision {} = {}'.format(average, precision))
        logger.info('recall {} = {}'.format(average, recall))
        logger.info('f1 score {} = {}'.format(average, f1))

    return acc, f1, recall, precision


def build_df_confusion_matrix(y_true, y_pred, reverse_map_label=None):
    """ Compute for binary or multi-class variable (y_true, y_pred) confusion matrix
    Args:
        y_true (Dataframe or array)
        y_pred (Dataframe or array)
        reverse_map_label (dict) reverse map of labels to get string label names
    Returns:
        df_cm (Dataframe) confusion matrix in format dataframe
    """
    cm = confusion_matrix(y_true, y_pred)
    if reverse_map_label is not None:
        labels = [reverse_map_label[i] for i in np.unique(list(reverse_map_label.keys()))]
        df_cm = pd.DataFrame(cm, labels, labels)
    else:
        df_cm = pd.DataFrame(cm)
    return df_cm


def calcul_metric_regression(y_true, y_pred, print_score=True):
    """ Compute for regression variable (y_true, y_pred) mse, rmse, explained variance and r2
    Args:
        y_true (Dataframe or array)
        y_pred (Dataframe or array)
        print_score (Boolean)
    Returns:
        mse, rmse, expl_var, r2 (float)
    """
    expl_var = np.round(explained_variance_score(y_true, y_pred), 4)
    r2 = np.round(r2_score(y_true, y_pred), 4)
    mse = np.round(mean_squared_error(y_true, y_pred), 4)
    rmse = np.round(mean_squared_error(y_true, y_pred, squared=False), 4)

    if print_score:
        logger.info('\nScores :')
        logger.info('explained variance = {}'.format(expl_var))
        logger.info('r2 = {}'.format(r2))
        logger.info('mse = {}'.format(mse))
        logger.info('rmse = {}'.format(rmse))

    return mse, rmse, expl_var, r2