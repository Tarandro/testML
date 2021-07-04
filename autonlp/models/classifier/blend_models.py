import numpy as np
from ...utils.metrics import roc, calcul_metric_binary, calcul_metric_classification, calcul_metric_regression

from ...utils.logging import get_logger

logger = get_logger(__name__)


class BlendModel:
    """ Average all model predictions """

    def __init__(self, objective, average_scoring="weighted", map_label={}):
        """
        Args:
            objective (str) : 'binary' or 'multi-class' or 'regression' / specify target objective
            map_label (dict) : use label map if labels are not numerics
        """
        self.objective = objective
        self.average_scoring = average_scoring
        self.map_label = map_label
        self.name_model = 'BlendModel'
        self.info_scores = {}

    def validation(self, models, x_train, y_train, x_val=None, y_val=None, thr_1=0.5):
        """ Average validation predictions of all models (autonlp.models[name_model].info_scores['oof_val'])
        Args:
            models (dict) get from autonlp.models
            x_train (List or Dict or Dataframe)
            y_train (Dataframe)
            x_val (List or Dict or Dataframe)
            y_val (Dataframe)
            thr_1 (float) threshold for label 1
        """

        oof_val = None
        self.y_shape1 = y_train.shape[1]
        fold_id = models[list(models.keys())[0]].info_scores['fold_id']

        nb_model = 0
        for name_model in models.keys():
            if name_model not in ['BlendModel']:
                if oof_val is None:
                    oof_val = models[name_model].info_scores['oof_val']
                else:
                    oof_val = oof_val + models[name_model].info_scores['oof_val']
                nb_model += 1

        oof_val = oof_val / nb_model

        if 'regression' not in self.objective:
            if self.y_shape1 == 1 and 'binary' not in self.objective:
                prediction_oof_val = np.argmax(oof_val, axis=1).reshape(-1)
            else:
                prediction_oof_val = np.where(oof_val > 0.5, 1, 0)
        else:
            prediction_oof_val = oof_val

        if x_val is None:
            # cross-validation
            y_true_sample = y_train.values[np.where(fold_id >= 0)[0]].copy()
        else:
            y_true_sample = y_val.values[np.where(fold_id >= 0)[0]].copy()

        # store information from validation in self.info_scores :
        if 'binary' in self.objective:
            self.info_scores['accuracy_val'], self.info_scores['f1_val'], self.info_scores['recall_val'], \
            self.info_scores['precision_val'], self.info_scores['roc_auc_val'] = calcul_metric_binary(y_true_sample, prediction_oof_val, thr_1)
            self.info_scores['fpr'], self.info_scores['tpr'] = roc(y_true_sample, oof_val)
        elif 'multi-class' in self.objective:
            self.info_scores['accuracy_val'], self.info_scores['f1_' + self.average_scoring + '_val'], self.info_scores[
                'recall_' + self.average_scoring + '_val'], self.info_scores[
                'precision_' + self.average_scoring + '_val'] = calcul_metric_classification(y_true_sample, prediction_oof_val, self.average_scoring)
        elif 'regression' in self.objective:
            self.info_scores['mse_val'], self.info_scores['rmse_val'], self.info_scores['explained_variance_val'], \
            self.info_scores['r2_val'] = calcul_metric_regression(y_true_sample, prediction_oof_val)

        self.info_scores['fold_id'], self.info_scores['oof_val'] = fold_id, oof_val

    def prediction(self, models, x_test, y_test=None, thr_1=0.5):
        """ Average test predictions of all models (models[name_model].info_scores['prediction'])
        Args:
            models (dict) get from autonlp.models
            x_test (List or Dict or Dataframe)
            y_test (Dataframe)
            thr_1 (float) threshold for label 1
        """

        prediction = None

        nb_model = 0
        for name_model in models.keys():
            if name_model not in ['BlendModel']:
                if prediction is None:
                    prediction = models[name_model].info_scores['prediction']
                else:
                    prediction = prediction + models[name_model].info_scores['prediction']
                nb_model += 1

        prediction = prediction / nb_model

        if y_test is not None:

            if 'regression' not in self.objective:
                if y_test.shape[1] == 1 and 'binary' not in self.objective:
                    prediction_test = np.argmax(prediction, axis=1).reshape(-1)
                else:
                    prediction_test = np.where(prediction > 0.5, 1, 0)
            else:
                prediction_test = prediction

            if self.map_label != {}:
                if y_test[y_test.columns[0]].iloc[0] in self.map_label.keys():
                    y_test[y_test.columns[0]] = y_test[y_test.columns[0]].map(self.map_label)
                    if y_test[y_test.columns[0]].isnull().sum() > 0:
                        logger.error("Unknown label name during map of test labels")

            # store information from prediction in self.info_scores :
            if 'binary' in self.objective:
                self.info_scores['accuracy_test'], self.info_scores['f1_test'], self.info_scores['recall_test'], \
                self.info_scores['precision_test'], self.info_scores['roc_auc_test'] = calcul_metric_binary(
                                y_test, prediction_test)
            elif 'multi-class' in self.objective:
                self.info_scores['accuracy_test'], self.info_scores['f1_' + self.average_scoring + '_test'], \
                self.info_scores['recall_' + self.average_scoring + '_test'], \
                self.info_scores['precision_' + self.average_scoring + '_test'] = calcul_metric_classification(y_test, prediction_test, self.average_scoring)
            elif 'regression' in self.objective:
                self.info_scores['mse_test'], self.info_scores['rmse_test'], self.info_scores[
                    'explained_variance_test'], self.info_scores['r2_test'] = calcul_metric_regression(y_test, prediction_test)

        self.info_scores['prediction'] = prediction