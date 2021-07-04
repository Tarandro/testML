import numpy as np
import pandas as pd
import os
from glob import glob
from .utils.metrics import calcul_metric_binary, calcul_metric_classification, calcul_metric_regression
from joblib import load
from tensorflow.keras import backend as K
from .flags import load_yaml

from .utils.logging import get_logger

logger = get_logger(__name__)


class Prediction:
    """ Class Prediction """

    def __init__(self, objective, name_embedding=None, name_model_full=None, outdir="./", name_logs='last_logs',
                 is_NN=False, class_weight=None, apply_mlflow=False, experiment_name="Experiment", apply_logs=True,
                 apply_app=False):
        """
        Args:
            objective (str) : 'binary' or 'multi-class' or 'regression'
            name_embedding (str) : embedding name method
            name_model_full (str) : full name of model (embedding+classifier+tag)
            outdir (str) path where model are saved
            name_logs ('best_logs' or 'last_logs')
            is_NN (Boolean) True if the model is a Neural Network
            class_weight (None or 'balanced')
            apply_mlflow (Boolean)
        """
        self.objective = objective
        self.name_embedding = name_embedding
        self.name_model_full = name_model_full
        self.outdir = outdir
        self.name_logs = name_logs
        self.is_NN = is_NN
        self.class_weight = class_weight
        self.apply_mlflow = apply_mlflow
        self.path_mlflow = "../mlruns"
        self.experiment_name = experiment_name
        self.apply_logs = apply_logs
        self.apply_app = apply_app

    def fit(self, model, x, y=None, loaded_models=None):
        """ Fit model on x with different model folds
            Models are loaded from the outdir/name_logs/name_embedding/name_model_full directory
            Average all folds prediction of a name_model to get final prediction
        Args:
            model (self.model) function of the model architecture instantiated : self.model() and not self.model
            x (List or dict or DataFrame)
            y (Dataframe)
        """

        # get path of models :
        if self.apply_logs:
            if self.name_embedding.lower() == "transformer":
                outdir_model = os.path.join(self.outdir, self.name_logs, self.name_embedding, self.name_model_full)
            else:
                outdir_model = os.path.join(self.outdir, self.name_logs, self.name_embedding,
                                            self.name_model_full.split('+')[1])
            # get path of model folds :
            try:
                models_or_paths = glob(outdir_model + '/fold*')
            except FileNotFoundError:
                logger.critical("Didn't find checkpoint model for {} in '{}'".format(self.name_model_full, outdir_model))

        elif self.apply_mlflow:
            import mlflow
            experiment_id = None
            for dir in os.listdir(self.path_mlflow):
                if os.path.exists(os.path.join(self.path_mlflow, dir, "meta.yaml")):
                    meta_flags = load_yaml(os.path.join(self.path_mlflow, dir, "meta.yaml"))
                    if meta_flags['name'] == self.experiment_name:
                        experiment_id = meta_flags['experiment_id']

        elif self.apply_app and loaded_models is None:
            outdir_model = self.outdir
            # get path of model folds :
            try:
                models_or_paths = glob(outdir_model + '/' + '*.joblib') + glob(outdir_model + '/' + '*.hdf5')
            except FileNotFoundError:
                logger.critical("Didn't find checkpoint model for {} in '{}'".format(self.name_model_full, outdir_model))

        elif self.apply_app and loaded_models is not None:
            if isinstance(loaded_models, list):
                models_or_paths = loaded_models
            else:
                models_or_paths = [loaded_models]

        if isinstance(x, pd.DataFrame):
            x_test = x.values
        else:
            x_test = x

        first_fold = True
        nb_model = 0

        if self.is_NN:
            # validation for neural network models :
            if self.apply_logs or self.apply_app:
                for i, model_or_path in enumerate(models_or_paths):
                    if isinstance(model_or_path, str):
                        K.clear_session()
                        model.load_weights(model_or_path)
                    else:
                        model = model_or_path

                    pred = model.predict(x_test)

                    if 'binary' in self.objective or ('regression' in self.objective and pred.shape[1] == 1):
                        pred = pred.reshape(-1)
                    if first_fold:
                        self.prediction = pred
                        first_fold = False
                    else:
                        self.prediction += pred
                    nb_model += 1

            else:
                if experiment_id is None:
                    logger.warning("The MLflow Tracking with experiment name '{}' is not provided in {}.".format(
                        self.experiment_name, self.path_mlflow))
                else:
                    path_mlflow_experiment_id = os.path.join(self.path_mlflow, experiment_id)
                    for i, dir_run in enumerate(os.listdir(path_mlflow_experiment_id)):
                        if os.path.exists(os.path.join(path_mlflow_experiment_id, dir_run, "tags")):
                            file1 = open(os.path.join(path_mlflow_experiment_id, dir_run, "tags", "mlflow.runName"), 'r')
                            Lines = file1.readlines()
                            if Lines[0] == self.name_model_full:
                                K.clear_session()
                                #model.load_weights(path)

                                logged_model = os.path.join(path_mlflow_experiment_id, dir_run, 'artifacts', 'model')
                                logger.info("Model from directory {}".format(logged_model))

                                # Load model as a PyFuncModel.
                                loaded_model = mlflow.pyfunc.load_model(logged_model)
                                pred = loaded_model.predict(x_test)

                                if 'binary' in self.objective or ('regression' in self.objective and pred.shape[1] == 1):
                                    pred = pred.reshape(-1)
                                if first_fold:
                                    self.prediction = pred
                                    first_fold = False
                                else:
                                    self.prediction += pred
                                nb_model += 1
        else:
            # validation for sklearn models, catboost, xgboost and lightgbm :
            if self.apply_logs or self.apply_app:
                for i, model_or_path in enumerate(models_or_paths):
                    if isinstance(model_or_path, str):
                        model = load(model_or_path)
                    else:
                        model = model_or_path
                    if 'regression' in self.objective:
                        pred = model.predict(x_test)
                    else:
                        pred = model.predict_proba(x_test)

                    if 'binary' in self.objective:
                        pred = pred[:, 1].reshape(x_test.shape[0], )
                    if first_fold:
                        self.prediction = pred
                        first_fold = False
                    else:
                        self.prediction += pred
                    nb_model += 1

            else:
                if experiment_id is None:
                    logger.warning("The MLflow Tracking with experiment name '{}' is not provided in {}.".format(
                        self.experiment_name, self.path_mlflow))
                else:
                    path_mlflow_experiment_id = os.path.join(self.path_mlflow, experiment_id)
                    for i, dir_run in enumerate(os.listdir(path_mlflow_experiment_id)):
                        if os.path.exists(os.path.join(path_mlflow_experiment_id, dir_run, "tags")):
                            file1 = open(os.path.join(path_mlflow_experiment_id, dir_run, "tags", "mlflow.runName"), 'r')
                            Lines = file1.readlines()
                            if Lines[0] == self.name_model_full:
                                K.clear_session()
                                #model.load_weights(path)

                                logged_model = os.path.join(path_mlflow_experiment_id, dir_run, 'artifacts', self.name_model_full)
                                logger.info("Model from directory {}".format(logged_model))

                                # Load model as a PyFuncModel.
                                # loaded_model = mlflow.pyfunc.load_model(logged_model)
                                loaded_model = mlflow.sklearn.load_model(logged_model)

                                if 'regression' in self.objective:
                                    pred = loaded_model.predict(x_test)
                                else:
                                    pred = loaded_model.predict_proba(x_test)

                                if 'binary' in self.objective:
                                    pred = pred[:, 1].reshape(x_test.shape[0], )
                                if first_fold:
                                    self.prediction = pred
                                    first_fold = False
                                else:
                                    self.prediction += pred
                                nb_model += 1

        self.prediction = self.prediction / nb_model

        if y is not None:
            # calculate metrics if y_true is provided
            if 'regression' not in self.objective:
                if y.shape[1] == 1 and 'binary' not in self.objective:
                    prediction = np.argmax(self.prediction, axis=1).reshape(-1)
                else:
                    prediction = np.where(self.prediction > 0.5, 1, 0)
            else:
                prediction = self.prediction

            if 'binary' in self.objective:
                self.acc_test, self.f1_test, self.recall_test, self.pre_test, self.roc_auc_test = calcul_metric_binary(y,
                                                                                                                       prediction)
                if self.acc_test < 0.5:
                    prediction = 1 - prediction
                    self.acc_test, self.f1_test, self.recall_test, self.pre_test, self.roc_auc_test = calcul_metric_binary(
                        y, prediction)
            elif 'multi-class' in self.objective:
                self.acc_w_test, self.f1_w_test, self.recall_w_test, self.pre_w_test = calcul_metric_classification(y,
                                                                                                                    prediction)
            elif 'regression' in self.objective:
                self.mse_test, self.rmse_test, self.expl_var_test, self.r2_test = calcul_metric_regression(y, prediction)

    def get_prediction(self):
        """
        Returns:
            prediction (array)
        """
        return self.prediction

    def get_scores(self):
        """
        Returns:
            scores (tuple(float)) score between y_true and prediction according to objective
        """
        if 'binary' in self.objective:
            return self.acc_test, self.f1_test, self.recall_test, self.pre_test, self.roc_auc_test
        elif 'multi-class' in self.objective:
            return self.acc_w_test, self.f1_w_test, self.recall_w_test, self.pre_w_test
        elif 'regression' in self.objective:
            return self.mse_test, self.rmse_test, self.expl_var_test, self.r2_test