import numpy as np
import pandas as pd
import os
from glob import glob
from .utils.metrics import calcul_metric_binary, calcul_metric_classification, calcul_metric_regression
from joblib import load
from tensorflow.keras import backend as K
from .flags import load_yaml

from .utils.logging import get_logger
from .utils.ts_preprocessing import build_lag_features_transform, build_rolling_features_transform

logger = get_logger(__name__)


class Prediction:
    """ Class Prediction """

    def __init__(self, objective, name_embedding=None, name_model_full=None, outdir="./", name_logs='last_logs',
                 is_NN=False, class_weight=None, apply_mlflow=False, experiment_name="Experiment", apply_logs=True,
                 apply_app=False, apply_autonlp=False, time_series_recursive=False, time_series_features=None,
                 scaler_info=None):
        """
        Args:
            objective (str) : 'binary' or 'multi-class' or 'regression'
            name_embedding (str) : embedding name method
            name_model_full (str) : full name of model (embedding+classifier_nlp+tag)
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
        self.apply_autonlp = apply_autonlp
        self.time_series_recursive = time_series_recursive
        self.time_series_features = time_series_features
        self.scaler_info = scaler_info

    def fit(self, model, x, y=None, loaded_models=None, x_train = None, y_train = None, position_id_test=None):
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
            if self.apply_autonlp:
                if self.name_embedding.lower() == "transformer":
                    outdir_model = os.path.join(self.outdir, self.name_logs, self.name_embedding, self.name_model_full)
                else:
                    outdir_model = os.path.join(self.outdir, self.name_logs, self.name_embedding,
                                                self.name_model_full.split('+')[1])
            else:
                outdir_model = os.path.join(self.outdir, self.name_logs, self.name_embedding, self.name_model_full)
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

                    if 'time_series' in self.objective and 'lstm' in self.name_model_full:
                        predictions = []

                        if isinstance(x, dict):
                            prediction_steps = x[list(x.keys())[1]].shape[0]
                            timesteps = x['inp'].shape[1]
                            x_shape2 = x['inp'].shape[2]

                            x_preprocessed = {}
                            for col in x.keys():
                                x_preprocessed[col] = np.array([x[col][0]])

                            for j in range(timesteps, timesteps + prediction_steps):
                                if x_preprocessed['tok'].shape[0] != prediction_steps:
                                    predicted_stock_price = model.predict(x_preprocessed)

                                if self.time_series_recursive or y is None:
                                    inp = np.append(x_preprocessed['inp'], predicted_stock_price).reshape(1,
                                                                                                          timesteps + 1,
                                                                                                          x_shape2)
                                else:
                                    try:
                                        inp = np.append(x_preprocessed['inp'], y.values[j - timesteps]).reshape(1,
                                                                                                                timesteps + 1,
                                                                                                                x_shape2)
                                    except:
                                        inp = np.append(x_preprocessed['inp'], y[j - timesteps]).reshape(1,
                                                                                                         timesteps + 1,
                                                                                                         x_shape2)

                                x_preprocessed = {'inp': inp[0, -timesteps:].reshape(1, timesteps, x_shape2)}
                                for col in x.keys():
                                    if col != 'inp':
                                        x_preprocessed[col] = np.array([x[col][j - timesteps]])

                                # pred = scaler_ts.inverse_transform(predicted_stock_price.reshape(1,-1))
                                pred_j = predicted_stock_price.reshape(1, -1)
                                y_shape2 = pred_j.shape[1]
                                predictions.append(pred_j)
                            pred = np.array(predictions).reshape(prediction_steps, y_shape2)

                    elif 'time_series' in self.objective and self.time_series_recursive:
                        step_lags, step_rolling, win_type, position_id, position_date = self.time_series_features

                        max_feat = np.max(step_lags + step_rolling) + 1
                        date_for_train = list(x_train['inp'][position_date].unique()[-max_feat:])
                        data_test = x_train['inp'][x_train['inp'][position_date].isin(date_for_train)].copy()
                        y_test = y_train[y_train.index.isin(list(data_test.index))].copy()

                        if position_id is None:
                            pass
                        else:
                            if isinstance(position_id, str):
                                pass
                            else:
                                position_id = pd.concat(
                                        [position_id[position_id.index.isin(list(data_test.index))], position_id_test],
                                        axis=0, ignore_index=True)

                        pred = np.zeros((x['inp'].shape[0], y_train.shape[1]))
                        index = 0
                        for date in x['inp'][position_date].unique():
                            x_pred_inp = x['inp'][x['inp'][position_date].isin([date])].copy()
                            data_test = pd.concat([data_test, x_pred_inp], axis=0, ignore_index=True)
                            data_test[self.scaler_info[1]] = self.scaler_info[0].inverse_transform(
                                    data_test[self.scaler_info[1]])

                            data_test = build_lag_features_transform(data_test, pd.concat([y_test, pd.DataFrame(
                                    pred[:len(data_test) - len(y_test)], columns=y_test.columns)], axis=0,
                                                                                              ignore_index=True),
                                                                         step_lags, position_id)
                            data_test = build_rolling_features_transform(data_test, pd.concat([y_test, pd.DataFrame(
                                    pred[:len(data_test) - len(y_test)], columns=y_test.columns)], axis=0,
                                                                                                  ignore_index=True),
                                                                             step_rolling, win_type, position_id)
                            data_test[self.scaler_info[1]] = self.scaler_info[0].transform(
                                    data_test[self.scaler_info[1]])

                            x_preprocessed = {'inp': data_test[-len(x_pred_inp):].values}
                            for col in x.keys():
                                if col != 'inp':
                                    x_preprocessed[col] = x[col][index:(index + len(x_pred_inp))]
                            pred[index:(index + len(x_pred_inp))] = model.predict(x_preprocessed)
                            index += len(x_pred_inp)
                        del data_test, x_pred_inp

                    else:
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

                                if 'time_series' in self.objective:
                                    logger.error("not implemented for mlflow, add a copy of the previous code ")

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

                    if 'time_series' in self.objective and self.time_series_recursive:
                        step_lags, step_rolling, win_type, position_id, position_date = self.time_series_features

                        max_feat = np.max(step_lags + step_rolling) + 1
                        date_for_train = list(x_train[position_date].unique()[-max_feat:])
                        data_test = x_train[x_train[position_date].isin(date_for_train)].copy()
                        y_test = y_train[y_train.index.isin(list(data_test.index))].copy()

                        if position_id is None:
                            pass
                        else:
                            if isinstance(position_id, str):
                                pass
                            else:
                                position_id = pd.concat(
                                    [position_id[position_id.index.isin(list(data_test.index))], position_id_test],
                                    axis=0, ignore_index=True)

                        if y_train.shape[1] == 1:
                            pred = np.zeros(x.shape[0])
                        else:
                            pred = np.zeros((x.shape[0], y_train.shape[1]))
                        index = 0
                        for date in x[position_date].unique():
                            x_pred = x[x[position_date].isin([date])].copy()
                            data_test = pd.concat([data_test, x_pred], axis=0, ignore_index=True)
                            data_test[self.scaler_info[1]] = self.scaler_info[0].inverse_transform(
                                data_test[self.scaler_info[1]])

                            data_test = build_lag_features_transform(data_test, pd.concat([y_test, pd.DataFrame(
                                pred[:len(data_test) - len(y_test)], columns=y_test.columns)], axis=0,
                                                                                          ignore_index=True),
                                                                     step_lags, position_id)
                            data_test = build_rolling_features_transform(data_test, pd.concat([y_test, pd.DataFrame(
                                pred[:len(data_test) - len(y_test)], columns=y_test.columns)], axis=0,
                                                                                              ignore_index=True),
                                                                         step_rolling, win_type, position_id)
                            data_test[self.scaler_info[1]] = self.scaler_info[0].transform(
                                data_test[self.scaler_info[1]])

                            pred[index:(index + len(x_pred))] = model.predict(
                                data_test[-len(x_pred):].values)
                            index += len(x_pred)
                        del data_test, x_pred

                    else:
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

                                if 'time_series' in self.objective and self.time_series_recursive:
                                    step_lags, step_rolling, win_type, position_id, position_date = self.time_series_features

                                    max_feat = np.max(step_lags + step_rolling) + 1
                                    date_for_train = list(x_train[position_date].unique()[-max_feat:])
                                    data_test = x_train[x_train[position_date].isin(date_for_train)].copy()
                                    y_test = y_train[y_train.index.isin(list(data_test.index))].copy()

                                    if position_id is None:
                                        pass
                                    else:
                                        if isinstance(position_id, str):
                                            pass
                                        else:
                                            position_id = pd.concat(
                                                [position_id[position_id.index.isin(list(data_test.index))],
                                                 position_id_test],
                                                axis=0, ignore_index=True)

                                    if y_train.shape[1] == 1:
                                        pred = np.zeros(x.shape[0])
                                    else:
                                        pred = np.zeros((x.shape[0], y_train.shape[1]))
                                    index = 0
                                    for date in x[position_date].unique():
                                        x_pred = x[x[position_date].isin([date])].copy()
                                        data_test = pd.concat([data_test, x_pred], axis=0, ignore_index=True)
                                        data_test[self.scaler_info[1]] = self.scaler_info[0].inverse_transform(
                                            data_test[self.scaler_info[1]])

                                        data_test = build_lag_features_transform(data_test,
                                                                                 pd.concat([y_test, pd.DataFrame(
                                                                                     pred[
                                                                                     :len(data_test) - len(y_test)],
                                                                                     columns=y_test.columns)], axis=0,
                                                                                           ignore_index=True),
                                                                                 step_lags, position_id)
                                        data_test = build_rolling_features_transform(data_test,
                                                                                     pd.concat([y_test, pd.DataFrame(
                                                                                         pred[
                                                                                         :len(data_test) - len(y_test)],
                                                                                         columns=y_test.columns)],
                                                                                               axis=0,
                                                                                               ignore_index=True),
                                                                                     step_rolling, win_type,
                                                                                     position_id)
                                        data_test[self.scaler_info[1]] = self.scaler_info[0].transform(
                                            data_test[self.scaler_info[1]])

                                        pred[index:(index + len(x_pred))] = loaded_model.predict(
                                            data_test[-len(x_pred):].values)
                                        index += len(x_pred)
                                    del data_test, x_pred

                                else:
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