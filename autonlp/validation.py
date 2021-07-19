import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.model_selection import KFold, StratifiedKFold
from joblib import dump, load
import random as rd
import os
import time

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from .utils.metrics import roc, calcul_metric_binary, calcul_metric_classification, calcul_metric_regression
from .utils.class_weight import compute_dict_class_weight

import logging
from .utils.logging import get_logger, verbosity_to_loglevel
from .utils.ts_preprocessing import build_lag_features_transform, build_rolling_features_transform

logger = get_logger(__name__)


class Validation:
    """ Class validation/cross-validation """

    def __init__(self, objective, seed=15, is_NN=False, name_embedding=None, name_model_full=None, class_weight=None,
                 average_scoring="weighted", apply_mlflow=False, experiment_name="Experiment", apply_logs=True,
                 apply_autonlp=False, size_train=10, time_series_recursive=False, time_series_features=None,
                 scaler_info=None):
        """
        Args:
            objective (str) : 'binary' or 'multi-class' or 'regression'
            seed (int)
            is_NN (Boolean) True if the model is a Neural Network
            name_model (str) : general name of model
            name_model_full (str) : full name of model (embedding+classifier_nlp+tag)
            class_weight (None or 'balanced')
            average_scoring (str) : 'micro', 'macro' or 'weighted'
            apply_mlflow (Boolean)
            experiment_name (str)
        """
        self.seed = seed
        self.objective = objective
        self.is_NN = is_NN
        self.name_embedding = name_embedding
        self.name_model_full = name_model_full
        self.class_weight = class_weight
        self.average_scoring = average_scoring
        self.apply_mlflow = apply_mlflow
        self.path_mlflow = "../mlruns"
        self.experiment_name = experiment_name
        self.apply_logs = apply_logs
        self.apply_autonlp = apply_autonlp
        self.size_train = size_train
        self.time_series_recursive = time_series_recursive
        self.time_series_features = time_series_features
        self.scaler_info = scaler_info

    def fit(self, model, x, y, x_valid=None, y_valid=None, nfolds=5, nfolds_train=5, cv_strategy="StratifiedKFold",
            scoring='accuracy', outdir='./', params_all=dict(), batch_size=16, patience=4, epochs=60, min_lr=1e-4):
        """ Fit model on train set and predict on cross-validation
        Args:
            model (self.model) function of the model architecture not instantiated : self.model and not self.model()
            x (List or dict or DataFrame)
            y (Dataframe)
            x_valid (List or dict or DataFrame)
            y_valid (Dataframe)
            nfolds (int) number of folds to split during optimization
            nfolds_train (int) number of folds to train during optimization
            cv_strategy ("StratifiedKFold" or "KFold")
            scoring (str) score to optimize
            outdir (str) logs path
            params_all (dict) params to save in order to reuse the model
            Only for NN models:
                batch_size (int)
                patience (int)
                epochs (int)
                min_lr (float) minimum for learning rate reduction
        """

        if x_valid is None:
            self.fold_id = np.ones((len(y),)) * -1
            if "time_series" not in self.objective:
                # Cross-validation split in self.nfolds but train only on self.nfolds_train chosen randomly :
                rd.seed(self.seed)
                fold_to_train = rd.sample([i for i in range(nfolds)], k=max(min(nfolds_train, nfolds), 1))
                if cv_strategy == "StratifiedKFold":
                    skf = StratifiedKFold(n_splits=nfolds, random_state=self.seed, shuffle=True)
                    folds = skf.split(y, y)
                else:
                    kf = KFold(n_splits=nfolds, shuffle=True, random_state=self.seed)
                    folds = kf.split(y)
            else:
                folds = [('all', [i for i in range(self.size_train)])]
                fold_to_train = [0]
        else:
            self.fold_id = np.ones((len(y_valid),)) * -1
            folds = [('all', [i for i in range(y_valid.shape[0])])]
            fold_to_train = [0]

        if self.apply_logs:
            outdir_embedding = os.path.join(outdir, 'last_logs', self.name_embedding)
            os.makedirs(outdir_embedding, exist_ok=True)
            if self.apply_autonlp:
                if self.name_embedding.lower() == "transformer":
                    outdir_model = os.path.join(outdir_embedding, self.name_model_full)
                else:
                    outdir_model = os.path.join(outdir_embedding, self.name_model_full.split('+')[1])
            else:
                outdir_model = os.path.join(outdir_embedding, self.name_model_full)
            os.makedirs(outdir_model, exist_ok=True)

        if self.apply_mlflow:
            import mlflow
            from mlflow.tracking import MlflowClient
            # mlflow : (mlflow ui --backend-store-uri ./mlruns)
            client = MlflowClient()
            experiment_name = self.experiment_name
            #experiment_id = client.create_experiment(experiment_name)
            try:
                mlflow.set_experiment(experiment_name=experiment_name)
            except:
                current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
                experiment_id = current_experiment['experiment_id']
                client.restore_experiment(experiment_id)
            current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
            experiment_id = current_experiment['experiment_id']

        if self.is_NN:
            # validation for neural network models :
            total_epochs = 0

            first_fold = True
            for num_fold, (train_index, val_index) in enumerate(folds):
                if num_fold not in fold_to_train:
                    continue
                if self.apply_mlflow:
                    mlflow.start_run(run_name=self.name_model_full, tags={'name_embedding': self.name_embedding})
                logger.info("Fold {}:".format(num_fold))

                if train_index == 'all':
                    # validation
                    if isinstance(x, dict):
                        x_train, x_val = x, x_valid
                    elif isinstance(x, list):
                        x_train, x_val = x, x_valid
                    else:
                        if isinstance(x, pd.DataFrame):
                            x_train, x_val = x.values, x_valid.values
                        else:
                            x_train, x_val = x, x_valid
                    if isinstance(y, pd.DataFrame):
                        y_train, y_val = y.values, y_valid.values
                    else:
                        y_train, y_val = y, y_valid
                else:
                    # cross-validation
                    if isinstance(x, dict):
                        x_train, x_val = {}, {}
                        for col in x.keys():
                            x_train[col], x_val[col] = x[col][train_index], x[col][val_index]
                    elif isinstance(x, list):
                        x_train, x_val = [], []
                        for col in range(len(x)):
                            x_train.append(x[col][train_index])
                            x_val.append(x[col][val_index])
                    else:
                        if isinstance(x, pd.DataFrame):
                            x_train, x_val = x.values[train_index], x.values[val_index]
                        else:
                            x_train, x_val = x[train_index], x[val_index]
                    if isinstance(y, pd.DataFrame):
                        y_train, y_val = y.values[train_index], y.values[val_index]
                    else:
                        y_train, y_val = y[train_index], y[val_index]

                K.clear_session()

                model_nn = model()

                if first_fold:
                    logger.info(model_nn.summary())

                if 'regression' in self.objective:
                    if 'mean_squared_error' in scoring or 'mse' in scoring:
                        monitor = 'mean_squared_error'
                    else:
                        monitor = 'loss'
                    monitor_checkpoint = 'mean_squared_error'
                else:
                    if y.shape[1] == 1:
                        if scoring == 'accuracy':
                            monitor = 'accuracy'
                            monitor_checkpoint = 'accuracy'
                        else:
                            monitor = 'loss'
                            monitor_checkpoint = 'loss' #'accuracy'
                    else:
                        monitor = 'binary_crossentropy'
                        monitor_checkpoint = 'binary_crossentropy'

                rlr = ReduceLROnPlateau(monitor='val_' + monitor, factor=0.1, patience=patience - 1,
                                        verbose=1, min_delta=1e-4, mode='auto', min_lr=min_lr)

                es = EarlyStopping(monitor='val_' + monitor, min_delta=0.0001, patience=patience, mode='auto',
                                   baseline=None, restore_best_weights=True, verbose=0)

                if self.apply_logs:
                    ckp = ModelCheckpoint('{}/fold{}.hdf5'.format(outdir_model, num_fold),
                                          monitor='val_' + monitor_checkpoint, verbose=1,
                                          save_best_only=True, save_weights_only=True, mode='auto')
                    callbacks = [rlr, es, ckp]
                else:
                    callbacks = [rlr, es]

                if self.apply_mlflow:
                    mlflow.tensorflow.autolog(every_n_iter=2, silent=True)

                train_history = model_nn.fit(x_train, y_train,
                                             validation_data=(x_val, y_val),
                                             class_weight=compute_dict_class_weight(y_train, self.class_weight,
                                                                                    self.objective),
                                             epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks)

                if self.apply_logs:
                    outdir_plot = os.path.join(outdir_model, "plot_metrics_fold{}.png".format(num_fold))
                    save_plot_metrics(train_history.history, outdir_plot)

                if len(train_history.history[monitor]) <= (patience+1):
                    p = 0
                else:
                    p = patience

                logger.info('Kfold #{} : train {} = {} and val {} = {}'.format(num_fold, monitor,
                                                                               train_history.history[monitor][-(p + 1)],
                                                                               monitor,
                                                                               train_history.history['val_' + monitor][
                                                                                   -(p + 1)]))
                total_epochs += len(train_history.history[monitor][:-(p + 1)])

                if "time_series" in self.objective and 'lstm' in self.name_model_full.lower() and self.time_series_recursive:  # use predicted time series for next prediction
                    prediction_steps = x_val['inp'].shape[0]
                    timesteps = x_val['inp'].shape[1]
                    x_shape2 = x_val['inp'].shape[2]
                    y_shape2 = y_val.shape[1]

                    x_preprocessed = {}
                    for col in x_val.keys():
                        x_preprocessed[col] = np.array([x_val[col][0]])

                    predictions = []
                    for j in range(timesteps, timesteps + prediction_steps):
                        predicted_stock_price = model_nn.predict(x_preprocessed)

                        inp = np.append(x_preprocessed['inp'], predicted_stock_price).reshape(1, timesteps + 1,
                                                                                              x_shape2)

                        x_preprocessed = {'inp': inp[0, -timesteps:].reshape(1, timesteps, x_shape2)}
                        for col in x.keys():
                            if col != 'inp':
                                x_preprocessed[col] = np.array([x[col][j - timesteps]])

                        # pred = scaler_ts.inverse_transform(predicted_stock_price.reshape(1,-1))
                        pred = predicted_stock_price.reshape(1, -1)
                        predictions.append(pred)
                    pred_val = np.array(predictions).reshape(prediction_steps, y_shape2)

                elif "time_series" in self.objective and "dense_network" in self.name_model_full.lower() and self.time_series_recursive:
                    # lag_features and rolling features day after day
                    step_lags, step_rolling, win_type, position_id, position_date = self.time_series_features

                    max_feat = np.max(step_lags + step_rolling) + 1
                    date_for_train = list(x['inp'].iloc[train_index][position_date].unique()[-max_feat:])
                    data_test = x['inp'].iloc[train_index][x['inp'][position_date].isin(date_for_train)].copy()
                    y_test = y[y.index.isin(list(data_test.index))].copy()
                    start = time.perf_counter()
                    if position_id is None:
                        pass
                    else:
                        if isinstance(position_id, str):
                            pass
                        else:
                            position_id = pd.concat(
                                [position_id[position_id.index.isin(list(data_test.index))], position_id.iloc[train_index]],
                                axis=0, ignore_index=True)

                    pred_val = np.zeros(y_val.shape)
                    index = 0
                    for date in x['inp'].iloc[train_index][position_date].unique():
                        x_val_inp = x['inp'].iloc[train_index][x['inp'][position_date].isin([date])].copy()
                        data_test = pd.concat([data_test, x_val_inp], axis=0, ignore_index=True)
                        data_test[self.scaler_info[1]] = self.scaler_info[0].inverse_transform(
                            data_test[self.scaler_info[1]])

                        data_test = build_lag_features_transform(data_test, pd.concat(
                            [y_test, pd.DataFrame(pred_val[:len(data_test) - len(y_test)], columns=y_test.columns)],
                            axis=0, ignore_index=True),
                                                                 step_lags, position_id)
                        data_test = build_rolling_features_transform(data_test, pd.concat(
                            [y_test, pd.DataFrame(pred_val[:len(data_test) - len(y_test)], columns=y_test.columns)],
                            axis=0, ignore_index=True),
                                                                     step_rolling, win_type, position_id)
                        data_test[self.scaler_info[1]] = self.scaler_info[0].transform(data_test[self.scaler_info[1]])

                        x_preprocessed = {'inp': data_test[-len(x_val_inp):].values}
                        for col in x_val.keys():
                            if col != 'inp':
                                x_preprocessed[col] = x_val[col][index:(index + len(x_val_inp))]
                        pred_val[index:(index + len(x_val_inp))] = model_nn.predict(x_preprocessed)
                        index += len(x_val_inp)
                    print('Time prediction_val :', time.perf_counter() - start)
                else:
                    pred_val = model_nn.predict(x_val)

                if first_fold:
                    first_fold = False
                    if 'binary' in self.objective or ('regression' in self.objective and y.shape[1] == 1):
                        if train_index == 'all':
                            self.oof_val = np.zeros((y_val.shape[0],))
                        else:
                            self.oof_val = np.zeros((y.shape[0],))
                    else:
                        if train_index == 'all':
                            self.oof_val = np.zeros((y_val.shape[0], pred_val.shape[1]))
                        else:
                            self.oof_val = np.zeros((y.shape[0], pred_val.shape[1]))
                if 'binary' in self.objective or ('regression' in self.objective and y.shape[1] == 1):
                    pred_val = pred_val.reshape(-1)
                self.oof_val[val_index] = pred_val
                self.fold_id[val_index] = num_fold

                # log_metrics
                if 'binary' in self.objective:
                    m_binary, roc_binary = self.get_metrics(y_val, pred_val, False)
                    acc_val, f1_val, recall_val, pre_val, roc_auc_val = m_binary
                    metrics = {"acc_val": acc_val, "f1_val": f1_val, "recall_val": recall_val, "pre_val": pre_val,
                               "roc_auc_val": roc_auc_val}
                elif 'multi-class' in self.objective:
                    acc_val, f1_val, recall_val, pre_val = self.get_metrics(y_val, pred_val, False)
                    metrics = {"acc_val": acc_val, "f1_val": f1_val, "recall_val": recall_val, "pre_val": pre_val}
                elif 'regression' in self.objective:
                    mse_val, rmse_val, expl_var_val, r2_val = self.get_metrics(y_val, pred_val, False)
                    metrics = {"mse_val": mse_val, "rmse_val": rmse_val, "expl_var_val": expl_var_val,
                               "r2_val": r2_val}
                #logger.info(metrics)

                if self.apply_mlflow:
                    mlflow.log_metrics(metrics)

                    # log params
                    params = {"seed": self.seed, "objective": self.objective,
                              "scoring": scoring, "average_scoring": self.average_scoring}
                    if x_valid is None:
                        params["nfolds"] = nfolds
                        params["num_fold"] = num_fold
                        params["cv_strategy"] = cv_strategy
                    mlflow.log_params(params)

                    # log params_all
                    mlflow.log_dict(params_all, "parameters.json")
                    mlflow.end_run()

                    mlflow.tensorflow.autolog(every_n_iter=2, log_models=False, disable=True, silent=True)

        else:
            # validation for sklearn models, catboost, xgboost and lightgbm :

            first_fold = True
            for num_fold, (train_index, val_index) in enumerate(folds):
                if num_fold not in fold_to_train:
                    continue
                if self.apply_mlflow:
                    mlflow.start_run(run_name=self.name_model_full, tags={'name_embedding':self.name_embedding})
                if train_index == 'all':
                    # validation
                    if isinstance(x, pd.DataFrame):
                        x_train, x_val = x.values, x_valid.values
                    else:
                        x_train, x_val = x, x_valid
                    if isinstance(y, pd.DataFrame):
                        y_train, y_val = y.values, y_valid.values
                    else:
                        y_train, y_val = y, y_valid
                else:
                    # cross-validation
                    if isinstance(x, pd.DataFrame):
                        x_train, x_val = x.values[train_index], x.values[val_index]
                    else:
                        x_train, x_val = x[train_index], x[val_index]
                    if isinstance(y, pd.DataFrame):
                        y_train, y_val = y.values[train_index], y.values[val_index]
                    else:
                        y_train, y_val = y[train_index], y[val_index]

                model_skl = model()
                #mlflow.sklearn.autolog(log_models=False, exclusive=True)
                model_skl.fit(x_train, y_train)
                if self.apply_mlflow:
                    mlflow.sklearn.log_model(model_skl, self.name_model_full)

                if self.apply_logs:
                    dump(model_skl, '{}/fold{}.joblib'.format(outdir_model, num_fold))

                if 'time_series' in self.objective and self.time_series_recursive:
                    # lag_features and rolling features day after day
                    step_lags, step_rolling, win_type, position_id, position_date = self.time_series_features

                    max_feat = np.max(step_lags + step_rolling) + 1
                    date_for_train = list(x.iloc[train_index][position_date].unique()[-max_feat:])
                    data_test = x.iloc[train_index][x[position_date].isin(date_for_train)].copy()
                    y_test = y[y.index.isin(list(data_test.index))].copy()
                    start = time.perf_counter()

                    if position_id is None:
                        pass
                    else:
                        if isinstance(position_id, str):
                            pass
                        else:
                            position_id = pd.concat(
                                [position_id[position_id.index.isin(list(data_test.index))], position_id.iloc[train_index]],
                                axis=0, ignore_index=True)

                    if y_val.shape[1] == 1:
                        pred_val = np.zeros(y_val.shape[0])
                    else:
                        pred_val = np.zeros(y_val.shape)
                    index = 0
                    for date in x.iloc[train_index][position_date].unique():
                        x_val = x.iloc[train_index][x[position_date].isin([date])].copy()
                        data_test = pd.concat([data_test, x_val], axis=0, ignore_index=True)
                        data_test[self.scaler_info[1]] = self.scaler_info[0].inverse_transform(
                            data_test[self.scaler_info[1]])

                        data_test = build_lag_features_transform(data_test, pd.concat(
                            [y_test, pd.DataFrame(pred_val[:len(data_test) - len(y_test)], columns=y_test.columns)],
                            axis=0, ignore_index=True),
                                                                 step_lags, position_id)
                        data_test = build_rolling_features_transform(data_test, pd.concat(
                            [y_test, pd.DataFrame(pred_val[:len(data_test) - len(y_test)], columns=y_test.columns)],
                            axis=0, ignore_index=True),
                                                                     step_rolling, win_type, position_id)
                        data_test[self.scaler_info[1]] = self.scaler_info[0].transform(data_test[self.scaler_info[1]])

                        if 'regression' in self.objective:
                            pred_val[index:(index + len(x_val))] = model_skl.predict(data_test[-len(x_val):].values)
                        else:
                            pred_val[index:(index + len(x_val))] = model_skl.predict_proba(data_test[-len(x_val):].values)
                        index += len(x_val)
                    print('Time prediction_val :', time.perf_counter() - start)
                    del data_test, y_test
                else:
                    if 'regression' in self.objective:
                        pred_val = model_skl.predict(x_val)
                    else:
                        pred_val = model_skl.predict_proba(x_val)



                if first_fold:
                    first_fold = False
                    if 'binary' in self.objective or ('regression' in self.objective and y.shape[1] == 1):
                        if train_index == 'all':
                            self.oof_val = np.zeros((y_val.shape[0],))
                        else:
                            self.oof_val = np.zeros((y.shape[0],))
                    else:
                        if train_index == 'all':
                            self.oof_val = np.zeros((y_val.shape[0], pred_val.shape[1]))
                        else:
                            self.oof_val = np.zeros((y.shape[0], pred_val.shape[1]))
                if 'binary' in self.objective:
                    pred_val = pred_val[:, 1].reshape(x_val.shape[0], )
                elif 'regression' in self.objective and y.shape[1] == 1:
                    pred_val = pred_val.reshape(x_val.shape[0], )
                self.oof_val[val_index] = pred_val
                self.fold_id[val_index] = num_fold

                # log_metrics
                if 'binary' in self.objective:
                    m_binary, roc_binary = self.get_metrics(y_val, pred_val, False)
                    acc_val, f1_val, recall_val, pre_val, roc_auc_val = m_binary
                    metrics = {"acc_val": acc_val, "f1_val": f1_val, "recall_val": recall_val, "pre_val": pre_val,
                               "roc_auc_val": roc_auc_val}
                elif 'multi-class' in self.objective:
                    acc_val, f1_val, recall_val, pre_val = self.get_metrics(y_val, pred_val, False)
                    metrics = {"acc_val": acc_val, "f1_val": f1_val, "recall_val": recall_val, "pre_val": pre_val}
                elif 'regression' in self.objective:
                    mse_val, rmse_val, expl_var_val, r2_val = self.get_metrics(y_val, pred_val, False)
                    metrics = {"mse_val": mse_val, "rmse_val": rmse_val, "expl_var_val": expl_var_val,
                               "r2_val": r2_val}
                #logger.info(metrics)
                if self.apply_mlflow:
                    mlflow.log_metrics(metrics)

                    # log params
                    params = {"seed": self.seed, "objective": self.objective, "class_weight": self.class_weight,
                              "scoring": scoring, "average_scoring": self.average_scoring}
                    if x_valid is None:
                        params["nfolds"] = nfolds
                        params["num_fold"] = num_fold
                        params["cv_strategy"] = cv_strategy
                    mlflow.log_params(params)

                    # log params_all
                    mlflow.log_dict(params_all, "parameters.json")
                    mlflow.end_run()

        prediction_oof_val = self.oof_val.copy()

        if isinstance(y, pd.DataFrame):
            if x_valid is None:
                # cross-validation
                y_true_sample = y.values[np.where(self.fold_id >= 0)[0]].copy()
            else:
                # validation
                y_true_sample = y_valid.values[np.where(self.fold_id >= 0)[0]].copy()
        else:
            if x_valid is None:
                # cross-validation
                y_true_sample = y[np.where(self.fold_id >= 0)[0]].copy()
            else:
                # validation
                y_true_sample = y_valid[np.where(self.fold_id >= 0)[0]].copy()
        prediction_oof_val = prediction_oof_val[np.where(self.fold_id >= 0)[0]]

        if 'binary' in self.objective:
            m_binary, roc_binary = self.get_metrics(y_true_sample, prediction_oof_val)
            self.acc_val, self.f1_val, self.recall_val, self.pre_val, self.roc_auc_val = m_binary
            self.fpr, self.tpr = roc_binary

        elif 'multi-class' in self.objective:
            self.acc_val, self.f1_val, self.recall_val, self.pre_val = self.get_metrics(y_true_sample, prediction_oof_val)
        elif 'regression' in self.objective:
            self.mse_val, self.rmse_val, self.expl_var_val, self.r2_val = self.get_metrics(y_true_sample, prediction_oof_val)
        del x_train, x_val, y_train, y_val, model

    def get_metrics(self, y_true, oof_val, print_score=True):
        if 'regression' not in self.objective:
            if y_true.shape[1] == 1 and 'binary' not in self.objective:
                prediction_oof_val = np.argmax(oof_val, axis=1).reshape(-1).copy()
            else:
                prediction_oof_val = np.where(oof_val > 0.5, 1, 0).copy()
        else:
            prediction_oof_val = oof_val.copy()

        if 'binary' in self.objective:
            m_binary = calcul_metric_binary(y_true, prediction_oof_val, 0.5, print_score)
            try:
                roc_binary = roc(y_true.values, oof_val)
            except:
                roc_binary = roc(y_true, oof_val)
            return m_binary, roc_binary

        elif 'multi-class' in self.objective:
            return calcul_metric_classification(y_true, prediction_oof_val, self.average_scoring, print_score)
        elif 'regression' in self.objective:
            return calcul_metric_regression(y_true, prediction_oof_val, print_score)

    def get_cv_prediction(self):
        """
        Returns:
            fold_id (array) number of fold of each data, -1 if it was not use for validation
            oof_val (array) validation prediction, data not use for validation are removed
        """
        return self.fold_id, self.oof_val[np.where(self.fold_id >= 0)[0]]

    def get_scores(self):
        """
        Returns:
            scores (tuple(float)) score between y_true and oof_val according to objective
        """
        if 'binary' in self.objective:
            return self.acc_val, self.f1_val, self.recall_val, self.pre_val, self.roc_auc_val
        elif 'multi-class' in self.objective:
            return self.acc_val, self.f1_val, self.recall_val, self.pre_val
        elif 'regression' in self.objective:
            return self.mse_val, self.rmse_val, self.expl_var_val, self.r2_val

    def get_roc(self):
        """
        Returns:
            fpr (array) Increasing false positive rates
            tpr (array) Increasing true positive rates
        """
        return self.fpr, self.tpr


def save_plot_metrics(history, outdir):
    """ Save plot of metrics from history for each epoch
    Args:
        history : history from tensorflow fit model
        outdir (str) save plot in outdir directory
    """
    # Plotting
    metrics = [x for x in history.keys() if 'val' not in x and 'lr' not in x]

    fig = make_subplots(rows=len(metrics), cols=1)
    # f, axs = plt.subplots(1, len(metrics), figsize=(15, 5))

    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Scatter(x=np.arange(1, len(history[metric]) + 1), y=history[metric], name=metric, mode='lines+markers',
                       line=dict(color='royalblue'), legendgroup=str(i + 1)),
            row=i + 1, col=1
        )
        # axs[i].plot(range(1, len(history[metric]) + 2), history[metric], label=metric)
        if 'val_' + metric in history.keys():
            fig.add_trace(
                go.Scatter(x=np.arange(1, len(history['val_' + metric]) + 1), y=history['val_' + metric],
                           name='val_' + metric, mode='lines+markers', line=dict(color='firebrick'),
                           legendgroup=str(i + 1)),
                row=i + 1, col=1
            )
            # axs[i].plot(range(1, len(history['val_' + metric]) + 2), history['val_' + metric], label='val_' + metric)
        # axs[i].legend()
        # axs[i].grid()
        fig.update_xaxes(title_text="epoch", row=i + 1, col=1)
        fig.update_yaxes(title_text=metric, row=i + 1, col=1)

    fig.update_layout(title='Model metrics', legend_tracegroupgap=160)

    # plt.tight_layout()

    try:
        fig.write_image(outdir)
    except:
        pass
