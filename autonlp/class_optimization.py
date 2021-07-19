import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
import random as rd

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import *

import warnings

warnings.filterwarnings("ignore")

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import gc
from hyperopt import hp, fmin, tpe, Trials

from .utils.class_weight import compute_dict_class_weight

import logging
from .utils.logging import get_logger, verbosity_to_loglevel

logger = get_logger(__name__)


##############################
##############################
##############################


class Optimiz_hyperopt:
    """ Class Hyperopt optimization for sklearn model type """

    def __init__(self, Model_sklearn, hyper_params, apply_optimization):
        """
        Args:
            Model_sklearn (Model class)
            hyper_params (dict) a hyperopt range for each hyperparameters
            apply_optimization (Boolean) if False, initialize random hyperparameters and give a null score
        """
        self.Model_sklearn = Model_sklearn
        self.hyper_params = hyper_params
        self.apply_optimization = apply_optimization

    def optimise(self, params):
        """ function to optimize by hyperopt library
        Args :
            params (dict) a hyperopt range for each hyperparameters
        Return:
            score (float) result of metric on validation set
        """

        self.Model_sklearn.initialize_params(self.y, params)

        # logger.info(self.Model_sklearn.p)

        start = time.perf_counter()

        if self.x_val is None:
            # cross-validation
            fold_id = np.ones((len(self.y),)) * -1
            oof_val = np.zeros((self.y.shape[0], self.y.shape[1]))
            # Cross-validation split in self.nfolds but train only on self.nfolds_train chosen randomly
            rd.seed(self.Model_sklearn.seed)
            fold_to_train = rd.sample([i for i in range(self.nfolds)], k=max(min(self.nfolds_train, self.nfolds), 1))

            if self.cv_strategy == "StratifiedKFold":
                skf = StratifiedKFold(n_splits=self.nfolds, random_state=self.Model_sklearn.seed, shuffle=True)
                folds = skf.split(self.y, self.y)
            else:
                kf = KFold(n_splits=self.nfolds, random_state=self.Model_sklearn.seed, shuffle=True)
                folds = kf.split(self.y)
        else:
            # validation
            fold_id = np.ones((len(self.y_val),)) * -1
            oof_val = np.zeros((self.y_val.shape[0], self.y_val.shape[1]))
            folds = [('all', [i for i in range(oof_val.shape[0])])]
            fold_to_train = [0]

        for n, (tr, te) in enumerate(folds):
            if n not in fold_to_train:
                continue

            if tr == 'all':
                # validation
                if isinstance(self.x, pd.DataFrame):
                    x_tr, x_val = self.x.values, self.x_val.values
                else:
                    x_tr, x_val = self.x, self.x_val
                if isinstance(self.y, pd.DataFrame):
                    y_tr, y_val = self.y.values, self.y_val.values
                else:
                    y_tr, y_val = self.y, self.y_val
            else:
                # cross-validation
                if isinstance(self.x, pd.DataFrame):
                    x_tr, x_val = self.x.values[tr], self.x.values[te]
                else:
                    x_tr, x_val = self.x[tr], self.x[te]
                if isinstance(self.y, pd.DataFrame):
                    y_tr, y_val = self.y.values[tr], self.y.values[te]
                else:
                    y_tr, y_val = self.y[tr], self.y[te]

            model = self.Model_sklearn.model()
            model.fit(x_tr, y_tr)

            if self.Model_sklearn.shape_y == 1:
                oof_val[te, :] = model.predict(x_val).reshape(-1, 1)
            else:
                oof_val[te, :] = model.predict(x_val)
            fold_id[te] = n
            del model

        metrics = []
        for i in range(self.Model_sklearn.shape_y):
            if self.x_val is None:
                # cross_validation
                y_true = self.y.iloc[:, i].copy()
            else:
                # validation
                y_true = self.y_val.iloc[:, i].copy()
            # subset, only use data where fold_id >= 0 :
            y_true_sample = y_true.values[np.where(fold_id >= 0)[0]]
            prediction_oof_val = oof_val[:, i][np.where(fold_id >= 0)[0]]
            if 'regression' in self.Model_sklearn.objective:
                if 'explained_variance' == self.scoring:
                    metrics.append(-explained_variance_score(y_true_sample, prediction_oof_val))
                elif 'r2' == self.scoring:
                    metrics.append(-r2_score(y_true_sample, prediction_oof_val))
                else:
                    metrics.append(mean_squared_error(y_true_sample, prediction_oof_val))
            else:
                if 'f1' in self.scoring:
                    if 'binary' in self.Model_sklearn.objective:
                        metrics.append(-f1_score(y_true_sample, prediction_oof_val))
                    else:
                        metrics.append(
                            -f1_score(y_true_sample, prediction_oof_val, average=self.Model_sklearn.average_scoring))
                elif 'recall' in self.scoring:
                    if 'binary' in self.Model_sklearn.objective:
                        metrics.append(-recall_score(y_true_sample, prediction_oof_val))
                    else:
                        metrics.append(-recall_score(y_true_sample, prediction_oof_val,
                                                     average=self.Model_sklearn.average_scoring))
                elif 'precision' in self.scoring:
                    if 'binary' in self.Model_sklearn.objective:
                        metrics.append(-precision_score(y_true_sample, prediction_oof_val))
                    else:
                        metrics.append(-precision_score(y_true_sample, prediction_oof_val,
                                                        average=self.Model_sklearn.average_scoring))
                elif 'roc' in self.scoring or 'auc' in self.scoring:
                    if 'binary' in self.Model_sklearn.objective:
                        metrics.append(-roc_auc_score(y_true_sample, prediction_oof_val))
                    else:
                        metrics.append(-roc_auc_score(y_true_sample, prediction_oof_val,
                                                      average=self.Model_sklearn.average_scoring))
                else:
                    metrics.append(-accuracy_score(y_true_sample, prediction_oof_val))

        score = -np.mean(metrics)
        logger.info('oof_val score {} Metric {}'.format(self.scoring, score))

        # store hyperparameters optimization in a Dataframe self.df_all_results:
        self.df_all_results['mean_fit_time'].append(time.perf_counter() - start)
        self.df_all_results['params'].append(params)
        self.df_all_results['mean_test_score'].append(score)
        self.df_all_results['std_test_score'].append(0)  # just 0

        return np.mean(metrics)

    def optimise_no_optimiz(self, params):
        """ function to optimize by hyperopt library
            use when apply_optimization is False
            initialize random hyperparameters and give a null score
        Args :
            params (dict) a hyperopt range for each hyperparameters
        Return:
            score (float) result of metric on validation set
        """

        self.Model_sklearn.initialize_params(self.y, params)

        score = 0

        self.df_all_results['mean_fit_time'].append(0)
        self.df_all_results['params'].append(params)
        self.df_all_results['mean_test_score'].append(score)
        self.df_all_results['std_test_score'].append(0)  # just 0

        return score

    def train(self, x_, y_, x_val, y_val, nfolds=5, nfolds_train=5, scoring='accuracy', verbose=0,
              time_limit_per_model=60, cv_strategy="StratifiedKFold", trials=None, max_trials=1000):
        """ Compute the function to minimize with hyperopt TPE optimization
            TPE optimization is a Naive Bayes Optimization
        Args:
            x_ (List or dict or DataFrame)
            y_ (Dataframe)
            x_val (List or dict or DataFrame)
            y_val (Dataframe)
            nfolds (int) number of folds to split during optimization
            nfolds_train (int) number of folds to train during optimization
            scoring (str) score to optimize
            verbose (int)
            time_limit_per_model (int) maximum Hyperparameters Optimization time in seconds
            cv_strategy ("StratifiedKFold" or "KFold")
            trials (None or Trials object from hyperopt) if a Trials object is given, it will continue optimization
                    with this Trials
            max_trials (int) maximum number of trials
        """
        self.x = x_  # .copy().reset_index(drop=True)
        self.y = y_  # .copy().reset_index(drop=True)
        self.x_val = x_val
        self.y_val = y_val
        self.nfolds = nfolds
        self.nfolds_train = nfolds_train
        self.scoring = scoring
        self.cv_strategy = cv_strategy
        # keep an hyperparameters optimization history :
        self.df_all_results = {'mean_fit_time': [], 'params': [], 'mean_test_score': [], 'std_test_score': []}

        if trials is None:
            self.trials = Trials()
        else:
            self.trials = trials

        if self.apply_optimization:
            self.hopt = fmin(fn=self.optimise,
                             space=self.hyper_params,
                             algo=tpe.suggest,
                             max_evals=max_trials,
                             timeout=time_limit_per_model,
                             trials=self.trials,
                             )
        else:
            self.hopt = fmin(fn=self.optimise_no_optimiz,
                             space=self.hyper_params,
                             algo=tpe.suggest,
                             max_evals=1,
                             timeout=time_limit_per_model,
                             trials=self.trials,
                             )

        self.df_all_results = pd.DataFrame(self.df_all_results)
        self.index_best_score = self.df_all_results.mean_test_score.argmax()

    def show_distribution_score(self):
        plt.hist(self.df_all_results.mean_test_score)
        plt.show()

    def search_best_params(self):
        """ Look in history ensemble hyperparameters with best score
        Return:
            params (dict) best parameters from hyperparameters optimization
        """
        return self.df_all_results.loc[self.index_best_score, 'params'].copy()

    def transform_list_stopwords(self, params):
        """ Transform list stop_words in params to boolean type
        Args:
            params (dict)
        """
        if 'vect__text__tfidf__stop_words' in params.keys() and params['vect__text__tfidf__stop_words'] is not None:
            params['vect__text__tfidf__stop_words'] = True
        if 'vect__text__tf__stop_words' in params.keys() and params['vect__text__tf__stop_words'] is not None:
            params['vect__text__tf__stop_words'] = True
        if 'vect__tfidf__stop_words' in params.keys() and params['vect__tfidf__stop_words'] is not None:
            params['vect__tfidf__stop_words'] = True
        if 'vect__tf__stop_words' in params.keys() and params['vect__tf__stop_words'] is not None:
            params['vect__tf__stop_words'] = True
        return params

    def best_params(self):
        """
        Return:
            params (dict) best parameters from hyperparameters optimization
        """
        print_params = self.search_best_params()
        print_params = self.transform_list_stopwords(print_params)
        logger.info('Best parameters: {}'.format(print_params))
        return self.search_best_params()

    def best_score(self):
        """
        Return:
            score (int) : best score from hyperparameters optimization
        """
        score = self.df_all_results.loc[self.index_best_score, 'mean_test_score'].copy()
        logger.info('Mean cross-validated score of the best_estimator: {}'.format(np.round(score, 4)))
        return score

    def best_estimator(self):
        return None

    def get_summary(self, sort_by='mean_test_score'):
        """ Get hyperparameters optimization history
        Return:
            df_all_results (Dataframe)
        """
        df_all_results = self.df_all_results[
            ['mean_fit_time', 'params', 'mean_test_score', 'std_test_score']].sort_values(
            by=sort_by, ascending=False).reset_index(drop=True)
        df_all_results.params = df_all_results.params.apply(lambda d: self.transform_list_stopwords(d))
        return df_all_results


##############################
##############################
##############################

class Optimiz_hyperopt_NN:
    """ Apply Hyperopt optimization for Neural Network Tensorflow model """

    def __init__(self, Model_NN, hyper_params, apply_optimization):
        """
        Args:
            Model_NN (Model class)
            hyper_params (dict) a hyperopt range for each hyperparameters
            apply_optimization (Boolean) if False, initialize random hyperparameters and give a null score
        """
        self.Model_NN = Model_NN
        self.hyper_params = hyper_params
        self.apply_optimization = apply_optimization

    def optimise(self, params):
        """ function to optimize by hyperopt library
        Args :
            params (dict) a hyperopt range for each hyperparameters
        Return:
            score (float) result of metric on validation set
        """

        self.Model_NN.initialize_params(self.y, params)

        logger.info(self.Model_NN.p)

        start = time.perf_counter()

        if self.x_val is None:
            # cross-validation
            fold_id = np.ones((len(self.y),)) * -1
            oof_val = np.zeros((self.y.shape[0], self.y.shape[1]))
            # Cross-validation split in self.nfolds but train only on self.nfolds_train chosen randomly
            rd.seed(self.Model_NN.seed)
            fold_to_train = rd.sample([i for i in range(self.nfolds)], k=max(min(self.nfolds_train, self.nfolds), 1))

            if self.cv_strategy == "StratifiedKFold":
                skf = StratifiedKFold(n_splits=self.nfolds, random_state=self.Model_NN.seed, shuffle=True)
                folds = skf.split(self.y, self.y)
            else:
                kf = KFold(n_splits=self.nfolds, random_state=self.Model_NN.seed, shuffle=True)
                folds = kf.split(self.y)
        else:
            # validation
            fold_id = np.ones((len(self.y_val),)) * -1
            oof_val = np.zeros((self.y_val.shape[0], self.y_val.shape[1]))
            folds = [('all', [i for i in range(oof_val.shape[0])])]
            fold_to_train = [0]

        for n, (tr, te) in enumerate(folds):
            if n not in fold_to_train:
                continue

            if tr == 'all':
                # validation
                if isinstance(self.x, dict):
                    x_tr, x_val = self.x, self.x_val
                    y_tr, y_val = self.y.values, self.y_val.values
                elif isinstance(self.x, list):
                    x_tr, x_val = self.x, self.x_val
                    y_tr, y_val = self.y.values, self.y_val.values
                else:
                    x_tr, x_val = self.x.values, self.x_val.values
                    y_tr, y_val = self.y.values, self.y_val.values
            else:
                # cross-validation
                if isinstance(self.x, dict):
                    x_tr, x_val = {}, {}
                    for col in self.x.keys():
                        x_tr[col], x_val[col] = self.x[col][tr], self.x[col][te]
                    y_tr, y_val = self.y.values[tr], self.y.values[te]
                elif isinstance(self.x, list):
                    x_tr, x_val = [], []
                    for col in range(len(self.x)):
                        x_tr.append(self.x[col][tr])
                        x_val.append(self.x[col][te])
                    y_tr, y_val = self.y.values[tr], self.y.values[te]
                else:
                    x_tr, x_val = self.x.values[tr], self.x.values[te]
                    y_tr, y_val = self.y.values[tr], self.y.values[te]

            model = self.Model_NN.model()

            if 'regression' in self.Model_NN.objective:
                if 'mean_squared_error' in self.scoring:
                    monitor = 'mean_squared_error'
                else:
                    monitor = 'loss'
            else:
                if self.Model_NN.shape_y == 1:
                    if self.scoring == 'accuracy':
                        monitor = 'accuracy'
                    else:
                        monitor = 'loss'
                else:
                    monitor = 'binary_crossentropy'

            rlr = ReduceLROnPlateau(monitor='val_' + monitor, factor=0.1, patience=self.Model_NN.patience - 1,
                                    verbose=0, min_delta=1e-4, mode='auto', min_lr=self.Model_NN.min_lr)

            # ckp = ModelCheckpoint(f'model_{n}.hdf5', monitor = 'val_loss', verbose = 0,
            #                      save_best_only = True, save_weights_only = True, mode = 'min')

            es = EarlyStopping(monitor='val_' + monitor, min_delta=0.0001, patience=self.Model_NN.patience, mode='auto',
                               baseline=None, restore_best_weights=True, verbose=0)

            history = model.fit(x_tr, y_tr, validation_data=(x_val, y_val),
                                epochs=self.Model_NN.epochs, batch_size=self.Model_NN.batch_size,
                                class_weight=compute_dict_class_weight(y_tr, self.Model_NN.class_weight,
                                                                       self.Model_NN.objective),
                                callbacks=[rlr, es], verbose=0)

            hist = pd.DataFrame(history.history)

            if 'regression' in self.Model_NN.objective:
                oof_val[te, :] = model.predict(x_val)
            else:
                if self.Model_NN.shape_y == 1 and 'binary' not in self.Model_NN.objective:
                    oof_val[te, :] = np.argmax(model.predict(x_val), axis=1).reshape(-1, 1)
                else:
                    oof_val[te, :] = np.where(model.predict(x_val) > 0.5, 1, 0)
            fold_id[te] = n
            self.total_epochs += len(history.history['val_loss'][:-(self.Model_NN.patience + 1)])

            K.clear_session()
            del model, history, hist
            d = gc.collect()

        metrics = []
        for i in range(self.Model_NN.shape_y):
            if self.x_val is None:
                # cross_validation
                y_true = self.y.iloc[:, i].copy()
            else:
                # validation
                y_true = self.y_val.iloc[:, i].copy()
            y_true_sample = y_true.values[np.where(fold_id >= 0)[0]]
            prediction_oof_val = oof_val[:, i][np.where(fold_id >= 0)[0]]
            if 'regression' in self.Model_NN.objective:
                if 'explained_variance' == self.scoring:
                    metrics.append(-explained_variance_score(y_true_sample, prediction_oof_val))
                elif 'r2' == self.scoring:
                    metrics.append(-r2_score(y_true_sample, prediction_oof_val))
                else:
                    metrics.append(mean_squared_error(y_true_sample, prediction_oof_val))
            else:
                if 'f1' in self.scoring:
                    if 'binary' in self.Model_NN.objective:
                        metrics.append(-f1_score(y_true_sample, prediction_oof_val))
                    else:
                        metrics.append(
                            -f1_score(y_true_sample, prediction_oof_val, average=self.Model_NN.average_scoring))
                elif 'recall' in self.scoring:
                    if 'binary' in self.Model_NN.objective:
                        metrics.append(-recall_score(y_true_sample, prediction_oof_val))
                    else:
                        metrics.append(
                            -recall_score(y_true_sample, prediction_oof_val, average=self.Model_NN.average_scoring))
                elif 'precision' in self.scoring:
                    if 'binary' in self.Model_NN.objective:
                        metrics.append(-precision_score(y_true_sample, prediction_oof_val))
                    else:
                        metrics.append(
                            -precision_score(y_true_sample, prediction_oof_val, average=self.Model_NN.average_scoring))
                elif 'roc' in self.scoring or 'auc' in self.scoring:
                    if 'binary' in self.Model_NN.objective:
                        metrics.append(-roc_auc_score(y_true_sample, prediction_oof_val))
                    else:
                        metrics.append(
                            -roc_auc_score(y_true_sample, prediction_oof_val, average=self.Model_NN.average_scoring))
                else:
                    metrics.append(-accuracy_score(y_true_sample, prediction_oof_val))

        score = -np.mean(metrics)
        logger.info('oof_val score {} Metric {}'.format(self.scoring, score))

        if 'hidden_units' in self.Model_NN.p.keys():
            self.list_hist[len(self.Model_NN.p['hidden_units']) - 1].append(score)
        else:
            self.list_hist[0].append(score)

        # store hyperparameters optimization in a Dataframe self.df_all_results:
        self.df_all_results['mean_fit_time'].append(time.perf_counter() - start)
        self.df_all_results['params'].append(params)
        self.df_all_results['mean_test_score'].append(score)
        self.df_all_results['std_test_score'].append(0)  # just 0

        return np.mean(metrics)

    def optimise_no_optimiz(self, params):
        """ function to optimize by hyperopt library
            use when apply_optimization is False
            initialize random hyperparameters and give a null score
        Args :
            params (dict) a hyperopt range for each hyperparameters
        Return:
            score (float) result of metric on validation set
        """

        self.Model_NN.initialize_params(self.y, params)

        logger.info(self.Model_NN.p)

        start = time.perf_counter()

        score = 0

        if 'hidden_units' in self.Model_NN.p.keys():
            self.list_hist[len(self.Model_NN.p['hidden_units']) - 1].append(score)
        else:
            self.list_hist[0].append(score)
        self.df_all_results['mean_fit_time'].append(time.perf_counter() - start)
        self.df_all_results['params'].append(params)
        self.df_all_results['mean_test_score'].append(score)
        self.df_all_results['std_test_score'].append(0)  # just 0

        return score

    def train(self, x_, y_, x_val, y_val, nfolds=5, nfolds_train=5, scoring='accuracy', verbose=0,
              time_limit_per_model=60, cv_strategy="StratifiedKFold", trials=None, max_trials=1000,
              apply_mlflow=False, experiment_name="Experiment"):
        """ Compute the function to minimize with hyperopt TPE optimization
            TPE optimization is a Naive Bayes Optimization
        Args:
            x_ (List or dict or DataFrame)
            y_ (Dataframe)
            x_val (List or dict or DataFrame)
            y_val (Dataframe)
            nfolds (int) number of folds to split during optimization
            nfolds_train (int) number of folds to train during optimization
            scoring (str) score to optimize
            verbose (int)
            time_limit_per_model (int) maximum Hyperparameters Optimization time in seconds
            cv_strategy ("StratifiedKFold" or "KFold")
            trials (None or Trials object from hyperopt) if a Trials object is given, it will continue optimization
                            with this Trials
            max_trials (int) maximum number of trials
            apply_mlflow (Boolean)
            experiment_name (str)
        """
        self.x = x_  # .copy().reset_index(drop=True)
        self.y = y_  # .copy().reset_index(drop=True)
        self.x_val = x_val
        self.y_val = y_val
        self.nfolds = nfolds
        self.nfolds_train = nfolds_train
        self.scoring = scoring
        self.cv_strategy = cv_strategy
        self.apply_mlflow = apply_mlflow
        self.df_all_results = {'mean_fit_time': [], 'params': [], 'mean_test_score': [], 'std_test_score': []}
        self.list_hist = [[] for name in self.hyper_params.keys() if 'hidden_unit' in name]
        if len(self.list_hist) == 0:
            self.list_hist = [[]]
        self.total_epochs = 0

        if trials is None:
            self.trials = Trials()
        else:
            self.trials = trials

        if self.apply_optimization:
            self.hopt = fmin(fn=self.optimise,
                             space=self.hyper_params,
                             algo=tpe.suggest,
                             max_evals=max_trials,
                             timeout=time_limit_per_model,
                             trials=self.trials,
                             )
        else:
            self.hopt = fmin(fn=self.optimise_no_optimiz,
                             space=self.hyper_params,
                             algo=tpe.suggest,
                             max_evals=1,
                             timeout=time_limit_per_model,
                             trials=self.trials,
                             )

        self.df_all_results = pd.DataFrame(self.df_all_results)
        self.index_best_score = self.df_all_results.mean_test_score.argmax()
        self.mean_epochs = int(self.total_epochs / self.nfolds) + 1

    def show_distribution_score(self):
        # not used anymore
        rows, cols = 1, 3
        fig, ax = plt.subplots(rows, cols, figsize=(50, 20))

        for row in range(rows):
            for col in range(cols):
                if row * cols + col + 1 <= len(self.list_hist) and len(self.list_hist[row * cols + col]) > 0:
                    ax[col].hist(self.list_hist[row * cols + col])
                    for tick in ax[col].xaxis.get_major_ticks():
                        tick.label.set_fontsize(30)
        plt.show()

    def search_best_params(self):
        """ Look in history ensemble hyperparameters with best score
        Return:
            params (dict) best parameters from hyperparameters optimization
        """
        return self.df_all_results.loc[self.index_best_score, 'params'].copy()

    def best_params(self):
        """
        Return:
            params (dict) best parameters from hyperparameters optimization
        """
        params = self.search_best_params()
        logger.info('Best parameters: {}'.format(params))
        return params

    def best_score(self):
        """
        Return:
            score (int) : best score from hyperparameters optimization
        """
        score = self.df_all_results.loc[self.index_best_score, 'mean_test_score'].copy()
        logger.info('Mean cross-validated score of the best_estimator: {}'.format(np.round(score, 4)))
        return score

    def best_estimator(self):
        return None

    def get_summary(self, sort_by='mean_test_score'):
        """ Get hyperparameters optimization history
        Return:
            df_all_results (Dataframe)
        """
        return self.df_all_results[['mean_fit_time', 'params', 'mean_test_score', 'std_test_score']].sort_values(
            by=sort_by, ascending=False).reset_index(drop=True)


##############################
##############################
##############################

class Optimiz_hyperopt_NN:
    """ Apply Hyperopt optimization for Neural Network Tensorflow model """

    def __init__(self, Model_NN, hyper_params, apply_optimization):
        """
        Args:
            Model_NN (Model class)
            hyper_params (dict) a hyperopt range for each hyperparameters
            apply_optimization (Boolean) if False, initialize random hyperparameters and give a null score
        """
        self.Model_NN = Model_NN
        self.hyper_params = hyper_params
        self.apply_optimization = apply_optimization

    def optimise(self, params):
        """ function to optimize by hyperopt library
        Args :
            params (dict) a hyperopt range for each hyperparameters
        Return:
            score (float) result of metric on validation set
        """

        self.Model_NN.initialize_params(self.y, params)

        logger.info(self.Model_NN.p)

        start = time.perf_counter()

        if self.x_val is None:
            fold_id = np.ones((len(self.y),)) * -1
            oof_val = np.zeros((self.y.shape[0], self.y.shape[1]))
            if "time_series" not in self.Model_NN.objective:
                # cross-validation
                # Cross-validation split in self.nfolds but train only on self.nfolds_train chosen randomly
                rd.seed(self.Model_NN.seed)
                fold_to_train = rd.sample([i for i in range(self.nfolds)], k=max(min(self.nfolds_train, self.nfolds), 1))

                if self.cv_strategy == "StratifiedKFold":
                    skf = StratifiedKFold(n_splits=self.nfolds, random_state=self.Model_NN.seed, shuffle=True)
                    folds = skf.split(self.y, self.y)
                else:
                    kf = KFold(n_splits=self.nfolds, random_state=self.Model_NN.seed, shuffle=True)
                    folds = kf.split(self.y)
            else:
                folds = [('all', [i for i in range(self.Model_NN.size_train)])]
                fold_to_train = [0]

        else:
            # validation
            fold_id = np.ones((len(self.y_val),)) * -1
            oof_val = np.zeros((self.y_val.shape[0], self.y_val.shape[1]))
            folds = [('all', [i for i in range(oof_val.shape[0])])]
            fold_to_train = [0]

        for n, (tr, te) in enumerate(folds):
            if n not in fold_to_train:
                continue

            if tr == 'all':
                # validation
                if isinstance(self.x, dict):
                    x_tr, x_val = self.x, self.x_val
                    y_tr, y_val = self.y.values, self.y_val.values
                elif isinstance(self.x, list):
                    x_tr, x_val = self.x, self.x_val
                    y_tr, y_val = self.y.values, self.y_val.values
                else:
                    x_tr, x_val = self.x.values, self.x_val.values
                    y_tr, y_val = self.y.values, self.y_val.values
            else:
                # cross-validation
                if isinstance(self.x, dict):
                    x_tr, x_val = {}, {}
                    for col in self.x.keys():
                        x_tr[col], x_val[col] = self.x[col][tr], self.x[col][te]
                    y_tr, y_val = self.y.values[tr], self.y.values[te]
                elif isinstance(self.x, list):
                    x_tr, x_val = [], []
                    for col in range(len(self.x)):
                        x_tr.append(self.x[col][tr])
                        x_val.append(self.x[col][te])
                    y_tr, y_val = self.y.values[tr], self.y.values[te]
                else:
                    x_tr, x_val = self.x.values[tr], self.x.values[te]
                    y_tr, y_val = self.y.values[tr], self.y.values[te]

            model = self.Model_NN.model()

            if 'regression' in self.Model_NN.objective:
                if 'mean_squared_error' in self.scoring:
                    monitor = 'mean_squared_error'
                else:
                    monitor = 'loss'
            else:
                if self.Model_NN.shape_y == 1:
                    if self.scoring == 'accuracy':
                        monitor = 'accuracy'
                    else:
                        monitor = 'loss'
                else:
                    monitor = 'binary_crossentropy'

            rlr = ReduceLROnPlateau(monitor='val_' + monitor, factor=0.1, patience=self.Model_NN.patience - 1,
                                    verbose=0, min_delta=1e-4, mode='auto', min_lr=self.Model_NN.min_lr)

            # ckp = ModelCheckpoint(f'model_{n}.hdf5', monitor = 'val_loss', verbose = 0,
            #                      save_best_only = True, save_weights_only = True, mode = 'min')

            es = EarlyStopping(monitor='val_' + monitor, min_delta=0.0001, patience=self.Model_NN.patience, mode='auto',
                               baseline=None, restore_best_weights=True, verbose=0)

            history = model.fit(x_tr, y_tr, validation_data=(x_val, y_val),
                                epochs=self.Model_NN.epochs, batch_size=self.Model_NN.batch_size,
                                class_weight=compute_dict_class_weight(y_tr, self.Model_NN.class_weight,
                                                                       self.Model_NN.objective),
                                callbacks=[rlr, es], verbose=0)

            hist = pd.DataFrame(history.history)

            if 'regression' in self.Model_NN.objective:
                oof_val[te, :] = model.predict(x_val)
            else:
                if self.Model_NN.shape_y == 1 and 'binary' not in self.Model_NN.objective:
                    oof_val[te, :] = np.argmax(model.predict(x_val), axis=1).reshape(-1, 1)
                else:
                    oof_val[te, :] = np.where(model.predict(x_val) > 0.5, 1, 0)
            fold_id[te] = n
            self.total_epochs += len(history.history['val_loss'][:-(self.Model_NN.patience + 1)])

            K.clear_session()
            del model, history, hist
            d = gc.collect()

        metrics = []
        for i in range(self.Model_NN.shape_y):
            if self.x_val is None:
                # cross_validation
                y_true = self.y.iloc[:, i].copy()
            else:
                # validation
                y_true = self.y_val.iloc[:, i].copy()
            y_true_sample = y_true.values[np.where(fold_id >= 0)[0]]
            prediction_oof_val = oof_val[:, i][np.where(fold_id >= 0)[0]]
            if 'regression' in self.Model_NN.objective:
                if 'explained_variance' == self.scoring:
                    metrics.append(-explained_variance_score(y_true_sample, prediction_oof_val))
                elif 'r2' == self.scoring:
                    metrics.append(-r2_score(y_true_sample, prediction_oof_val))
                else:
                    metrics.append(mean_squared_error(y_true_sample, prediction_oof_val))
            else:
                if 'f1' in self.scoring:
                    if 'binary' in self.Model_NN.objective:
                        metrics.append(-f1_score(y_true_sample, prediction_oof_val))
                    else:
                        metrics.append(
                            -f1_score(y_true_sample, prediction_oof_val, average=self.Model_NN.average_scoring))
                elif 'recall' in self.scoring:
                    if 'binary' in self.Model_NN.objective:
                        metrics.append(-recall_score(y_true_sample, prediction_oof_val))
                    else:
                        metrics.append(
                            -recall_score(y_true_sample, prediction_oof_val, average=self.Model_NN.average_scoring))
                elif 'precision' in self.scoring:
                    if 'binary' in self.Model_NN.objective:
                        metrics.append(-precision_score(y_true_sample, prediction_oof_val))
                    else:
                        metrics.append(
                            -precision_score(y_true_sample, prediction_oof_val, average=self.Model_NN.average_scoring))
                elif 'roc' in self.scoring or 'auc' in self.scoring:
                    if 'binary' in self.Model_NN.objective:
                        metrics.append(-roc_auc_score(y_true_sample, prediction_oof_val))
                    else:
                        metrics.append(
                            -roc_auc_score(y_true_sample, prediction_oof_val, average=self.Model_NN.average_scoring))
                else:
                    metrics.append(-accuracy_score(y_true_sample, prediction_oof_val))

        score = -np.mean(metrics)
        logger.info('oof_val score {} Metric {}'.format(self.scoring, score))

        if 'hidden_units' in self.Model_NN.p.keys():
            self.list_hist[len(self.Model_NN.p['hidden_units']) - 1].append(score)
        else:
            self.list_hist[0].append(score)

        # store hyperparameters optimization in a Dataframe self.df_all_results:
        self.df_all_results['mean_fit_time'].append(time.perf_counter() - start)
        self.df_all_results['params'].append(params)
        self.df_all_results['mean_test_score'].append(score)
        self.df_all_results['std_test_score'].append(0)  # just 0

        return np.mean(metrics)

    def optimise_no_optimiz(self, params):
        """ function to optimize by hyperopt library
            use when apply_optimization is False
            initialize random hyperparameters and give a null score
        Args :
            params (dict) a hyperopt range for each hyperparameters
        Return:
            score (float) result of metric on validation set
        """

        self.Model_NN.initialize_params(self.y, params)

        logger.info(self.Model_NN.p)

        start = time.perf_counter()

        score = 0

        if 'hidden_units' in self.Model_NN.p.keys():
            self.list_hist[len(self.Model_NN.p['hidden_units']) - 1].append(score)
        else:
            self.list_hist[0].append(score)
        self.df_all_results['mean_fit_time'].append(time.perf_counter() - start)
        self.df_all_results['params'].append(params)
        self.df_all_results['mean_test_score'].append(score)
        self.df_all_results['std_test_score'].append(0)  # just 0

        return score

    def train(self, x_, y_, x_val, y_val, nfolds=5, nfolds_train=5, scoring='accuracy', verbose=0,
              time_limit_per_model=60, cv_strategy="StratifiedKFold", trials=None, max_trials=1000,
              apply_mlflow=False, experiment_name="Experiment"):
        """ Compute the function to minimize with hyperopt TPE optimization
            TPE optimization is a Naive Bayes Optimization
        Args:
            x_ (List or dict or DataFrame)
            y_ (Dataframe)
            x_val (List or dict or DataFrame)
            y_val (Dataframe)
            nfolds (int) number of folds to split during optimization
            nfolds_train (int) number of folds to train during optimization
            scoring (str) score to optimize
            verbose (int)
            time_limit_per_model (int) maximum Hyperparameters Optimization time in seconds
            cv_strategy ("StratifiedKFold" or "KFold")
            trials (None or Trials object from hyperopt) if a Trials object is given, it will continue optimization
                            with this Trials
            max_trials (int) maximum number of trials
            apply_mlflow (Boolean)
            experiment_name (str)
        """
        self.x = x_  # .copy().reset_index(drop=True)
        self.y = y_  # .copy().reset_index(drop=True)
        self.x_val = x_val
        self.y_val = y_val
        self.nfolds = nfolds
        self.nfolds_train = nfolds_train
        self.scoring = scoring
        self.cv_strategy = cv_strategy
        self.apply_mlflow = apply_mlflow
        self.df_all_results = {'mean_fit_time': [], 'params': [], 'mean_test_score': [], 'std_test_score': []}
        self.list_hist = [[] for name in self.hyper_params.keys() if 'hidden_unit' in name]
        if len(self.list_hist) == 0:
            self.list_hist = [[]]
        self.total_epochs = 0

        if trials is None:
            self.trials = Trials()
        else:
            self.trials = trials

        if self.apply_optimization:
            self.hopt = fmin(fn=self.optimise,
                             space=self.hyper_params,
                             algo=tpe.suggest,
                             max_evals=max_trials,
                             timeout=time_limit_per_model,
                             trials=self.trials,
                             )
        else:
            self.hopt = fmin(fn=self.optimise_no_optimiz,
                             space=self.hyper_params,
                             algo=tpe.suggest,
                             max_evals=1,
                             timeout=time_limit_per_model,
                             trials=self.trials,
                             )

        self.df_all_results = pd.DataFrame(self.df_all_results)
        self.index_best_score = self.df_all_results.mean_test_score.argmax()
        self.mean_epochs = int(self.total_epochs / self.nfolds) + 1

    def show_distribution_score(self):
        # not used anymore
        rows, cols = 1, 3
        fig, ax = plt.subplots(rows, cols, figsize=(50, 20))

        for row in range(rows):
            for col in range(cols):
                if row * cols + col + 1 <= len(self.list_hist) and len(self.list_hist[row * cols + col]) > 0:
                    ax[col].hist(self.list_hist[row * cols + col])
                    for tick in ax[col].xaxis.get_major_ticks():
                        tick.label.set_fontsize(30)
        plt.show()

    def search_best_params(self):
        """ Look in history ensemble hyperparameters with best score
        Return:
            params (dict) best parameters from hyperparameters optimization
        """
        return self.df_all_results.loc[self.index_best_score, 'params'].copy()

    def best_params(self):
        """
        Return:
            params (dict) best parameters from hyperparameters optimization
        """
        params = self.search_best_params()
        logger.info('Best parameters: {}'.format(params))
        return params

    def best_score(self):
        """
        Return:
            score (int) : best score from hyperparameters optimization
        """
        score = self.df_all_results.loc[self.index_best_score, 'mean_test_score'].copy()
        logger.info('Mean cross-validated score of the best_estimator: {}'.format(np.round(score, 4)))
        return score

    def best_estimator(self):
        return None

    def get_summary(self, sort_by='mean_test_score'):
        """ Get hyperparameters optimization history
        Return:
            df_all_results (Dataframe)
        """
        return self.df_all_results[['mean_fit_time', 'params', 'mean_test_score', 'std_test_score']].sort_values(
            by=sort_by, ascending=False).reset_index(drop=True)
