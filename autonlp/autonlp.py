import numpy as np
import pandas as pd
import plotly.express as px
# conda install -c conda-forge python-kaleido
# pip install kaleido

import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
from glob import glob
from shutil import rmtree, copytree, copyfile
from matplotlib.ticker import FormatStrFormatter
from .flags import save_yaml, load_yaml
import dataclasses
from joblib import load, dump
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .features.cleaner import Preprocessing

from .features.embeddings.tf_embedding import Tf_embedding
from .features.embeddings.tfidf import Tfidf
from .features.embeddings.word2vec import Word2Vec
from .features.embeddings.fastText import Fasttext
from .features.embeddings.doc2vec import Doc2Vec
from .features.embeddings.transformernlp import TransformerNLP
from .utils.extraction_words import extract_influent_word, get_top_influent_word
from .utils.metrics import build_df_confusion_matrix

from .models.classifier_nlp.naive_bayes import Naive_Bayes
from .models.classifier_nlp.logistic_regression import Logistic_Regression
from .models.classifier_nlp.sgd_classifier import SGD_Classifier
from .models.classifier_nlp.sgd_regressor import SGD_Regressor
from .models.classifier_nlp.global_average import Global_Average
from .models.classifier_nlp.attention import Attention
from .models.classifier_nlp.birnn import Birnn
from .models.classifier_nlp.birnn_attention import Birnn_Attention
from .models.classifier_nlp.bilstm import Bilstm
from .models.classifier_nlp.bilstm_attention import Bilstm_Attention
from .models.classifier_nlp.bigru import Bigru
from .models.classifier_nlp.bigru_attention import Bigru_Attention
from .models.classifier_nlp.xgboost_tree import XGBoost
from .models.classifier_nlp.blend_models import BlendModel

from .models.classifier.logistic_regression import ML_Logistic_Regression
from .models.classifier.randomforest_classifier import ML_RandomForest
from .models.classifier.lightgbm import ML_LightGBM
from .models.classifier.xgboost import ML_XGBoost
from .models.classifier.catboost import ML_CatBoost
from .models.classifier.dense_network import ML_DenseNetwork

from .models.embeddings.trainer import Embedding

import logging
from .utils.logging import get_logger, verbosity_to_loglevel

logger = get_logger(__name__)


# logging.getLogger().setLevel(verbosity_to_loglevel(2))


class AutoNLP:
    """Class for compile full pipeline of AutoNLP task.
        AutoNLP steps:
            - Preprocessing data
              :class:Preprocessing_NLP
            - Create a test set: split data
            - Compute each NLP model :
                    - Optimization hyperparameters
                    - Validation/Cross-validation with best hyperparameters
                    - Save models from cross-validation
            - Blend all NLP models
            - Returns prediction/metric results on validation data.
            - Returns prediction/metric results on test data.
        Example:
            Common usecase - create custom AutoNLP:
            flags = Flags()
            data = pd.read_csv(flags.path_data)
            autonlp = AutoNLP(flags)
            autonlp.data_preprocessing(data)
            autonlp.train()
            autonlp.ensemble()
            autonlp.leader_predict()
        """

    def __init__(self, flags_parameters):
        """
        Args:
            flags_parameters : Instance of Flags class object

        From flags_parameters:
        include_model (list) : name of models to include
        column_text (str) : name of the column with texts (only one column)
        apply_mlflow (Boolean) : use MLflow Tracking
        experiment_name (str) : name of MLflow experiment
        apply_logs (Boolean) : use manual logs
        max_run_time_per_model (int) : maximum Hyperparameters Optimization time in seconds for each model
        frac_trainset (float) : (]0;1]) fraction of dataset to use for train set
        seed (int) : seed used for train/test split, cross-validation split and fold choice
        debug (Boolean) : if True use only 50 data rows for training
        class_weight (Boolean) : apply a weight for each class
        scoring (str) : metric optimized during optimization
        verbose (int) : verbose for optimization
        objective (str) : 'binary' or 'multi-class' or 'regression' / specify target objective
        size_params (str) : 'small' or 'big', size of parameters range for optimization
        apply_optimization (Boolean) : if True apply Hyperparameters Optimization else load models from path
                                           flags_parameters.path_models_parameters
        apply_validation (Boolean) : apply validation / cross-validation
        apply_blend_model (Boolean) : apply blend of all NLP models
        method_embedding (dict) : information about embedding method
                e.g : {'Word2vec': 'Word2Vec',
                       'Fasttext': 'FastText',
                       'Doc2Vec': 'Doc2Vec',
                       'Transformer': 'CamemBERT',
                       'spacy': [('all', False), (['ADJ', 'NOUN', 'VERB', 'DET'], False),
                                 (['ADJ', 'NOUN', 'VERB', 'DET'], True)]}
        outdir (str) : path of output logs

        Note:
        There are several verbosity levels:
            - `0`: No messages.
            - `1`: Warnings.
            - `2`: Info.
            - `3`: Debug.
        """
        self.flags_parameters = flags_parameters
        self.include_model = flags_parameters.include_model
        self.ordinal_features = flags_parameters.ordinal_features
        self.normalize = flags_parameters.normalize
        self.column_text = flags_parameters.column_text
        self.apply_mlflow = flags_parameters.apply_mlflow
        self.experiment_name = flags_parameters.experiment_name
        self.apply_logs = flags_parameters.apply_logs
        self.apply_app = flags_parameters.apply_app
        self.max_run_time_per_model = flags_parameters.max_run_time_per_model
        self.frac_trainset = flags_parameters.frac_trainset
        self.seed = flags_parameters.seed
        self.debug = flags_parameters.debug
        if not flags_parameters.class_weight:
            self.class_weight = None
        else:
            self.class_weight = "balanced"
        self.scoring = flags_parameters.scoring
        self.verbose = flags_parameters.verbose
        logging.getLogger().setLevel(verbosity_to_loglevel(self.verbose))
        self.objective = flags_parameters.objective
        self.size_params = flags_parameters.size_params
        self.apply_optimization = flags_parameters.apply_optimization
        self.apply_validation = flags_parameters.apply_validation
        self.apply_blend_model = flags_parameters.apply_blend_model
        self.method_embedding = {k.lower(): v for k, v in flags_parameters.method_embedding.items()}

        self.position_id = flags_parameters.position_id
        self.position_date = flags_parameters.position_date
        self.size_train_prc = flags_parameters.size_train_prc
        self.time_series_recursive = flags_parameters.time_series_recursive
        self.LSTM_date_features = flags_parameters.LSTM_date_features

        self.outdir = self.flags_parameters.outdir
        flags_dict = dataclasses.asdict(self.flags_parameters)

        if self.apply_app:
            if os.path.exists("./model_deployment"):
                self.apply_logs, self.apply_mlflow = False, False
                self.flags_parameters.apply_logs, self.flags_parameters.apply_mlflow = False, False
                self.outdir = "./model_deployment"
                self.flags_parameters.outdir = self.outdir
            else:
                self.apply_app = False

        if self.apply_logs:
            os.makedirs(self.outdir, exist_ok=True)
            save_yaml(os.path.join(self.outdir, "flags.yaml"), flags_dict)

        if self.apply_mlflow:
            import mlflow
            from mlflow.tracking import MlflowClient
            # CREATE EXPERIMENT :
            self.path_mlflow = "./mlruns"
            client = MlflowClient()
            try:
                mlflow.set_experiment(experiment_name=self.experiment_name)
            except:
                current_experiment = dict(mlflow.get_experiment_by_name(self.experiment_name))
                experiment_id = current_experiment['experiment_id']
                client.restore_experiment(experiment_id)
            current_experiment = dict(mlflow.get_experiment_by_name(self.experiment_name))
            self.experiment_id = current_experiment['experiment_id']

            # ADD FLAGS PARAMETERS to meta.yaml of MLflow experiment
            save_yaml(os.path.join(self.path_mlflow, self.experiment_id, "flags.yaml"), flags_dict)

        self.pre = None
        self.models = {}  # variable to store all models from :class: Model
        self.embeddings = {}  # variable to store all embeddings from :class: Embedding
        self.info_models = {}
        self.info_scores = {}  # variable to store all information scores

        # Assert parameters
        assert self.flags_parameters.objective in ['binary', 'multi-class',
                                                   'regression'], "Possible objective are 'binary', 'multi-class' or 'regression'"
        assert self.flags_parameters.language_text in ['fr', 'en'], "Possible language are 'fr' and 'en'"
        assert self.flags_parameters.cv_strategy in ['StratifiedKFold',
                                                     'KFold'], "Only StratifiedKFold and Kfold cv_strategy are implemented"
        assert self.frac_trainset <= 1, "frac_trainset must be inferior or equal to 1"

        assert self.apply_mlflow is True or self.apply_logs is True or self.apply_app is True, "at least apply_mlflow or apply_logs or apply_app must be True"

        for lg in ['fr', 'en']:
            if self.flags_parameters.language_text == lg and lg + '_core' not in self.flags_parameters.name_spacy_model:
                logger.warning("\nYou set language '{}' but {} is not a spacy french model".format(lg,
                                                                                                   self.flags_parameters.name_spacy_model))

        if self.flags_parameters.objective == 'binary':
            if self.flags_parameters.scoring not in ['accuracy', 'f1', 'recall', 'precision', 'roc_auc']:
                logger.warning("\nYou probably set a scoring metric not adapted for binary classification")

        if self.flags_parameters.objective == 'multi-class':
            if self.flags_parameters.scoring not in ['accuracy', 'f1', 'recall', 'precision']:
                logger.warning("\nYou probably set a scoring metric not adapted for multi-class classification")

        if self.flags_parameters.objective == 'regression':
            if self.flags_parameters.scoring not in ['mse', 'explained_variance', 'r2']:
                logger.warning("\nYou probably set a scoring metric not adapted for regression classification")

    def split_data(self):
        """ split self.data and build X_train, X_test, Y_train, Y_test
            if doc_spacy_data (documents preprocessed by spacy) is not None :
                - build doc_spacy_data_train and doc_spacy_data_test """
        # DEBUG
        if self.debug:
            logger.info("\n DEBUG MODE : only a small portion is used for training set")
            train_data = self.data.sample(n=min(50, len(self.data)), random_state=self.seed)
        else:
            train_data = self.data.sample(frac=self.frac_trainset, random_state=self.seed)

        # Train set
        self.X_train = train_data.copy()
        logger.info("\nTraining set size : {}".format(len(self.X_train)))
        if self.Y is not None:
            self.Y_train = self.Y.loc[train_data.index, :]
        else:
            self.Y_train = None
        if self.doc_spacy_data is not None:
            self.doc_spacy_data_train = np.array(self.doc_spacy_data)[list(train_data.index)]
        else:
            self.doc_spacy_data_train = None

        # Test set
        if self.frac_trainset < 1:
            test_data = self.data.drop(train_data.index)
            self.X_test = test_data.copy()
            if self.Y is not None:
                self.Y_test = self.Y.drop(train_data.index)
            else:
                self.Y_test = None
            if self.doc_spacy_data is not None:
                self.doc_spacy_data_test = np.array(self.doc_spacy_data)[
                    [i for i in range(len(self.doc_spacy_data)) if i not in list(train_data.index)]]
            else:
                self.doc_spacy_data_test = None
            logger.info("Test set size : {}".format(len(self.X_test)))
        else:
            logger.info("Test set size : 0")

    def split_data_ts(self, startDate_train, endDate_train):

        # DEBUG
        if self.debug:
            logger.info("\n DEBUG MODE : only a small dataset portion is used")
            self.data = self.data.sample(n=min(50, len(self.data)), random_state=self.seed)

        if self.data.shape[1] == 0:  # can't do tabular prediction (case if only column text)
            self.exclude_model = ['LightGBM', 'XGBoost', 'SimpleNeuralNetwork']
            if startDate_train == 'all' and endDate_train == 'all':
                self.Y_train = self.Y.copy()
            elif startDate_train == 'all':
                self.Y_train = self.Y.loc[:endDate_train, :].copy()
            elif endDate_train == 'all':
                self.Y_train = self.Y.loc[startDate_train:, :].copy()
            else:
                self.Y_train = self.Y.loc[startDate_train:endDate_train, :].copy()
            if endDate_train != 'all' and endDate_train != self.Y.index[-1]:
                self.Y_test = self.Y.loc[endDate_train:, :].copy()

        else:
            if startDate_train == 'all' and endDate_train == 'all':
                self.X_train = self.data
            elif startDate_train == 'all':
                self.X_train = self.data[self.data[self.position_date] <= endDate_train]
            elif endDate_train == 'all':
                self.X_train = self.data[self.data[self.position_date] >= startDate_train]
            else:
                self.X_train = self.data[(self.data[self.position_date] >= startDate_train) & (
                            self.data[self.position_date] <= endDate_train)]

            if self.doc_spacy_data is not None:
                self.doc_spacy_data_train = np.array(self.doc_spacy_data)[list(self.X_train.index)]
            else:
                self.doc_spacy_data_train = None

            if endDate_train != 'all' and endDate_train != np.max(self.data[self.position_date]):
                self.X_test = self.data[self.data[self.position_date] > endDate_train]
                if self.doc_spacy_data is not None:
                    self.doc_spacy_data_test = np.array(self.doc_spacy_data)[list(self.X_test.index)]
                else:
                    self.doc_spacy_data_test = None
            else:
                self.doc_spacy_data_test = None

            #del self.data

            self.Y_train = self.Y.loc[self.X_train.index, :]
            try:
                self.Y_test = self.Y.loc[self.X_test.index, :]
            except:
                pass

    def fit_transform_normalize_data(self):
        self.features = self.X_train.columns.values
        if self.flags_parameters.method_scaling == 'MinMaxScaler':
            self.scaler = MinMaxScaler(feature_range=(0, 1), copy=False)  # or (-1,1)
        elif self.flags_parameters.method_scaling == 'RobustScaler':
            self.scaler = RobustScaler(copy=False)
        else:
            self.scaler = StandardScaler(copy=False)

        self.column_to_normalize = [col for col in self.features if
                                    col not in self.ordinal_features + [self.pre.column_text]]  # from pre because int

        if len(self.column_to_normalize) > 0:
            self.scaler.fit(self.X_train[self.column_to_normalize])

            dump(self.scaler, os.path.join(self.outdir, "scaler.pkl"))
            self.scaler_info = [self.scaler, self.column_to_normalize].copy()

            #for col in self.column_to_normalize:
            #    self.scaler.fit(self.X_train[[col]])
            #    a = self.X_train[[col]].values
            #    self.X_train[[col]] = self.scaler.transform(a)
            #    del a
            #    try:
            #        a = self.X_test[[col]].values
            #        self.X_test[[col]] = self.scaler.transform(a)
            #        del a
            #    except:
            #        pass

            ### take a lot of memory to do it all together !
            self.X_train[self.column_to_normalize] = self.scaler.transform(self.X_train[self.column_to_normalize].values)
            try:
                self.X_val[self.column_to_normalize] = self.scaler.transform(self.X_val[self.column_to_normalize].values)
            except:
                pass
            try:
                self.X_test[self.column_to_normalize] = self.scaler.transform(self.X_test[self.column_to_normalize].values)
            except:
                pass
        else:
            self.scaler_info = None

    def transform_normalize_data(self, X):
        self.features = X.columns.values

        self.column_to_normalize = [col for col in self.features if
                                    col not in self.ordinal_features + [self.pre.column_text]]  # from pre because int

        if len(self.column_to_normalize) > 0:
            try:
                scaler = self.scaler
            except:
                scaler = load(os.path.join(self.outdir, "scaler.pkl"))
            self.scaler_info = [scaler, self.column_to_normalize].copy()

            ### take a lot of memory to do it all together !
            X[self.column_to_normalize] = scaler.transform(X[self.column_to_normalize].values)
        else:
            self.scaler_info = None
        return X

    def data_preprocessing(self, data=None):
        """ Apply :class:Preprocessing_NLP from preprocessing_nlp.py
        Args :
            data (Dataframe)
        """
        # Read data
        if data is None:
            logger.info("\nRead data...")
            data = pd.read_csv(self.flags_parameters.path_data)

        # Preprocessing
        logger.info("\nBegin preprocessing of {} data :".format(len(data)))
        self.pre = Preprocessing(data, self.flags_parameters)
        self.data, self.doc_spacy_data = self.pre.fit_transform(data)

        # WARNING : self.column_text (int) is now the column number of self.column_text (str) in self.data
        if self.pre.column_text is not None:
            self.column_text = list(self.data.columns).index(self.pre.column_text)
        else:
            self.column_text = None

        # del data

        # update and save flags if map_label has been created during preprocessing
        if self.pre.map_label != self.flags_parameters.map_label:
            self.flags_parameters.map_label = self.pre.map_label
            flags_dict = dataclasses.asdict(self.flags_parameters)
            if self.apply_logs:
                save_yaml(os.path.join(self.outdir, "flags.yaml"), flags_dict)
            if self.apply_mlflow:
                save_yaml(os.path.join(self.path_mlflow, self.experiment_id, "flags.yaml"), flags_dict)

        self.target = self.pre.target
        self.Y = self.pre.Y

        self.ordinal_features = [col for col in self.ordinal_features if col in self.data.columns]
        self.LSTM_date_features = [col for col in self.LSTM_date_features if col in self.data.columns]
        self.len_unique_value = {}
        for col in list(set(self.ordinal_features + self.LSTM_date_features)):
            self.len_unique_value[col] = len(self.data[col].unique())

        # Cross-validation :
        if self.flags_parameters.path_data_validation == '' or self.flags_parameters.path_data_validation is None:
            self.Y_val = None
            self.X_val = None
            self.doc_spacy_data_val = None

            # split self.data
            if 'time_series' in self.objective:
                self.split_data_ts(self.flags_parameters.startDate_train, self.flags_parameters.endDate_train)
                self.time_series_features = (
                self.pre.step_lags, self.pre.step_rolling, self.pre.win_type, self.position_id, self.position_date)
            else:
                self.split_data()
                self.time_series_features = None

        # Validation
        # use a loaded validation dataset :
        else:
            # all rows from data will be training rows
            if 'time_series' in self.objective:
                self.split_data_ts("all", "all")
                self.time_series_features = (
                    self.pre.step_lags, self.pre.step_rolling, self.pre.win_type, self.position_id, self.position_date)
            else:
                self.frac_trainset = 1
                self.split_data()
            dataset_val = pd.read_csv(self.flags_parameters.path_data_validation)

            if len([col for col in self.target if col in dataset_val.columns]) > 0:
                self.Y_val = dataset_val[[col for col in self.target if col in dataset_val.columns]]
                for col in self.target:
                    if col in dataset_val.columns:
                        dataset_val = dataset_val.drop([col], axis=1)
            else:
                self.Y_val = None

            self.X_val, self.doc_spacy_data_val = self.preprocess_test_data(dataset_val)

            if self.Y_val is not None:
                assert self.Y_val.shape[1] > 0, 'target specifying the column with labels to predict is not in data_val'

            # use label map if labels are not numerics
            if self.Y_val.shape[1] == 1 and self.flags_parameters.map_label != {}:
                if self.Y_val[self.Y_val.columns[0]].iloc[0] in self.flags_parameters.map_label.keys():
                    self.Y_val[self.Y_val.columns[0]] = self.Y_val[self.Y_val.columns[0]].map(
                        self.flags_parameters.map_label)
                    if self.Y_val[self.Y_val.columns[0]].isnull().sum() > 0:
                        logger.error("Unknown label name during map of test labels")
            elif self.Y_val.shape[1] > 1 and self.flags_parameters.map_label != {}:
                for i in range(self.Y_val.shape[1]):
                    if i in self.flags_parameters.map_label.keys() and self.Y_val[self.Y_val.columns[i]].iloc[0] in self.flags_parameters.map_label.keys():
                        self.Y_val[self.Y_val.columns[i]] = self.Y_val[self.Y_val.columns[i]].map(self.flags_parameters.map_label)

            self.X_test = self.X_val.copy()
            self.Y_test = self.Y_val.copy()
            self.doc_spacy_data_test = self.doc_spacy_data_val

        if self.normalize:
            self.fit_transform_normalize_data()
        else:
            self.scaler_info = None

        self.apply_autonlp = False
        if self.column_text is not None:
            if len(self.X_train.columns) == 1:
                self.apply_autonlp = True

    def preprocess_test_data(self, data_test):
        """ Apply same transformation as in the function self.data_preprocessing for data_test
        Args:
            data_test (str, list, dict, dataframe)
        Returns:
            data_test (Dataframe)
            doc_spacy_data_test (List) : documents from data_test preprocessed by spacy nlp
        """
        if isinstance(data_test, str):
            data_test = pd.DataFrame({self.flags_parameters.column_text: [data_test]})
        elif isinstance(data_test, list):
            data_test = pd.DataFrame({self.flags_parameters.column_text: data_test})
        elif isinstance(data_test, dict):
            data_test = pd.DataFrame(data_test)

        if self.pre is not None:
            if len([col for col in self.target if col in data_test.columns]) > 0:
                y_test = data_test[[col for col in self.target if col in data_test.columns]]

                if y_test.shape[1] == 1 and self.flags_parameters.map_label != {}:
                    if y_test[y_test.columns[0]].iloc[0] in self.flags_parameters.map_label.keys():
                        y_test[y_test.columns[0]] = y_test[y_test.columns[0]].map(self.flags_parameters.map_label)
                elif y_test.shape[1] > 1 and self.flags_parameters.map_label != {}:
                    for i in range(y_test.shape[1]):
                        if i in self.flags_parameters.map_label.keys() and y_test[y_test.columns[i]].iloc[0] in self.flags_parameters.map_label.keys():
                            y_test[y_test.columns[i]] = y_test[y_test.columns[i]].map(self.flags_parameters.map_label)
            else:
                y_test = None
            data_test, doc_spacy_data_test = self.pre.transform(data_test)
        else:
            self.pre = Preprocessing(data_test, self.flags_parameters)
            self.pre.load_parameters()
            data_test, doc_spacy_data_test = self.pre.transform(data_test)
            if self.pre.column_text is not None:
                self.column_text = list(data_test.columns).index(self.pre.column_text)
            else:
                self.column_text = None
            self.target = self.pre.target
            y_test = self.pre.Y

        if self.normalize:
            data_test = self.transform_normalize_data(data_test)
        else:
            self.scaler_info = None

        self.apply_autonlp = False
        if self.column_text is not None:
            if len(data_test.columns) == 1:
                self.apply_autonlp = True

        if y_test is None:
            return data_test, doc_spacy_data_test
        else:
            return data_test, doc_spacy_data_test, y_test

    def prepare_model(self, x=None, y=None, x_val=None, y_val=None):
        """ Compute each NLP model :
                    - Optimization hyperparameters
                    - Validation/Cross-validation with best hyperparameters
                    - Save models from cross-validation
        Args:
              x (Dataframe)
              y (Dataframe)
        """

        # if x and y are None use self.X_train and self.Y_train else use x and y :
        if x is not None:
            self.x_train = x
        else:
            self.x_train = self.X_train
        if y is not None:
            self.y_train = y
        else:
            self.y_train = self.Y_train
        if x_val is not None:
            self.x_val = x_val
        else:
            if self.X_val is not None:
                self.x_val = self.X_val
            else:
                self.x_val = None
        if y_val is not None:
            self.y_val = y_val
        else:
            if self.Y_val is not None:
                self.y_val = self.Y_val
            else:
                self.y_val = None

        if self.y_train is not None:
            assert isinstance(self.y_train, pd.DataFrame), "y/self.y_train must be a DataFrame type"

        self.name_models = []
        self.name_classifiers = []
        self.class_embeddings = []

        # if exist column text : need name model in embedding+classifier_nlp format
        if self.apply_autonlp:
            for name_model in self.include_model:
                name_split = name_model.split("+")

                # TF or TF-IDF embeddings :
                if name_split[0].lower() in ['tf-idf', 'tf']:
                    if 'spacy' not in self.method_embedding.keys() or self.method_embedding[
                        'spacy'] == [] or not self.flags_parameters.apply_spacy_preprocessing:
                        if len(name_split) > 1:
                            self.name_classifiers.append(''.join([i for i in name_split[1] if not i.isdigit()]))
                        self.name_models.append(name_model)
                        if name_split[0].lower() == 'tf-idf':
                            self.class_embeddings.append(Tfidf)
                        else:
                            self.class_embeddings.append(Tf_embedding)
                        self.method_embedding[name_model] = ('all', False)
                    else:
                        for (keep_pos_tag, lemmatize) in self.method_embedding['spacy']:
                            if len(name_split) > 1:
                                self.name_classifiers.append(''.join([i for i in name_split[1] if not i.isdigit()]))
                            if keep_pos_tag == 'all':
                                if lemmatize == True:
                                    self.name_models.append(name_model + '_lem')
                                    self.method_embedding[name_model + '_lem'] = (keep_pos_tag, lemmatize)
                                else:
                                    self.name_models.append(name_model)
                                    self.method_embedding[name_model] = (keep_pos_tag, lemmatize)

                            else:
                                if lemmatize == True:
                                    self.name_models.append(name_model + '_' + "_".join(keep_pos_tag) + '_lem')
                                    self.method_embedding[name_model + '_' + "_".join(keep_pos_tag) + '_lem'] = (
                                        keep_pos_tag, lemmatize)
                                else:
                                    self.name_models.append(name_model + '_' + "_".join(keep_pos_tag))
                                    self.method_embedding[name_model + '_' + "_".join(keep_pos_tag)] = (
                                        keep_pos_tag, lemmatize)
                            if name_split[0].lower() == 'tf-idf':
                                self.class_embeddings.append(Tfidf)
                            else:
                                self.class_embeddings.append(Tf_embedding)

                # Word2Vec embeddings :
                elif name_split[0].lower() in ['word2vec']:
                    if len(name_split) > 1:
                        self.name_classifiers.append(''.join([i for i in name_split[1] if not i.isdigit()]))  # remove digit
                    if 'word2vec' in self.method_embedding.keys():
                        self.name_models.append(name_model)
                        self.class_embeddings.append(Word2Vec)
                        self.method_embedding[name_model] = self.method_embedding['word2vec']
                    else:
                        logger.warning(
                            "\nInfo : Pre-training weight is not provided in method_embedding, continue without {}".format(
                                name_model))

                # FastText embeddings :
                elif name_split[0].lower() in ['fasttext']:
                    if len(name_split) > 1:
                        self.name_classifiers.append(''.join([i for i in name_split[1] if not i.isdigit()]))  # remove digit
                    if 'fasttext' in self.method_embedding.keys():
                        self.name_models.append(name_model)
                        self.class_embeddings.append(Fasttext)
                        self.method_embedding[name_model] = self.method_embedding['fasttext']
                    else:
                        logger.warning(
                            "\nInfo : Pre-training weight is not provided in method_embedding, continue without {}".format(
                                name_model))

                # Doc2Vec embeddings :
                elif name_split[0].lower() in ['doc2vec']:
                    if len(name_split) > 1:
                        self.name_classifiers.append(''.join([i for i in name_split[1] if not i.isdigit()]))  # remove digit
                    if 'doc2vec' in self.method_embedding.keys():
                        self.name_models.append(name_model)
                        self.class_embeddings.append(Doc2Vec)
                        self.method_embedding[name_model] = self.method_embedding['doc2vec']
                    else:
                        logger.warning(
                            "\nInfo : Pre-training weight is not provided in method_embedding, continue without {}".format(
                                name_model))

                # Transformer embeddings :
                elif name_split[0].lower() in ['transformer']:
                    if len(name_split) > 1:
                        self.name_classifiers.append(''.join([i for i in name_split[1] if not i.isdigit()]))
                    if 'transformer' in self.method_embedding.keys():
                        name_model = self.method_embedding['transformer'] + "+" + name_split[1]
                        self.name_models.append(name_model)
                        self.class_embeddings.append(TransformerNLP)
                        self.method_embedding[name_model] = self.method_embedding['transformer']
                    else:
                        logger.warning(
                            "\nInfo : Name of Transformer model is not provided in method_embedding, continue without {}".format(
                                name_model))

                else:
                    logger.warning("\nInfo : Unknown Name of the embedding method, continue without {}".format(name_model))

        # if not exist column text : need name model in classifier_nlp format
        else:
            for name_model in self.include_model:
                self.name_models.append(name_model)
                self.name_classifiers.append(''.join([i for i in name_model if not i.isdigit()]))  # remove digit

    def train(self, x=None, y=None, x_val=None, y_val=None):
        """ Compute each NLP model :
                    - Optimization hyperparameters
                    - Validation/Cross-validation with best hyperparameters
                    - Save models from cross-validation
        Args:
              x (Dataframe)
              y (Dataframe)
        """

        self.prepare_model(x=x, y=y, x_val=x_val, y_val=y_val)

        logger.info("List of models : {}".format(self.name_models))

        # variable to store best Hyperparameters with specific "seed" and "scoring"
        dict_models_parameters = {"seed": self.seed, "scoring": self.scoring}

        # Replace dict_models_parameters if already exist a json file with all precedent best Hyperparameters
        try:
            with open(self.flags_parameters.path_models_best_parameters) as json_file:
                dict_models_parameters = json.load(json_file)
                # WARNING: it is preferably that "seed" and "scoring" from file are equal to current "seed" and "scoring"
                if dict_models_parameters['seed'] != self.seed:
                    logger.warning("INFO : seed from loaded dict_models_parameters is not equal to seed from flags")
                if dict_models_parameters['scoring'] != self.scoring:
                    logger.warning(
                        "INFO : scoring from loaded dict_models_parameters is not equal to scoring from flags")
        except:
            if self.apply_logs:
                path_models_best_parameters = os.path.join(self.flags_parameters.outdir, "models_best_parameters.json")
            elif self.apply_mlflow:
                path_models_best_parameters = os.path.join(self.path_mlflow, self.experiment_id,
                                                           "models_best_parameters.json")
            try:
                with open(path_models_best_parameters) as json_file:
                    dict_models_parameters = json.load(json_file)
                    # WARNING: it is preferably that "seed" and "scoring" from file are equal to current "seed" and "scoring"
                    if dict_models_parameters['seed'] != self.seed:
                        logger.warning("INFO : seed from loaded dict_models_parameters is not equal to seed from flags")
                    if dict_models_parameters['scoring'] != self.scoring:
                        logger.warning(
                            "INFO : scoring from loaded dict_models_parameters is not equal to scoring from flags")
            except:
                logger.warning(
                    "Unknown path for path_models_best_parameters, a new dictionary has been created.")

        if self.apply_logs:
            # Create a new 'last_logs' directory
            dir = os.path.join(self.outdir, 'last_logs')
            if os.path.exists(dir):
                rmtree(dir)
            os.makedirs(dir)
            # Create a 'best_logs' directory if not exist
            os.makedirs(os.path.join(self.flags_parameters.outdir, 'best_logs'), exist_ok=True)

        if self.apply_autonlp:
            dict_classifiers = {'naive_bayes': Naive_Bayes, 'logistic_regression': Logistic_Regression, 'sgd_classifier': SGD_Classifier,
                                'sgd_regressor': SGD_Regressor, 'xgboost': XGBoost, 'global_average': Global_Average,
                                'attention': Attention, 'birnn': Birnn, 'birnn_attention': Birnn_Attention,
                                'bilstm': Bilstm, 'bilstm_attention': Bilstm_Attention, 'bigru': Bigru,
                                'bigru_attention': Bigru_Attention}
            class_models = [dict_classifiers[name_classifier.lower()] for name_classifier in self.name_classifiers]
        else:
            dict_classifiers = {'logistic_regression': ML_Logistic_Regression, 'randomforest': ML_RandomForest,
                                'lightgbm': ML_LightGBM, 'xgboost': ML_XGBoost, 'catboost': ML_CatBoost,
                                'dense_network': ML_DenseNetwork}
            class_models = [dict_classifiers[name_classifier.lower()] for name_classifier in self.name_classifiers]

        # Compute each NLP model :
        for i, name_model in enumerate(self.name_models):
            logger.info('\n\033[4m{} Model\033[0m:'.format(name_model))

            #####################
            # MODELS AUTONLP :
            #####################
            if self.apply_autonlp:

                self.models[name_model] = class_models[i](self.flags_parameters, self.class_embeddings[i],
                                                          name_model, self.column_text, self.class_weight)

                self.models[name_model].autonlp(self.x_train, self.y_train, self.x_val, self.y_val,
                                                self.apply_optimization, self.apply_validation,
                                                self.method_embedding[name_model],
                                                self.doc_spacy_data_train, self.doc_spacy_data_val)
            #####################
            # MODELS AUTOML :
            #####################
            else:

                self.models[name_model] = class_models[i](self.flags_parameters, name_model, self.class_weight,
                                                          self.len_unique_value)

                self.models[name_model].automl(self.x_train, self.y_train, self.x_val, self.y_val,
                                               self.apply_optimization, self.apply_validation)

            #####################

            # The following code consists to update and save best Hyperparameters / best models /
            # validation leaderboard / history of Hyperparameters Optimization

            # Save best Hyperparameters / best models :
            # if name of the model already in dictionary of best hyperparameters
            #    then apply saving only if current cv_score is better than latest best_cv_score
            # else apply saving
            actual_best_cv_score = self.models[name_model].best_cv_score
            if self.apply_logs:

                if self.apply_autonlp:
                    dir_best_logs_embedding = os.path.join(self.flags_parameters.outdir, 'best_logs',
                                                           self.models[name_model].embedding.name_model)
                    os.makedirs(dir_best_logs_embedding, exist_ok=True)
                    if self.models[name_model].embedding.name_model == "transformer":
                        dir_best_logs_model = os.path.join(dir_best_logs_embedding, name_model)
                    else:
                        dir_best_logs_model = os.path.join(dir_best_logs_embedding, name_model.split('+')[1])
                else:
                    dir_best_logs_classifier = os.path.join(self.flags_parameters.outdir, 'best_logs',
                                                       self.models[name_model].name_classifier)
                    os.makedirs(dir_best_logs_classifier, exist_ok=True)
                    dir_best_logs_model = os.path.join(dir_best_logs_classifier, name_model)

            if name_model in dict_models_parameters.keys():
                latest_best_cv_score = dict_models_parameters[name_model]['best_cv_score']
                if actual_best_cv_score > latest_best_cv_score:
                    logger.info(
                        "The best cv score {} improved from latest best cv score {}, saving parameters.".format(
                            actual_best_cv_score, latest_best_cv_score))
                    # Save parameters :
                    dict_models_parameters[name_model] = self.models[name_model].params_all[name_model].copy()
                    dict_models_parameters[name_model]['best_cv_score'] = float(actual_best_cv_score)

                    # Save model in best_logs :
                    if self.apply_logs:
                        if os.path.exists(dir_best_logs_model):
                            rmtree(dir_best_logs_model)
                        copytree(dir_best_logs_model.replace('best_logs', 'last_logs'), dir_best_logs_model)

                else:
                    if self.apply_logs and os.path.exists(dir_best_logs_model) and len(
                            glob(dir_best_logs_model + '/fold*')) == 0:
                        with open(os.path.join(dir_best_logs_model, "parameters.json")) as json_file:
                            dict_models_best_parameters = json.load(json_file)
                        if self.models[name_model].params_all[name_model] == dict_models_best_parameters[name_model]:
                            logger.info(
                                "In directory '{}', there is no registered model and has same parameters than model "
                                "from last_logs. Copy models from last_logs to best_logs for {}.".format(
                                    dir_best_logs_model, name_model))
                            if os.path.exists(dir_best_logs_model):
                                rmtree(dir_best_logs_model)
                            copytree(dir_best_logs_model.replace('best_logs', 'last_logs'), dir_best_logs_model)
                    else:
                        logger.info(
                            "The best cv score {} is not an improvement from current best cv score {}, parameters are not saved.".format(
                                actual_best_cv_score, latest_best_cv_score))
            else:
                # Save parameters :
                dict_models_parameters[name_model] = self.models[name_model].params_all[name_model].copy()
                dict_models_parameters[name_model]['best_cv_score'] = float(actual_best_cv_score)

                # Save model in best_logs :
                if self.apply_logs:
                    if os.path.exists(dir_best_logs_model):
                        rmtree(dir_best_logs_model)
                    copytree(dir_best_logs_model.replace('best_logs', 'last_logs'), dir_best_logs_model)

            # Best Hyperparameters dictionary will be save in "models_best_parameters.json"
            if self.apply_logs:
                with open(os.path.join(self.flags_parameters.outdir, "models_best_parameters.json"), "w") as outfile:
                    json.dump(dict_models_parameters, outfile)
            if self.apply_mlflow:
                with open(os.path.join(self.path_mlflow, self.experiment_id, "models_best_parameters.json"),
                          "w") as outfile:
                    json.dump(dict_models_parameters, outfile)

            if self.apply_logs and self.apply_validation:
                # Save plot of validation leaderboard score after each model because algo can stop before due to error
                leaderboard_val = self.get_leaderboard(sort_by=self.flags_parameters.sort_leaderboard, dataset='val')
                self.save_scores_plot(leaderboard_val, 'last_logs')

            # Save history results from Optimization after each model because algo can stop before due to error
            if len(self.models[name_model].df_all_results) > 0:
                try:
                    if self.apply_logs:
                        df_all_results = pd.read_csv(os.path.join(self.outdir, "df_all_results.csv"))
                    elif self.apply_mlflow:
                        df_all_results = pd.read_csv(
                            os.path.join(self.path_mlflow, self.experiment_id, "df_all_results.csv"))
                    else:
                        df_all_results = pd.DataFrame()
                except:
                    df_all_results = pd.DataFrame()
                df_all_results = pd.concat([df_all_results, self.models[name_model].df_all_results],
                                           axis=0).reset_index(drop=True)
                if self.apply_logs:
                    df_all_results.to_csv(os.path.join(self.outdir, "df_all_results.csv"), index=False)
                if self.apply_mlflow:
                    df_all_results.to_csv(os.path.join(self.path_mlflow, self.experiment_id, "df_all_results.csv"),
                                          index=False)

            # Save Boxplot of history results from Optimization
            if self.apply_optimization:
                self.save_boxplot_df_all_results()

        if self.apply_blend_model:
            self.ensemble()

    def ensemble(self):
        """ Apply ensemble model :class:BlendModel """

        # blend model average:
        if self.apply_blend_model:
            if len(self.models.keys()) >= 2 and self.apply_validation:
                logger.info('\n\033[4mBlend Model\033[0m:')
                model_blend = BlendModel(self.objective, self.flags_parameters.average_scoring,
                                         self.flags_parameters.map_label)
                model_blend.validation(self.models, self.x_train, self.y_train, self.x_val, self.y_val)
                self.models[model_blend.name_model] = model_blend

                leaderboard_val = self.get_leaderboard(sort_by=self.flags_parameters.sort_leaderboard, dataset='val')
                self.save_scores_plot(leaderboard_val, 'last_logs')

            else:
                self.apply_blend_model = False

    def fit_transform_embedding(self, x=None, y=None, x_val=None, y_val=None):
        """ Compute each NLP model :
                    - Optimization hyperparameters
                    - Validation/Cross-validation with best hyperparameters
                    - Save models from cross-validation
        Args:
              x (Dataframe)
              y (Dataframe)
        """

        logger.info('\nFit and Transform Embeddings :')

        self.prepare_model(x=x, y=y, x_val=x_val, y_val=y_val)

        logger.info("List of embeddings : {}".format(self.name_models))

        dict_embeddings = {}

        # Compute each NLP model :
        for i, name_model in enumerate(self.name_models):
            logger.info('\n\033[4m{} Embedding\033[0m...'.format(name_model))

            #####################
            # EMBEDDING METHOD :
            #####################
            self.embeddings[name_model] = Embedding(self.flags_parameters, self.class_embeddings[i],
                                                    self.flags_parameters.dimension_embedding, self.column_text)

            dict_preprocessed = self.embeddings[name_model].fit_transform(self.x_train, self.x_val,
                                                                          self.method_embedding[name_model],
                                                                          self.doc_spacy_data_train,
                                                                          self.doc_spacy_data_val)
            dict_embeddings[name_model] = dict_preprocessed

        return dict_embeddings

    def transform_embedding(self, name_logs='last_logs', on_test_data=True, x=None, doc_spacy_data_test=None):
        """ Compute each NLP model :
                    - Optimization hyperparameters
                    - Validation/Cross-validation with best hyperparameters
                    - Save models from cross-validation
        Args:
              x (Dataframe)
              y (Dataframe)
        """

        logger.info('\nTransform Embeddings :')

        if on_test_data and x is None:  # predict on self.X_test
            x = self.X_test
            doc_spacy_data_test = self.doc_spacy_data_test

        dict_embeddings = {}

        # use self.models to get models:
        if name_logs == 'last_logs':
            for name_model in self.embeddings.keys():
                logger.info('\n\033[4m{} Embedding\033[0m...'.format(name_model))

                x_test_preprocessed = self.embeddings[name_model].transform(x, doc_spacy_data_test)
                dict_embeddings[name_model] = x_test_preprocessed

        # Load model from outdir_name_logs:
        else:
            self.info_models = {}
            n_models = 0

            if self.apply_mlflow:
                pass

            else:
                outdir_file_embeddings = os.path.join(self.outdir, "embedding")
                # possible_embeddings = ['tf', 'tf-idf', 'fasttext', 'word2vec', 'doc2vec', 'transformer']
                try:
                    name_embeddings = [name_embedding for name_embedding in os.listdir(outdir_file_embeddings) if
                                       os.path.isdir(os.path.join(outdir_file_embeddings, name_embedding))]
                except FileNotFoundError:
                    name_embeddings = []
                    logger.error("path '{}' is not provided".format(outdir_file_embeddings))

                class_embeddings = []
                for name_embedding in name_embeddings:

                    if name_embedding == 'tf':
                        class_embedding = Tf_embedding
                    elif name_embedding == 'tf-idf':
                        class_embedding = Tfidf
                    elif name_embedding == 'word2vec':
                        class_embedding = Word2Vec
                    elif name_embedding == 'fasttext':
                        class_embedding = Fasttext
                    elif name_embedding == 'doc2vec':
                        class_embedding = Doc2Vec
                    elif name_embedding == 'transformer':
                        class_embedding = TransformerNLP
                    else:
                        logger.error("\nInfo : Unknown Name of the embedding method : {}".format(name_embedding))

                    class_embeddings.append(class_embedding)

            self.info_models = {}

            for i, name_model in enumerate(name_embeddings):
                logger.info('\n\033[4m{} Embedding\033[0m...'.format(name_model))

                #####################
                # EMBEDDING METHOD :
                #####################
                self.info_models[name_model] = Embedding(self.flags_parameters, class_embeddings[i],
                                                         self.flags_parameters.dimension_embedding, self.column_text)

                x_test_preprocessed = self.info_models[name_model].transform(x, doc_spacy_data_test)
                dict_embeddings[name_model] = x_test_preprocessed

        return dict_embeddings

    def get_list_name_models(self):
        """ Returns : (List) name of all NLP models """
        if self.apply_blend_model:
            return self.name_models + ['BlendModel']
        else:
            return self.name_models

    def get_leaderboard(self, dataset='val', sort_by=None, ascending=False, info_models=None):
        """ Metric scores for each model of self.models or info_models
            if no optimization and validation : you need to give info_model (dictionary with Model class)
        Args:
            dataset (str) : 'val' or 'test', which prediction to use
            sort_by (str) : metric column name to sort
            ascending (Boolean)
            info_models (dict) : dictionary with Model class
        Return:
             self.leaderboard (Dataframe)
        """
        if 'binary' in self.objective:
            metrics = ['accuracy', 'recall', 'precision', 'f1', 'roc_auc']
        elif 'multi-class' in self.objective:
            metrics = ['accuracy', 'recall_' + self.flags_parameters.average_scoring,
                       'precision_' + self.flags_parameters.average_scoring,
                       'f1_' + self.flags_parameters.average_scoring]
        elif 'regression' in self.objective:
            metrics = ['mse', 'rmse', 'explained_variance', 'r2']

        try:
            if info_models is None:
                info_models = self.models
            leaderboard = {"name": list(info_models.keys())}
            for metric in metrics:
                self.info_scores[metric + '_' + dataset] = [info_models[name_model].info_scores[metric + '_' + dataset]
                                                            for
                                                            name_model in info_models.keys()]
                leaderboard[metric + '_' + dataset] = np.round(self.info_scores[metric + '_' + dataset], 4)
            if sort_by == 'mse' or sort_by == 'rmse':
                ascending = True
            # for older version:
            if sort_by[-2:] == "_w":
                sort_by = sort_by[:-1] + "weighted"
            if sort_by in ['recall', 'precision', 'f1'] and 'multi-class' in self.objective:
                sort_by = sort_by + '_' + self.flags_parameters.average_scoring

            leaderboard = pd.DataFrame(leaderboard)
            if sort_by:
                leaderboard = leaderboard.sort_values(by=sort_by + '_' + dataset, ascending=ascending)
            return leaderboard
        except:
            return pd.DataFrame()

    def save_scores_plot(self, leaderboard, name_logs):
        """ Save line plot of metric leaderboard in name_logs directory"""
        if len(leaderboard) > 0:
            dataset = 'val' if 'val' in leaderboard.columns[1] else 'test'
            fig = px.line(leaderboard, x="name", y=leaderboard.columns,
                          title='Metric scores with {} set'.format(dataset), labels={'name': 'Models'},
                          template='plotly_dark')
            fig.update_traces(mode='lines+markers')
            # fig.show()
            try:
                if self.apply_logs:
                    fig.write_image(os.path.join(self.outdir, name_logs, "metric_scores_" + dataset + ".png"))
            except:
                pass

    def get_cross_validation_prediction(self, name_model):
        """ Get oof_val prediction for name_model
        Args:
            name_model (str)
        Return:
            oof_val (array)
        """
        return self.models[name_model].info_scores['oof_val']

    def get_df_all_results(self):
        """ Information of Hyperparameters Optimization history for each model
        Return:
            df_all_results (Dataframe)
        """
        df_all_results = pd.DataFrame()
        try:
            for name_model in self.models.keys():
                if name_model not in ['BlendModel']:
                    df_all_results_model = self.models[name_model].df_all_results
                    df_all_results_model['model'] = name_model
                    df_all_results = pd.concat([df_all_results, df_all_results_model], axis=0).reset_index(drop=True)
        except AttributeError:
            pass
        return df_all_results

    def save_boxplot_df_all_results(self):
        """ Save Boxplot of history results from Optimization """
        try:
            if self.apply_logs:
                df_all_results = pd.read_csv(os.path.join(self.outdir, "df_all_results.csv"))
            elif self.apply_mlflow:
                df_all_results = pd.read_csv(os.path.join(self.path_mlflow, self.experiment_id, "df_all_results.csv"))
        except:
            if self.apply_logs:
                logger.error("'df_all_results.csv' is not provided in {}".format(self.outdir))
            elif self.apply_mlflow:
                logger.error("'df_all_results.csv' is not provided in {}".format(
                    os.path.join(self.path_mlflow, self.experiment_id)))
        fig = px.box(df_all_results, x="model", y="mean_test_score", points="all",
                     title="Boxplot of history scoring results from Optimization")
        try:
            if self.apply_logs:
                fig.write_image(os.path.join(self.outdir, "boxplot_df_all_results.png"))
            if self.apply_mlflow:
                fig.write_image(os.path.join(self.path_mlflow, self.experiment_id, "boxplot_df_all_results.png"))
        except:
            pass

    def show_distribution_scores(self):
        # not used anymore
        df_all_results = self.get_df_all_results()
        list_name_models = list(df_all_results.model.unique())
        try:
            rows, cols = 2, 3
            fig, ax = plt.subplots(rows, cols, figsize=(50, 20))

            for row in range(rows):
                for col in range(cols):
                    if row * cols + col + 1 <= len(list_name_models):
                        name_model = list_name_models[row * cols + col]
                        values = df_all_results[df_all_results.model.isin([name_model])].mean_test_score
                        if np.std(values) < 1e-4:
                            ax[row, col].hist(values, range=(values.min() - 1e-3, values.max() + 1e-3))
                        else:
                            ax[row, col].hist(values)
                        ax[row, col].set_xlabel(name_model + ' (' + str(len(values)) + ' models)', size=30)
                        ax[row, col].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                        for tick in ax[row, col].xaxis.get_major_ticks():
                            tick.label.set_fontsize(30)
            plt.show()
        except AttributeError:
            pass

    def get_roc_curves(self):
        """ Build roc curve for all models if target objective is 'binary' """
        plt.figure(figsize=(15, 15), linewidth=1)
        for name_model, model in self.models.items():
            plt.plot(model.info_scores['fpr'], model.info_scores['tpr'], label=name_model)
        plt.plot([0, 1], [0, 1], 'k--', label='Random: 0.5')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC Curve')
        plt.legend(loc='best')
        plt.show()

    def correlation_models(self, apply_argmax=False):
        """ Correlation between cross-validation prediction of best models
        Args:
            apply_argmax (Boolean) : if we need to apply argmax on prediction
        """
        example_of_oof_val = self.models[list(self.models.keys())[0]].info_scores['oof_val']
        if len(example_of_oof_val.shape) == 1:
            dict_result = {name_model: self.models[name_model].info_scores['oof_val'] for name_model in
                           self.models.keys()}
            result_val = pd.DataFrame(dict_result)
            sns.set(rc={'figure.figsize': (10, 10)})
            sns.heatmap(result_val.corr(), annot=True, cmap=sns.cm.rocket_r)
            plt.show()
        else:
            y_shape1 = example_of_oof_val.shape[1]
            if apply_argmax and y_shape1 > 1:
                dict_result = {name_model: np.argmax(self.models[name_model].info_scores['oof_val'], axis=1).reshape(-1)
                               for name_model in self.models.keys()}

                result_val = pd.DataFrame(dict_result)
                sns.set(rc={'figure.figsize': (10, 10)})
                sns.heatmap(result_val.corr(), annot=True, cmap=sns.cm.rocket_r)
                plt.show()
            else:
                # column 0 correlation :
                data_corr = pd.DataFrame(
                    {name_model: self.models[name_model].info_scores['oof_val'][:, 0].reshape(-1) for name_model in
                     self.models.keys()}).corr()
                for i in range(1, y_shape1):
                    result_val = pd.DataFrame(
                        {name_model: self.models[name_model].info_scores['oof_val'][:, i].reshape(-1) for name_model in
                         self.models.keys()})
                    data_corr = data_corr + result_val.corr()
                data_corr = data_corr / y_shape1
                sns.set(rc={'figure.figsize': (10, 10)})
                sns.heatmap(data_corr, annot=True, cmap=sns.cm.rocket_r)
                plt.show()

    def show_confusion_matrix(self, name_model, type_data='val', on_test_data=True, y=None):
        """ Show confusion matrix if target objective is 'binary' or 'multi-class'
        Args:
            name_model (str) : full name of the model
            type_data (str) : 'val' or 'test', if 'val' use cross-validation prediction
            on_test_data (Boolean) : if True use prediction self.Y_pred as y_true else use y as y_true
            y (dataframe)
        e.g : self.show_confusion_matrix('tf+Naive_Bayes', 'test')
        """
        if self.Y_train.shape[1] != 1 or 'regression' in self.objective:
            return None

        if type_data == 'val':
            fold_id = self.models[name_model].info_scores['fold_id']
            if 'binary' in self.objective:
                y_pred = np.where(self.models[name_model].info_scores['oof_val'] > 0.5, 1, 0)
            else:
                y_pred = np.argmax(self.models[name_model].info_scores['oof_val'], axis=1).reshape(-1)
            if self.x_val is None:
                y_true = np.array(self.y_train).reshape(-1)[np.where(fold_id >= 0)[0]]
            else:
                y_true = np.array(self.y_val).reshape(-1)[np.where(fold_id >= 0)[0]]
        else:
            # use last prediction on test set
            if 'binary' in self.objective:
                y_pred = np.where(self.models[name_model].info_scores['prediction'] > 0.5, 1, 0)
            else:
                y_pred = np.argmax(self.models[name_model].info_scores['prediction'], axis=1).reshape(-1)
            if on_test_data:
                y_true = np.array(self.Y_test).reshape(-1)
            else:
                y_true = np.array(y).reshape(-1)

        reverse_map_label = None
        if self.flags_parameters.map_label is not None:
            reverse_map_label = {v: k for k, v in self.flags_parameters.map_label.items()}
        df_cm = build_df_confusion_matrix(y_true, y_pred, reverse_map_label)

        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(df_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="BuPu")
        plt.title('Confusion Matrix')
        plt.xlabel('Y pred')
        plt.ylabel('Y true')
        plt.show()

    def create_model_class(self, name_embedding=None, name_model=None, name_logs='last_logs'):
        """ Prediction on x or X_test (if on_test_data=True or x == None) for best name_model model
            Do not work for 'BlendModel'
        Args:
            name_embedding (str) name of the embedding method
            name_model (str) full name of the model
            name_logs (str) 'last_logs' or 'best_logs'
        Returns:
            prediction (array)
            self.models[name_model] (Model) optional
        """

        if self.apply_app:
            try:
                with open(os.path.join(self.outdir, "parameters.json")) as json_file:
                    params_all = json.load(json_file)
                name_model = list(params_all.keys())[0]
                logger.info('\n\033[4m{} Model\033[0m:'.format(name_model))
                params_all = params_all[name_model]
                if self.apply_autonlp:
                    name_embedding = params_all["name_embedding"]
            except:
                logger.error("File 'parameters.json' in {} is not provided".format(self.outdir))

        if self.apply_autonlp:
            if name_embedding == 'tf':
                class_embedding = Tf_embedding
            elif name_embedding == 'tf-idf':
                class_embedding = Tfidf
            elif name_embedding == 'word2vec':
                class_embedding = Word2Vec
            elif name_embedding == 'fasttext':
                class_embedding = Fasttext
            elif name_embedding == 'doc2vec':
                class_embedding = Doc2Vec
            elif name_embedding == 'transformer':
                class_embedding = TransformerNLP
            else:
                logger.error("\nInfo : Unknown Name of the embedding method : {}".format(name_model))

        try:
            if self.apply_logs:
                outdir_name_logs = os.path.join(self.outdir, name_logs)
                outdir_embedding = os.path.join(outdir_name_logs, name_embedding)
                if self.apply_autonlp:
                    if name_embedding == "transformer":
                        outdir_model = os.path.join(outdir_embedding, name_model)
                    else:
                        outdir_model = os.path.join(outdir_embedding, name_model.split('+')[1])
                else:
                    outdir_model = os.path.join(outdir_embedding, name_model)

                with open(os.path.join(outdir_model, "parameters.json")) as json_file:
                    params_all = json.load(json_file)
                params_all = params_all[name_model]

            elif self.apply_mlflow:
                params_all = dict()
                path_mlflow_experiment_id = os.path.join(self.path_mlflow, self.experiment_id)
                for i, dir_run in enumerate(os.listdir(path_mlflow_experiment_id)):
                    if os.path.exists(os.path.join(path_mlflow_experiment_id, dir_run, "tags")):
                        file1 = open(os.path.join(path_mlflow_experiment_id, dir_run, "tags", "mlflow.runName"),
                                     'r')
                        Lines = file1.readlines()
                        if Lines[0] == name_model:
                            run_id_name_model = dir_run
                            break

                with open(os.path.join(path_mlflow_experiment_id, run_id_name_model, "artifacts",
                                       "parameters.json")) as json_file:
                    params_all = json.load(json_file)
                params_all = params_all[name_model]

        except:
            if self.apply_logs:
                logger.error("File 'parameters.json' in {} is not provided".format(outdir_model))
            if self.apply_mlflow:
                logger.error("MLflow Run experiment with name {} is not provided".format(name_model))

        if self.apply_autonlp:
            dict_classifiers = {'naive_bayes': Naive_Bayes, 'logistic_regression': Logistic_Regression,
                                'sgd_classifier': SGD_Classifier,
                                'sgd_regressor': SGD_Regressor, 'xgboost': XGBoost, 'global_average': Global_Average,
                                'attention': Attention, 'birnn': Birnn, 'birnn_attention': Birnn_Attention,
                                'bilstm': Bilstm, 'bilstm_attention': Bilstm_Attention, 'bigru': Bigru,
                                'bigru_attention': Bigru_Attention}
            name_classifier = params_all["name_classifier"]
            class_model = dict_classifiers[name_classifier.lower()]
        else:
            dict_classifiers = {'logistic_regression': ML_Logistic_Regression, 'randomforest': ML_RandomForest,
                                'lightgbm': ML_LightGBM, 'xgboost': ML_XGBoost, 'catboost': ML_CatBoost,
                                'dense_network': ML_DenseNetwork}
            name_classifier = params_all["name_classifier"]
            class_model = dict_classifiers[name_classifier.lower()]

        # load flags ?
        if self.apply_autonlp:
            model_ml = class_model(self.flags_parameters, class_embedding, name_model, self.column_text)
        else:
            model_ml = class_model(self.flags_parameters, name_model)

        if self.apply_logs:
            outdir = self.outdir
        elif self.apply_mlflow:
            outdir = os.path.join(self.path_mlflow, self.experiment_id)
        elif self.apply_app:
            outdir = self.outdir
        model_ml.load_params(params_all, outdir)

        if self.apply_app:
            outdir_model = self.outdir
            # get path of model folds :
            try:
                model_paths = glob(outdir_model + '/' + '*.joblib') + glob(outdir_model + '/' + '*.hdf5')
            except FileNotFoundError:
                logger.critical(
                    "Didn't find checkpoint model for {} in '{}'".format(name_model, outdir_model))
            loaded_models = []
            for path in model_paths:
                model = model_ml.model()
                if model_ml.is_NN:
                    model.load_weights(path)
                else:
                    model = load(path)
                loaded_models.append(model)

            return model_ml, loaded_models

        return model_ml

    def single_prediction(self, name_embedding=None, name_model=None, model_ml=None, loaded_models=None,
                          name_logs='last_logs', on_test_data=True, x=None, y=None, doc_spacy_data_test=None,
                          return_model=False, return_scores=False, proba=True):
        """ Prediction on x or X_test (if on_test_data=True or x == None) for best name_model model
                    Do not work for 'BlendModel'
            Args:
                name_embedding (str) name of the embedding method
                name_model (str) full name of the model
                name_logs (str) 'last_logs' or 'best_logs'
                on_test_data (Boolean) : if True use prediction self.Y_pred as y_true else use y as y_true
                x (DataFrame)
                y (DataFrame)
                doc_spacy_data_test (List) : documents from data_test preprocessed by spacy nlp
                return_model (Boolean) : return :class:Model
                return_scores (Boolean) : return dictionary of metric results
            Returns:
                prediction (array)
                self.models[name_model] (Model) optional
        """

        assert name_model != 'BlendModel', "single_prediction do not support 'BlendModel', use leader_predict instead"
        if name_model is not None:
            logger.info('\n\033[4m{} Model\033[0m:'.format(name_model))
        if on_test_data and x is None:  # predict on self.X_test
            x = self.X_test
            y = self.Y_test
            doc_spacy_data_test = self.doc_spacy_data_test

        if name_logs == 'last_logs' and self.apply_app is False:
            self.models[name_model].prediction(x, y, doc_spacy_data_test, name_logs)

            prediction = self.models[name_model].info_scores['prediction']
            if not proba and 'regression' not in self.objective:
                if 'binary' not in self.objective:   # y.shape[1] == 1
                    confidence = np.max(prediction, axis=1).reshape(-1)
                    prediction = np.argmax(prediction, axis=1).reshape(-1)
                else:
                    confidence = prediction.copy()
                    prediction = np.where(prediction > 0.5, 1, 0)
                prediction = {"prediction": prediction, "confidence": confidence}

            if return_model:
                return prediction, self.models[name_model]
            else:
                return prediction

        else:

            if model_ml is None:
                if self.apply_app:
                    model_ml, loaded_models = self.create_model_class(name_embedding, name_model, name_logs)
                else:
                    model_ml = self.create_model_class(name_embedding, name_model, name_logs)

            model_ml.prediction(x, y, doc_spacy_data_test, name_logs, loaded_models)

            prediction = model_ml.info_scores['prediction']
            if not proba and 'regression' not in self.objective:
                if 'binary' not in self.objective:   #y.shape[1] == 1
                    confidence = np.max(prediction, axis=1).reshape(-1)
                    prediction = np.argmax(prediction, axis=1).reshape(-1)
                else:
                    confidence = prediction.copy()
                    prediction = np.where(prediction > 0.5, 1, 0)
                prediction = {"prediction": prediction, "confidence": confidence}

            if not return_scores:
                if return_model:
                    return prediction, model_ml
                else:
                    return prediction
            else:
                if y is None:
                    scores = None
                else:
                    if 'binary' in self.objective:
                        metrics = ['accuracy', 'recall', 'precision', 'f1', 'roc_auc']
                    elif 'multi-class' in self.objective:
                        metrics = ['accuracy', 'recall_' + self.flags_parameters.average_scoring,
                                   'precision_' + self.flags_parameters.average_scoring,
                                   'f1_' + self.flags_parameters.average_scoring]
                    elif 'regression' in self.objective:
                        metrics = ['mse', 'rmse', 'explained_variance', 'r2']

                    scores = {metric: model_ml.info_scores[metric + '_' + 'test'] for metric in metrics}

                if return_model:
                    return prediction, model_ml, scores
                else:
                    return prediction, scores

    def leader_predict(self, name_logs='last_logs', on_test_data=True, x=None, y=None, doc_spacy_data_test=None):
        """ Prediction on x or X_test (if on_test_data=True or x == None) for each best models
        Args:
            name_logs (str) 'last_logs' or 'best_logs'
            on_test_data (Boolean) : if True use prediction self.Y_pred as y_true else use y as y_true
            x (DataFrame)
            y (DataFrame)
            doc_spacy_data_test (List) : documents from data_test preprocessed by spacy nlp
        """
        if on_test_data or x is None:  # predict on self.X_test
            x = self.X_test
            y = self.Y_test
            doc_spacy_data_test = self.doc_spacy_data_test

        # use self.models to get models:
        if name_logs == 'last_logs':
            for name_model in self.models.keys():
                if name_model == 'BlendModel':
                    logger.info('\n\033[4m{} Model\033[0m:'.format(name_model))
                    self.models[name_model].prediction(self.models, x, y)
                else:
                    name_embedding = self.models[name_model].name_classifier
                    _ = self.single_prediction(name_embedding, name_model, name_logs=name_logs,
                                               on_test_data=on_test_data, x=x,
                                               y=y, doc_spacy_data_test=doc_spacy_data_test)
            if self.apply_logs:
                leaderboard_test = self.get_leaderboard(sort_by=self.flags_parameters.sort_leaderboard, dataset='test')
                self.save_scores_plot(leaderboard_test, name_logs)

        # Load model from outdir_name_logs:
        else:
            self.info_models = {}
            n_models = 0

            if self.apply_mlflow:
                import mlflow
                experiment_id = None
                for dir in os.listdir(self.path_mlflow):
                    if os.path.exists(os.path.join(self.path_mlflow, dir, "meta.yaml")):
                        meta_flags = load_yaml(os.path.join(self.path_mlflow, dir, "meta.yaml"))
                        if meta_flags['name'] == self.experiment_name:
                            experiment_id = meta_flags['experiment_id']
                if experiment_id is None:
                    logger.warning("The MLflow Tracking with experiment name '{}' is not provided in {}.".format(
                        self.experiment_name, self.path_mlflow))

                list_name_models = []
                list_name_embeddings = []
                path_mlflow_experiment_id = os.path.join(self.path_mlflow, experiment_id)
                for i, dir_run in enumerate(os.listdir(path_mlflow_experiment_id)):
                    if os.path.exists(os.path.join(path_mlflow_experiment_id, dir_run, "tags")):
                        file = open(os.path.join(path_mlflow_experiment_id, dir_run, "tags", "mlflow.runName"), 'r')
                        name_model = file.readlines()
                        list_name_models.append(name_model)
                        if self.apply_autonlp:
                            file = open(os.path.join(path_mlflow_experiment_id, dir_run, "tags", "name_embedding"), 'r')
                            name_embedding = file.readlines()
                            list_name_embeddings.append(name_embedding)

            else:
                outdir_name_logs = os.path.join(self.outdir, name_logs)
                # possible_embeddings = ['tf', 'tf-idf', 'fasttext', 'word2vec', 'doc2vec', 'transformer']

                try:
                    name_embeddings = [name_embedding for name_embedding in os.listdir(outdir_name_logs) if
                                       os.path.isdir(os.path.join(outdir_name_logs, name_embedding))]
                except FileNotFoundError:
                    name_embeddings = []
                    logger.error("path '{}' is not provided".format(outdir_name_logs))

                list_name_models = []
                list_name_embeddings = []
                for name_embedding in name_embeddings:
                    outdir_embedding = os.path.join(outdir_name_logs, name_embedding)
                    name_models_p = [name_model for name_model in os.listdir(outdir_embedding) if
                                     os.path.isdir(os.path.join(outdir_embedding, name_model))]
                    name_models = []
                    if self.apply_autonlp:
                        for name_model in name_models_p:
                            if "+" not in name_model:
                                name_models.append(name_embedding + "+" + name_model)
                            else:
                                name_models.append(name_model)
                    else:
                        name_models = name_models_p
                    list_name_models.extend(name_models)
                    list_name_embeddings.extend([name_embedding for i in range(len(name_models))])

            for name_embedding, name_model in zip(list_name_embeddings, list_name_models):
                _, model_nlp = self.single_prediction(name_embedding, name_model, name_logs=name_logs,
                                                      on_test_data=on_test_data, x=x,
                                                      y=y, doc_spacy_data_test=doc_spacy_data_test,
                                                      return_model=True)
                self.info_models[name_model] = model_nlp
                n_models += 1

            # BlendModel :
            if n_models >= 2:
                logger.info('\n\033[4m{} Model\033[0m:'.format('BlendModel'))
                model_blend = BlendModel(self.objective, self.flags_parameters.average_scoring,
                                         self.flags_parameters.map_label)
                model_blend.prediction(self.info_models, x, y)
                self.info_models[model_blend.name_model] = model_blend

            if self.apply_logs:
                leaderboard_test = self.get_leaderboard(sort_by=self.flags_parameters.sort_leaderboard, dataset='test',
                                                        info_models=self.info_models)
                self.save_scores_plot(leaderboard_test, name_logs)

        # Create a dataframe with predictions of each model + y_true
        if self.Y_train.shape[1] == 1:
            dict_prediction = {}
            if on_test_data:
                if self.Y_test is not None:
                    dict_prediction['y_true'] = np.array(self.Y_test).reshape(-1)
            else:
                if y is not None:
                    dict_prediction['y_true'] = np.array(y).reshape(-1)

            if self.info_models == {}:
                for name_model in self.models.keys():
                    if 'regression' not in self.objective:
                        if 'binary' in self.objective:
                            dict_prediction[name_model] = np.where(
                                self.models[name_model].info_scores['prediction'] > 0.5, 1, 0).reshape(-1)
                        else:
                            dict_prediction[name_model] = np.argmax(self.models[name_model].info_scores['prediction'],
                                                                    axis=1).reshape(-1)
                    else:
                        dict_prediction[name_model] = self.models[name_model].info_scores['prediction']
            else:
                for name_model in self.info_models.keys():
                    if 'regression' not in self.objective:
                        if 'binary' in self.objective:
                            dict_prediction[name_model] = np.where(
                                self.info_models[name_model].info_scores['prediction'] > 0.5, 1, 0).reshape(-1)
                        else:
                            dict_prediction[name_model] = np.argmax(
                                self.info_models[name_model].info_scores['prediction'], axis=1).reshape(-1)
                    else:
                        dict_prediction[name_model] = self.info_models[name_model].info_scores['prediction']

            self.dataframe_predictions = pd.DataFrame(dict_prediction)

    def get_test_prediction(self, name_model):
        """ Get test prediction for name_model
        Args:
            name_model (str)
        Return:
            prediction (array)
        """
        return self.models[name_model].info_scores['prediction']

    def launch_to_model_deployment(self, name_model):

        if self.apply_logs:
            path_file_model = None
            params_all_model = None
            path_best_logs = os.path.join(self.outdir, "best_logs")
            directories = [x[0] for x in os.walk(path_best_logs)]
            for dir in directories:
                if os.path.exists(os.path.join(dir, "parameters.json")):
                    with open(os.path.join(dir, "parameters.json")) as json_file:
                        params_all = json.load(json_file)
                    name_model_dir = list(params_all.keys())[0]
                    if name_model.lower() == name_model_dir.lower():
                        path_file_model = dir
                        params_all_model = params_all[name_model_dir]
                        break
            if path_file_model is None:
                logger.error(
                    "File 'parameters.json' for model '{}' in {} is not provided".format(name_model, path_best_logs))
                return

            if os.path.exists("./model_deployment"):
                rmtree("./model_deployment")
            os.makedirs("./model_deployment", exist_ok=True)

            # copy/deploy parameters.json + saved models
            for file in os.listdir(path_file_model):
                copyfile(os.path.join(path_file_model, file), os.path.join("./model_deployment", file))

            # copy/deploy flags.yaml
            copyfile(os.path.join(self.outdir, "flags.yaml"), os.path.join("./model_deployment", "flags.yaml"))

            # copy/deploy tokenizer.pickle
            if self.apply_autonlp:
                name_classifier = params_all_model["name_classifier"]
                name_embedding = params_all_model["name_embedding"]
                method_embedding = params_all_model["method_embedding"]
                if name_classifier.lower() in ['global_average', 'attention', 'birnn', 'birnn_attention', 'bilstm',
                                               'bilstm_attention', 'bigru', 'bigru_attention']:
                    # For these classifier_nlp models, it need an embedding of words :
                    dimension_embedding = 'word_embedding'
                    if name_embedding in ['tf', 'tf-idf']:
                        keep_pos_tag, lemmatize = method_embedding[0], method_embedding[1]
                        if keep_pos_tag == 'all':
                            if lemmatize == True:
                                tokenizer_name = 'tokenizer_lem'
                            else:
                                tokenizer_name = 'tokenizer_ALL'
                        else:
                            if lemmatize == True:
                                tokenizer_name = 'tokenizer' + '_' + "_".join(keep_pos_tag) + '_lem'
                            else:
                                tokenizer_name = 'tokenizer' + '_' + "_".join(keep_pos_tag)
                        copyfile(os.path.join(self.outdir, tokenizer_name + '.pickle'),
                                 os.path.join("./model_deployment", tokenizer_name + '.pickle'))
                    elif name_embedding == 'transformer':
                        pass
                    elif name_embedding in ['word2vec', 'fasttext', 'doc2vec']:
                        copyfile(os.path.join(self.outdir, 'tokenizer.pickle'),
                                 os.path.join("./model_deployment", 'tokenizer.pickle'))
                else:
                    dimension_embedding = 'doc_embedding'

                if name_embedding in ['word2vec', 'fasttext', 'doc2vec']:
                    found_dir = False
                    for name_directory in ['Word2Vec', 'FastText', 'Doc2Vec']:
                        if os.path.samefile(os.path.dirname(method_embedding), os.path.join(self.outdir, name_directory)):
                            copytree(os.path.join(self.outdir, name_directory),
                                     os.path.join("./model_deployment", name_directory))
                            found_dir = True
                            break
                    if not found_dir:
                        copyfile(method_embedding)


    def extraction(self, name_embedding=None, name_model=None, model_nlp=None, loaded_models=None,
                          name_logs='last_logs', on_test_data=True, x=None, y=None, X_is_train_data=False,
                          n_influent_word = 10, nb_example="max"):
        """ Prediction on x or X_test (if on_test_data=True or x == None) for best name_model model
                    Do not work for 'BlendModel'
            Args:
                name_embedding (str) name of the embedding method
                name_model (str) full name of the model
                name_logs (str) 'last_logs' or 'best_logs'
                on_test_data (Boolean) : if True use prediction self.Y_pred as y_true else use y as y_true
                x (DataFrame)
                y (DataFrame)
                doc_spacy_data_test (List) : documents from data_test preprocessed by spacy nlp
                return_model (Boolean) : return :class:Model
                return_scores (Boolean) : return dictionary of metric results
            Returns:
                prediction (array)
                self.models[name_model] (Model) optional
        """

        assert name_model != 'BlendModel', "single_prediction do not support 'BlendModel', use leader_predict instead"
        if name_model is not None:
            logger.info('\n\033[4m{} Model\033[0m:'.format(name_model))
        if on_test_data and x is None:  # predict on self.X_test
            x = self.X_test
            y = self.Y_test
            doc_spacy_data_test = self.doc_spacy_data_test

        if name_logs == 'last_logs' and self.apply_app is False:
            #self.models[name_model].prediction(x, y, doc_spacy_data_test, name_logs)
            pass

        else:

            if model_nlp is None:
                if self.apply_app:
                    model_nlp, loaded_models = self.create_model_class(name_embedding, name_model, name_logs)
                else:
                    model_nlp = self.create_model_class(name_embedding, name_model, name_logs)

            outdir_model = self.outdir
            # get path of model folds :
            try:
                model_paths = glob(outdir_model + '/' + '*.joblib') + glob(outdir_model + '/' + '*.hdf5')
            except FileNotFoundError:
                logger.critical(
                    "Didn't find checkpoint model for {} in '{}'".format(name_model, outdir_model))

            html, dict_result = extract_influent_word(x, y, model_nlp, loaded_models, model_paths, X_is_train_data,
                                                      n_influent_word, nb_example)

            Html_file = open(os.path.join(self.outdir, "extraction_word.html"), "w", encoding='utf8')
            Html_file.write(html)
            Html_file.close()

            return html, dict_result

    def show_influent_terms(self, dict_result, model_nlp, n_gram=2, top_k=20, min_threshold=10):
        html = get_top_influent_word(dict_result, model_nlp, n_gram, top_k, min_threshold)

        Html_file = open(os.path.join(self.outdir, "influent_terms.html"), "w", encoding='utf8')
        Html_file.write(html)
        Html_file.close()

        return html

