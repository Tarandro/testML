import spacy
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import random as rd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from joblib import load, dump
import pickle

from ..utils.logging import get_logger
from ..utils.nlp_preprocessing import small_clean_text, nlp_preprocessing_spacy, transform_entities
from ..utils.ml_preprocessing import *
from ..utils.feature_importances import get_features_importance

logger = get_logger(__name__)


class Preprocessing:
    """Class for compile full pipeline of NLP preprocessing task.
            Preprocessing_NLP steps:
                - create a Dataframe for targets (self.Y in __init__)
                - create a Dataframe with only the column with texts (data in __init__)
                - load spacy model according to name_spacy_model (self.nlp)
                - (Optional) can apply a small cleaning on text column
                - (Optional) preprocess text column with nlp.pipe spacy
                - (Optional) replace Named entities by tags
    """

    def __init__(self, data, flags_parameters):
        """
        Args:
            data (Dataframe)
            flags_parameters : Instance of Flags class object
        From flags_parameters:
            column_text (str) : name of the column with texts (only one column)
            target (str or list) : names of target columns
            apply_small_clean (Boolean) step 1 of transform
            apply_spacy_preprocessing (Boolean) step 2 of transform
            apply_entity_preprocessing (Boolean) step 3 of transform
        """

        self.flags_parameters = flags_parameters

        self.type_columns = flags_parameters.type_columns
        self.ordinal_features = flags_parameters.ordinal_features

        self.apply_preprocessing_mandatory = flags_parameters.apply_preprocessing_mandatory
        self.remove_categorical = flags_parameters.remove_categorical

        self.method_nan_categorical = flags_parameters.method_nan_categorical
        self.method_nan_numeric = flags_parameters.method_nan_numeric
        self.class_weight = flags_parameters.class_weight
        self.subsample = flags_parameters.subsample
        self.feature_interaction = flags_parameters.feature_interaction
        self.feature_ratio = flags_parameters.feature_ratio
        self.polynomial_features = flags_parameters.polynomial_features
        self.remove_multicollinearity = flags_parameters.remove_multicollinearity
        self.feature_selection = flags_parameters.feature_selection
        self.bin_numeric_features = flags_parameters.bin_numeric_features
        self.remove_low_variance = flags_parameters.remove_low_variance
        self.columns_to_remove = flags_parameters.columns_to_remove
        self.info_pca = flags_parameters.info_pca
        self.info_tsne = flags_parameters.info_tsne
        self.info_stats = flags_parameters.info_stats

        self.multicollinearity_threshold = flags_parameters.multicollinearity_threshold
        self.feature_selection_threshold = flags_parameters.feature_selection_threshold
        self.remove_percentage = flags_parameters.remove_percentage

        self.step_lags = flags_parameters.step_lags
        self.step_rolling = flags_parameters.step_rolling
        self.win_type = flags_parameters.win_type

        self.column_text = flags_parameters.column_text
        self.apply_small_clean = flags_parameters.apply_small_clean
        self.name_spacy_model = flags_parameters.name_spacy_model
        self.apply_spacy_preprocessing = flags_parameters.apply_spacy_preprocessing
        self.apply_entity_preprocessing = flags_parameters.apply_entity_preprocessing
        # you can't apply entity preprocessing if apply_spacy_preprocessing is False :
        if not self.apply_spacy_preprocessing:
            self.apply_entity_preprocessing = False
        self.last_spacy_model_download = None
        self.nlp = None
        self.map_label = dict()

        assert isinstance(data, pd.DataFrame), "data must be a DataFrame type"

        # self.target need to be a List
        self.target = flags_parameters.target
        if self.target is not None:
            if isinstance(self.target, list):
                self.target = self.target
            else:
                self.target = [self.target]

            if len([col for col in self.target if col in data.columns])> 0:
                self.Y = data[[col for col in self.target if col in data.columns]]
                assert self.Y.shape[1] > 0, 'target specifying the column with labels to predict is not in data'

                # create a map label if labels are not numerics
                self.map_label = flags_parameters.map_label
                if self.Y.shape[1] == 1 and self.map_label == {}:
                    if not pd.api.types.is_numeric_dtype(self.Y.iloc[:, 0]):
                        self.map_label = {label: i for i, label in enumerate(self.Y.iloc[:, 0].unique())}
                elif self.Y.shape[1] == 1 and self.map_label != {}:
                    if self.Y[self.Y.columns[0]].iloc[0] in self.map_label.keys():
                        self.Y[self.Y.columns[0]] = self.Y[self.Y.columns[0]].map(self.map_label)
                # more than one column target
                elif self.map_label == {}:
                    for i in range(self.Y.shape[1]):
                        if not pd.api.types.is_numeric_dtype(self.Y.iloc[:, i]):
                            self.map_label[i] = {label: j for j, label in enumerate(self.Y.iloc[:, i].unique())}
                elif self.map_label != {}:
                    for i in range(self.Y.shape[1]):
                        if i in self.map_label.keys() and self.Y[self.Y.columns[i]].iloc[0] in self.map_label.keys():
                            self.Y[self.Y.columns[i]] = self.Y[self.Y.columns[i]].map(self.map_label)
            else:
                self.Y = None
        else:
            self.Y = None

        if self.type_columns is None:
            self.type_columns = {
                'numeric': list(data.loc[:, data.dtypes.astype(str).isin(
                    ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64', 'float32',
                     'float64'])].columns),
                'categorical': list(
                    data.loc[:, data.dtypes.astype(str).isin(['O', 'object', 'category', 'bool'])].columns),
                'date': list(data.loc[:,
                             data.dtypes.astype(str).isin(['datetime64', 'datetime64[ns]', 'datetime'])].columns)
            }
            self.type_columns = {k: [col for col in v if col not in self.target+self.columns_to_remove] for k, v in self.type_columns.items()}

        self.base_features = [col for col in list(data.columns) if col not in self.target+self.columns_to_remove]  # useful for pca / tsne in case of categorical features

        self.outdir_pre = os.path.join(flags_parameters.outdir, "preprocessing")
        os.makedirs(self.outdir_pre, exist_ok=True)

    def load_spacy_model(self, name_spacy_model="fr_core_news_md"):
        """ Download Spacy pre-train model
        Args:
            name_spacy_model (str)
        """
        if self.apply_spacy_preprocessing:
            if '/' not in name_spacy_model:
                spacy.cli.download(name_spacy_model)
            if name_spacy_model != self.last_spacy_model_download:
                try:
                    self.nlp = spacy.load(name_spacy_model)
                    self.last_spacy_model_download = name_spacy_model
                except Exception:
                    logger.error("unknown spacy model name")

    def preprocessing_mandatory(self, data):

        data = data.drop([col for col in self.columns_to_remove if col in data.columns], axis=1)

        ###### Ordinal Encoding
        self.enc = OrdinalEncoder(dtype="int")
        for col in self.ordinal_features:
            if col in self.type_columns['numeric']:
                data[col] = interpolate_missing_data_numeric(data[col], self.method_nan_numeric)
            else:
                data[col] = interpolate_missing_data_categorical(data[col], self.method_nan_categorical)
        data[self.ordinal_features] = self.enc.fit_transform(data[self.ordinal_features])
        dump(self.enc, os.path.join(self.outdir_pre, "ordinalencoder.pkl"))

        for col in self.type_columns['numeric']:
            if col not in self.ordinal_features and col in data.columns:
                if data[col].dtypes not in ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32',
                                                 'int64', 'float32', 'float64']:
                    data[col] = data[col].astype('float32')

        for col in self.type_columns['categorical']:
            if col not in self.ordinal_features and col in data.columns:
                if data[col].dtypes in ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64',
                                             'float32', 'float64']:
                    data[col] = data[col].astype(str)

        if self.remove_categorical:
            data = remove_not_numeric(data)
            for col in data.columns:
                if col not in self.ordinal_features and data[col].isnull().sum() > 0:
                    data[col] = interpolate_missing_data_numeric(data[col], self.method_nan_numeric)

        else:
            for col in data.columns:
                if col in self.type_columns['categorical'] and col not in self.ordinal_features:
                    if data[col].isnull().sum() > 0:
                        data[col] = interpolate_missing_data_categorical(data[col], self.method_nan_categorical)
                if col in self.type_columns['numeric'] and col not in self.ordinal_features:
                    if data[col].isnull().sum() > 0:
                        data[col] = interpolate_missing_data_numeric(data[col], self.method_nan_numeric)

        self.apply_dummies = False
        for col_categorical in self.type_columns['categorical']:
            if col_categorical in data.columns:
                self.apply_dummies = True
                break

        self.info_feat_categorical = {}
        if self.apply_dummies:
            self.start_data = data.iloc[:100,:].copy()
            self.start_data.to_csv(os.path.join(self.outdir_pre, "start_data.csv"), index=False)

            for col_categorical in self.type_columns['categorical']:
                if col_categorical not in self.ordinal_features and col_categorical in data.columns:
                    dummies = one_hot_encode(data[col_categorical])
                    if dummies.shape[1] == 1:
                        dummies.columns = [col_categorical + '_' + dummies.columns[0]]
                    index_column = list(data.columns).index(col_categorical)
                    order_columns = list(data.columns)[:index_column] + list(dummies.columns) + list(data.columns)[(index_column + 1):]
                    data = pd.concat([dummies, data.drop([col_categorical], axis=1)], axis=1)
                    data = data[order_columns]
                    self.info_feat_categorical[col_categorical] = list(dummies.columns)
        return data

    def preprocessing_mandatory_transform(self, data_test):

        data_test = data_test.drop([col for col in self.columns_to_remove if col in data_test.columns], axis=1)

        ###### Ordinal Encoding
        for col in self.ordinal_features:
            if col in self.type_columns['numeric']:
                data_test[col] = interpolate_missing_data_numeric(data_test[col], self.method_nan_numeric)
            else:
                data_test[col] = interpolate_missing_data_categorical(data_test[col], self.method_nan_categorical)
        data_test[self.ordinal_features] = self.enc.transform(data_test[self.ordinal_features])

        for col in self.type_columns['numeric']:
            if col not in self.ordinal_features and col in data_test.columns:
                if data_test[col].dtypes not in ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32',
                                                 'int64', 'float32', 'float64']:
                    data_test[col] = data_test[col].astype('float32')

        for col in self.type_columns['categorical']:
            if col not in self.ordinal_features and col in data_test.columns:
                if data_test[col].dtypes in ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64',
                                             'float32', 'float64']:
                    data_test[col] = data_test[col].astype(str)

        if self.remove_categorical:
            data_test = remove_not_numeric(data_test)
            for col in data_test.columns:
                if col not in self.ordinal_features and data_test[col].isnull().sum() > 0:
                    data_test[col] = interpolate_missing_data_numeric(data_test[col], self.method_nan_numeric)

        else:
            for col in data_test.columns:
                if col in self.type_columns['categorical'] and col not in self.ordinal_features:
                    if data_test[col].isnull().sum() > 0:
                        data_test[col] = interpolate_missing_data_categorical(data_test[col],
                                                                              self.method_nan_categorical)
                if col in self.type_columns['numeric'] and col not in self.ordinal_features:
                    if data_test[col].isnull().sum() > 0:
                        data_test[col] = interpolate_missing_data_numeric(data_test[col], self.method_nan_numeric)

        self.apply_dummies = False
        for col_categorical in self.type_columns['categorical']:
            if col_categorical in data_test.columns:
                self.apply_dummies = True
                break

        if self.apply_dummies:
            concat_data = pd.concat([self.start_data, data_test], axis=0, ignore_index=True)

            for col_categorical in self.type_columns['categorical']:
                if col_categorical not in self.ordinal_features and col_categorical in concat_data.columns:
                    dummies = one_hot_encode(concat_data[col_categorical])
                    dummies_parallel = one_hot_encode(self.start_data[col_categorical])
                    if dummies.shape[1] == 1:
                        dummies.columns = [col_categorical + '_' + dummies.columns[0]]
                    if dummies_parallel.shape[1] == 1:
                        dummies_parallel.columns = [col_categorical + '_' + dummies_parallel.columns[0]]

                    for col in dummies.columns:
                        if col not in dummies_parallel.columns:
                            dummies = dummies.drop([col], axis=1)

                    if len(dummies.columns) != len(dummies_parallel.columns):
                        print('error in one_hot encoding categorical')

                    index_column = list(concat_data.columns).index(col_categorical)
                    order_columns = list(concat_data.columns)[:index_column] + list(dummies.columns) + list(
                        concat_data.columns)[(index_column + 1):]
                    concat_data = pd.concat([dummies, concat_data.drop([col_categorical], axis=1)], axis=1)
                    concat_data = concat_data[order_columns]

            data_test = concat_data[len(self.start_data):].reset_index(drop=True)  # !!!
        return data_test

    def build_feature_bin_numeric(self, data):
        dict_new_features = {}
        self.information_feature_bin_numeric = {}
        if len(self.bin_numeric_features) > 0:
            for col in self.bin_numeric_features:
                # base on optimal number:
                index_sample = rd.sample(list(data.index), int(len(data) * self.subsample))
                best_n_clusters = find_optimal_number(data[[col]].loc[index_sample])
                kmeans = KMeans(n_clusters=best_n_clusters, random_state=0)
                kmeans.fit(data[[col]])
                data[col + '_bin_' + str(best_n_clusters)] = kmeans.labels_
                dict_new_features[col] = col + '_bin_' + str(best_n_clusters)
                self.information_feature_bin_numeric[col] = (str(best_n_clusters), kmeans)

            features_importance = get_features_importance(data, self.Y, self.subsample, self.class_weight)

            self.col_drop_bin_numeric = []
            name_columns = data.columns
            for col in dict_new_features.keys():
                importance_col = features_importance[list(name_columns).index(col)]
                importance_new_col = features_importance[list(name_columns).index(dict_new_features[col])]
                if importance_col > importance_new_col:
                    data = data.drop([dict_new_features[col]], axis=1)
                    self.col_drop_bin_numeric.append(dict_new_features[col])

        pickle.dump(self.information_feature_bin_numeric, open(os.path.join(self.outdir_pre, "information_feature_bin_numeric.pkl"), "wb"))
        pickle.dump(self.col_drop_bin_numeric, open(os.path.join(self.outdir_pre, "col_drop_bin_numeric.pkl"), "wb"))

        return data

    def build_feature_bin_numeric_transform(self, data_test):
        if len(self.bin_numeric_features) > 0:
            for col in self.bin_numeric_features:
                # base on optimal number:
                best_n_clusters, kmeans = self.information_feature_bin_numeric[col]
                if col + '_bin_' + best_n_clusters not in self.col_drop_bin_numeric:
                    data_test[col + '_bin_' + best_n_clusters] = kmeans.predict(data_test[[col]])
        return data_test

    def build_feature_polynomial(self, data):
        columns_num = self.type_columns['numeric']
        dict_new_features = {col: [] for col in columns_num}

        self.col_power_2 = {}
        for col in columns_num:
            if len(data[col].unique()) > 2:
                data[col + '_power2'] = data[col] ** 2
                dict_new_features[col].append(col + '_power2')
                self.col_power_2[col + '_power2'] = col

        #self.col_multi_polynomial = {}
        #for i in range(len(columns_num)):
        #    for j in range(i + 1, len(columns_num)):
        #        data[columns_num[i] + '_multi_' + columns_num[j]] = data[columns_num[i]] * data[columns_num[j]]
        #        dict_new_features[columns_num[i]].append(columns_num[i] + '_multi_' + columns_num[j])
        #        dict_new_features[columns_num[j]].append(columns_num[i] + '_multi_' + columns_num[j])
        #        self.col_multi_polynomial[columns_num[i] + '_multi_' + columns_num[j]] = (
        #        columns_num[i], columns_num[j])

        features_importance = get_features_importance(data, self.Y, self.subsample, self.class_weight)

        name_columns = data.columns
        for col in dict_new_features.keys():
            for new_col in dict_new_features[col]:
                importance_col = features_importance[list(name_columns).index(col)]
                importance_new_col = features_importance[list(name_columns).index(new_col)]
                if importance_col > importance_new_col and new_col in data.columns:
                    data = data.drop([new_col], axis=1)
                    if new_col in self.col_power_2.keys():
                        del self.col_power_2[new_col]
                    #elif new_col in self.col_multi_polynomial.keys():
                    #    del self.col_multi_polynomial[new_col]
        pickle.dump(self.col_power_2, open(os.path.join(self.outdir_pre, "col_power_2.pkl"), "wb"))
        return data

    def build_feature_polynomial_transform(self, data_test):
        for col in self.col_power_2.values():
            data_test[col + '_power2'] = data_test[col] ** 2

        #for cols in self.col_multi_polynomial.values():
        #    data_test[cols[0] + '_multi_' + cols[1]] = data_test[cols[0]] * data_test[cols[1]]
        return data_test

    def build_feature_interaction(self, data):
        columns_num = self.type_columns['numeric']  ### feature engineering créer ne sont pas considéré !!!
        dict_new_features = {col: [] for col in columns_num}

        self.col_multi_interaction = {}
        for i in range(len(columns_num)):
            for j in range(i + 1, len(columns_num)):
                data[columns_num[i] + '_multi_' + columns_num[j]] = data[columns_num[i]] * data[columns_num[j]]
                dict_new_features[columns_num[i]].append(columns_num[i] + '_multi_' + columns_num[j])
                dict_new_features[columns_num[j]].append(columns_num[i] + '_multi_' + columns_num[j])
                self.col_multi_interaction[columns_num[i] + '_multi_' + columns_num[j]] = (
                columns_num[i], columns_num[j])

        features_importance = get_features_importance(data, self.Y, self.subsample, self.class_weight)

        name_columns = data.columns
        for col in dict_new_features.keys():
            for new_col in dict_new_features[col]:
                importance_col = features_importance[list(name_columns).index(col)]
                importance_new_col = features_importance[list(name_columns).index(new_col)]
                if importance_col > importance_new_col and new_col in data.columns:
                    data = data.drop([new_col], axis=1)
                    if new_col in self.col_multi_interaction.keys():
                        del self.col_multi_interaction[new_col]
        pickle.dump(self.col_multi_interaction, open(os.path.join(self.outdir_pre, "col_multi_interaction.pkl"), "wb"))
        return data

    def build_feature_interaction_transform(self, data_test):
        for cols in self.col_multi_interaction.values():
            data_test[cols[0] + '_multi_' + cols[1]] = data_test[cols[0]] * data_test[cols[1]]
        return data_test

    def build_feature_ratio(self, data):
        columns_num = self.type_columns['numeric']
        dict_new_features = {col: [] for col in columns_num}

        self.col_ratio = {}
        for i in range(len(columns_num)):
            for j in range(i + 1, len(columns_num)):
                data[columns_num[i] + '_ratio_' + columns_num[j]] = np.round(data[columns_num[i]] / (data[columns_num[j]] + 0.001), 3)
                dict_new_features[columns_num[i]].append(columns_num[i] + '_ratio_' + columns_num[j])
                dict_new_features[columns_num[j]].append(columns_num[i] + '_ratio_' + columns_num[j])
                self.col_ratio[columns_num[i] + '_ratio_' + columns_num[j]] = (columns_num[i], columns_num[j])

        features_importance = get_features_importance(data, self.Y, self.subsample, self.class_weight)

        name_columns = data.columns
        for col in dict_new_features.keys():
            for new_col in dict_new_features[col]:
                importance_col = features_importance[list(name_columns).index(col)]
                importance_new_col = features_importance[list(name_columns).index(new_col)]
                if importance_col > importance_new_col and new_col in data.columns:
                    data = data.drop([new_col], axis=1)
                    if new_col in self.col_ratio.keys():
                        del self.col_ratio[new_col]
        pickle.dump(self.col_ratio, open(os.path.join(self.outdir_pre, "col_ratio.pkl"), "wb"))
        return data

    def build_feature_ratio_transform(self, data_test):
        for cols in self.col_ratio.values():
            data_test[cols[0] + '_ratio_' + cols[1]] = np.round(data_test[cols[0]] / (data_test[cols[1]] + 0.001), 3)
        return data_test

    # Function to extract pca features
    def build_fe_pca(self, data):

        def create_pca(data_, kind, features, n_components):
            if features == 'all':
                features = self.base_features
            true_features = []
            for col in features:
                if col in self.info_feat_categorical.keys():
                    true_features = true_features + self.info_feat_categorical[col]
                else:
                    true_features.append(col)
            features = [col for col in true_features if col in data_.columns]
            data = data_[features].copy()
            if data.shape[1] > n_components:
                pca = PCA(n_components=n_components, random_state=15)
                data = pca.fit_transform(data)
                columns = [f'pca_{kind}{i + 1}' for i in range(data.shape[1])]
                data = pd.DataFrame(data, columns=columns)
                data = pd.concat([data_, data], axis=1)
                del data_
                return data, features, pca, columns
            else:
                del data
                return data_, features, 0, []

        self.info_pca_for_transform = {}
        for name in self.info_pca.keys():
            data, features, pca, columns = create_pca(data, name, self.info_pca[name][0], self.info_pca[name][1])
            if len(columns) > 0:
                self.info_pca_for_transform[name] = (features, pca, columns)
        pickle.dump(self.info_pca_for_transform, open(os.path.join(self.outdir_pre, "info_pca_for_transform.pkl"), "wb"))
        return data

    def build_fe_pca_transform(self, data_test):
        for name in self.info_pca_for_transform.keys():
            data = data_test[self.info_pca_for_transform[name][0]].copy()
            pca = self.info_pca_for_transform[name][1]
            data = pca.transform(data)
            data = pd.DataFrame(data, columns=self.info_pca_for_transform[name][2])
            data_test = pd.concat([data_test, data], axis=1)
        return data_test

    def build_fe_tsne(self, data):

        def create_tsne(data_, kind, features, n_components):
            if features == 'all':
                features = self.base_features
            true_features = []
            for col in features:
                if col in self.info_feat_categorical.keys():
                    true_features = true_features + self.info_feat_categorical[col]
                else:
                    true_features.append(col)
            features = [col for col in true_features if col in data_.columns]
            data = data_[features].copy()
            if data.shape[1] > n_components:
                tsne = TSNE(n_components=n_components, random_state=15, verbose=0)
                data = tsne.fit_transform(data)
                columns = [f'tsne_{kind}{i + 1}' for i in range(data.shape[1])]
                data = pd.DataFrame(data, columns=columns)
                data = pd.concat([data_, data], axis=1)
                del data_
                return data, features, tsne, columns
            else:
                del data
                return data_, features, 0, []

        self.info_tsne_for_transform = {}
        for name in self.info_tsne.keys():
            data, features, tsne, columns = create_tsne(data, name, self.info_tsne[name][0], self.info_tsne[name][1])
            if len(columns) > 0:
                self.info_tsne_for_transform[name] = (features, tsne, columns)
        pickle.dump(self.info_tsne_for_transform, open(os.path.join(self.outdir_pre, "info_tsne_for_transform.pkl"), "wb"))
        return data

    def build_fe_tsne_transform(self, data_test):
        for name in self.info_tsne_for_transform.keys():
            data = data_test[self.info_tsne_for_transform[name][0]].copy()
            tsne = self.info_tsne_for_transform[name][1]
            data = tsne.transform(data)
            data = pd.DataFrame(data, columns=self.info_tsne_for_transform[name][2])
            data_test = pd.concat([data_test, data], axis=1)
        return data_test

    def build_fe_stats(self, data):

        self.info_stats_for_transform = {}
        for name in self.info_stats.keys():

            features = self.info_stats[name][1]
            if features == 'all':
                features = self.base_features
            true_features = []
            for col in features:
                if col in self.info_feat_categorical.keys():
                    true_features = true_features + self.info_feat_categorical[col]
                else:
                    true_features.append(col)
            features = [col for col in true_features if col in data.columns]

            if type(self.info_stats[name][0]) is list:
                method = self.info_stats[name][0]
            else:
                method = [self.info_stats[name][0]]
            if len(features) >= 2:
                if 'sum' in method:
                    data['sum_' + name] = data[features].sum(axis=1)
                    self.info_stats_for_transform[name] = ('sum', features, 'sum_' + name)
                if 'mean' in method:
                    data['mean_' + name] = data[features].mean(axis=1)
                    self.info_stats_for_transform[name] = ('mean', features, 'mean_' + name)
                if 'std' in method:
                    data['std_' + name] = data[features].std(axis=1)
                    self.info_stats_for_transform[name] = ('std', features, 'std_' + name)
                if 'kurtosis' in method:
                    data['kurtosis_' + name] = data[features].kurtosis(axis=1)
                    self.info_stats_for_transform[name] = ('kurtosis', features, 'kurtosis_' + name)
                if 'skew' in method:
                    data['skew_' + name] = data[features].skew(axis=1)
                    self.info_stats_for_transform[name] = ('skew', features, 'skew_' + name)

                if 'multi' in method:
                    data_multi = data[features[0]]
                    for col in features[1:]:
                        data_multi = data_multi * data[col]
                    name_column = 'multi_' + '_'.join(features)
                    data[name] = data_multi
                    self.info_stats_for_transform[name] = ('multi', features, name)

                if 'div' in method:
                    if len(features) == 2:
                        data[name] = np.round(data[features[0]] / (data[features[1]] + 0.001), 3)
                        self.info_stats_for_transform[name] = ('div', features, name)

                if 'power' in method:
                    for col in features:
                        data[name] = data[col] ** 2
                        data = data.drop([col], axis=1)
                    self.info_stats_for_transform[name] = ('power', features, name)

        self.remove_column_correlation = []
        for name in self.info_stats_for_transform.keys():
            method, features, col_name = self.info_stats_for_transform[name]
            if method in ['multi', 'div', 'sum', 'mean']:
                for col in features:
                    if col in data.columns and data.corr().loc[col_name, col] > 0.85:
                        data = data.drop([col], axis=1)
                        print('column', col, 'removes due to high correlation (>0.85) with', col_name)
                        self.remove_column_correlation.append(col)

        pickle.dump(self.info_stats_for_transform, open(os.path.join(self.outdir_pre, "info_stats_for_transform.pkl"), "wb"))
        pickle.dump(self.remove_column_correlation, open(os.path.join(self.outdir_pre, "remove_column_correlation.pkl"), "wb"))
        return data

    def build_fe_stats_transform(self, data_test):
        for name in self.info_stats_for_transform.keys():
            method, features, col_name = self.info_stats_for_transform[name]
            if 'sum' in method:
                data_test[col_name] = data_test[features].sum(axis=1)
            if 'mean' in method:
                data_test[col_name] = data_test[features].mean(axis=1)
            if 'std' in method:
                data_test[col_name] = data_test[features].std(axis=1)
            if 'kurtosis' in method:
                data_test[col_name] = data_test[features].kurtosis(axis=1)
            if 'skew' in method:
                data_test[col_name] = data_test[features].skew(axis=1)

            if 'multi' in method:
                data_multi = data_test[features[0]]
                for col in features[1:]:
                    data_multi = data_multi * data_test[col]
                data_test[col_name] = data_multi

            if 'div' in method:
                if len(features) == 2:
                    data_test[col_name] = np.round(data_test[features[0]] / (data_test[features[1]] + 0.001), 3)

            if 'power' in method:
                for col in features:
                    data_test[col_name] = data_test[col] ** 2
                    data_test = data_test.drop([col], axis=1)

        for col in self.remove_column_correlation:
            data_test = data_test.drop([col], axis=1)
        return data_test

    def feature_selection_VarianceThreshold(self, data):

        Numeric_features = list(data.loc[:, data.dtypes.astype(str).isin(
            ['uint8', 'int8', 'int16', 'int32', 'int64', 'float32', 'float64'])].columns)
        Numeric_features = [col for col in Numeric_features if col not in list(self.info_feat_categorical.values())]

        thresh = np.percentile([np.var(data[col]) for col in Numeric_features], self.remove_percentage)

        self.columns_to_drop_variance = []
        for col in Numeric_features:
            if np.var(data[col]) <= thresh:
                self.columns_to_drop_variance.append(col)
        print('columns remove due to low variance :\n')
        print(self.columns_to_drop_variance)
        data = data.drop(self.columns_to_drop_variance, axis=1)
        pickle.dump(self.columns_to_drop_variance, open(os.path.join(self.outdir_pre, "columns_to_drop_variance.pkl"), "wb"))
        return data

    def feature_selection_VarianceThreshold_transform(self, data_test):
        data_test = data_test.drop(self.columns_to_drop_variance, axis=1)
        return data_test

    def fct_remove_multicollinearity(self, data):
        name_columns = data.columns
        matrix_corr = np.array(data.corr())

        features_importance = get_features_importance(data, self.Y, self.subsample, self.class_weight)

        self.columns_to_drop_multicollinearity = []
        for i in range(len(name_columns)):
            for j in range(i + 1, len(name_columns)):
                if name_columns[i] not in self.columns_to_drop_multicollinearity and name_columns[
                    j] not in self.columns_to_drop_multicollinearity:
                    cor_ij = matrix_corr[i][j]
                    if np.abs(cor_ij) > self.multicollinearity_threshold:
                        importance_i = features_importance[list(name_columns).index(name_columns[i])]
                        importance_j = features_importance[list(name_columns).index(name_columns[j])]

                        if importance_i >= importance_j:
                            self.columns_to_drop_multicollinearity.append(name_columns[j])
                        else:
                            self.columns_to_drop_multicollinearity.append(name_columns[i])
        print('columns remove due to high multicollinearity :\n')
        print(self.columns_to_drop_multicollinearity)
        data = data.drop(self.columns_to_drop_multicollinearity, axis=1)
        pickle.dump(self.columns_to_drop_multicollinearity, open(os.path.join(self.outdir_pre, "columns_to_drop_multicollinearity.pkl"), "wb"))
        return data

    def fct_remove_multicollinearity_transform(self, data_test):
        data_test = data_test.drop(self.columns_to_drop_multicollinearity, axis=1)
        return data_test

    def select_feature_by_importance(self, data):
        name_columns = data.columns

        features_importance = get_features_importance(data, self.Y, self.subsample, self.class_weight)

        sorted_idx = features_importance.argsort()
        name_columns_sorted = name_columns[sorted_idx][::-1]

        nb_column_to_keep = int(self.feature_selection_threshold * len(name_columns))
        print('\ncolumns remove due to low importance value :\n')
        self.columns_to_drop_importance = list(name_columns_sorted)[nb_column_to_keep:]
        print(self.columns_to_drop_importance)
        data = data.drop(self.columns_to_drop_importance, axis=1)
        pickle.dump(self.columns_to_drop_importance, open(os.path.join(self.outdir_pre, "columns_to_drop_importance.pkl"), "wb"))
        return data

    def select_feature_by_importance_transform(self, data_test):
        data_test = data_test.drop(self.columns_to_drop_importance, axis=1)
        return data_test

    ##################
    # Time Series
    ##################

    def build_lag_features(self, data, position_id):
        """ Lag features : take the value from i step_date of the current date"""

        print('Shifting:', self.step_lags)
        if position_id is None:
            for col in self.target:
                for i in self.step_lags:
                    data[col + '_lag_' + str(i)] = self.Y[col].transform(lambda x: x.shift(i)).fillna(0)
        else:
            if isinstance(position_id, str):
                dt = pd.concat([data[[position_id]], self.Y], axis=1)
            else:
                dt = pd.concat([position_id, self.Y], axis=1)
            for i in self.step_lags:
                new_name_column = [col + '_lag_' + str(i) for col in self.target]
                data[new_name_column] = dt.groupby(dt.columns[0], as_index=False)[self.target].shift(i).fillna(0)
        return data

    def build_rolling_features(self, data, position_id):
        """ Rolling features : mean and std of i last step_date """

        print('Rolling period:', self.step_rolling)
        if position_id is None:
            for col in self.target:
                for i in self.step_rolling:
                    data[col + '_rolling_mean_' + str(i)] = self.Y[col].transform(
                        lambda x: x.rolling(i, win_type=self.win_type).mean()).fillna(0)
                    data[col + '_rolling_std_' + str(i)] = self.Y[col].transform(
                        lambda x: x.rolling(i, win_type=self.win_type).std()).fillna(0)
        else:
            if isinstance(position_id, str):
                dt = pd.concat([data[[position_id]], self.Y], axis=1)
            else:
                dt = pd.concat([position_id, self.Y], axis=1)
            for i in self.step_rolling:
                new_name_column = [col + '_rolling_mean_' + str(i) for col in self.target]
                data[new_name_column] = dt.groupby(dt.columns[0], as_index=False)[self.target].transform(
                    lambda x: x.rolling(window=i, win_type=self.win_type).mean()).fillna(0)
                new_name_column = [col + '_rolling_std_' + str(i) for col in self.target]
                data[new_name_column] = dt.groupby(dt.columns[0], as_index=False)[self.target].transform(
                    lambda x: x.rolling(window=i, win_type=self.win_type).std()).fillna(0)
        return data

    def print_feature_importances(self, data):
        features_importance = get_features_importance(data, self.Y, self.subsample, self.class_weight)
        sorted_idx = features_importance.argsort()
        plt.barh(data.columns[sorted_idx], features_importance[sorted_idx])
        plt.show()

    def print_feature_correlation(self, data):
        sns.set(rc={'figure.figsize': (12, 12)})
        sns.heatmap(data.corr(), annot=True, cmap=sns.cm.rocket_r)
        plt.show()

    def transform_nlp(self, data):
        """ Fit and transform data :
            + can apply a small cleaning on text column (self.apply_small_clean)
            + preprocess text column with nlp.pipe spacy (self.apply_spacy_preprocessing)
            + replace Named entities  (self.apply_entity_preprocessing)
        Return:
            data (DataFrame) only have one column : column_text
        """

        #data_copy = data.copy()

        self.load_spacy_model(self.name_spacy_model)

        if self.apply_small_clean:
            logger.info("- Apply small clean of texts...")
            data[self.column_text] = data[self.column_text].apply(lambda text: small_clean_text(text))

        if self.apply_spacy_preprocessing:
            logger.info("- Apply nlp.pipe from spacy...")
            doc_spacy_data = nlp_preprocessing_spacy(data[self.column_text], self.nlp,
                                                          disable_ner=self.apply_entity_preprocessing)
        else:
            doc_spacy_data = None

        if self.apply_entity_preprocessing:
            logger.info("- Apply entities preprocessing...")
            data[self.column_text] = transform_entities(doc_spacy_data)
            doc_spacy_data = nlp_preprocessing_spacy(data[self.column_text], self.nlp, disable_ner=True)

        # keep only the column with texts
        return list(data[self.column_text]), doc_spacy_data

    def fit_transform(self, data):

        for col in self.target:
            if col in data.columns:
                data = data.drop([col], axis=1)

        if self.column_text is not None:
            if self.column_text in data.columns:
                list_texts, doc_spacy_data = self.transform_nlp(data[self.column_text])
                data = data.drop([self.column_text], axis=1)
            else:
                logger.warning("you give a name for column_text : {}. But it is not in data columns.".format(self.column_text))
                self.column_text = None
                doc_spacy_data = None
        else:
            self.column_text = None
            doc_spacy_data = None

        # Position ID for time_series objective
        if self.flags_parameters.position_id is not None and self.flags_parameters.position_id in data.columns and isinstance(
                self.flags_parameters.position_id, str):
            position_id = data[[self.flags_parameters.position_id]]
        else:
            position_id = self.flags_parameters.position_id

        if self.apply_preprocessing_mandatory:
            data = self.preprocessing_mandatory(data)

        if len(self.bin_numeric_features) > 0:
            data = self.build_feature_bin_numeric(data)
        if self.polynomial_features:
            data = self.build_feature_polynomial(data)
        if self.feature_interaction:
            data = self.build_feature_interaction(data)
        if self.feature_ratio:
            data = self.build_feature_ratio(data)

        # for time series:
        if len(self.step_lags) > 0:
            data = self.build_lag_features(data, position_id)
        if len(self.step_rolling) > 0:
            data = self.build_rolling_features(data, position_id)

        if len(self.info_tsne.keys()) > 0:
            data = self.build_fe_tsne(data)
        if len(self.info_pca.keys()) > 0:
            data = self.build_fe_pca(data)
        if len(self.info_stats.keys()) > 0:
            data = self.build_fe_stats(data)
        if self.remove_low_variance:
            data = self.feature_selection_VarianceThreshold(data)
        if self.remove_multicollinearity:
            data = self.fct_remove_multicollinearity(data)
        if self.feature_selection:
            data = self.select_feature_by_importance(data)

        if self.column_text is not None:
            data[self.column_text] = list_texts

        self.list_final_columns = list(data.columns)
        pickle.dump(self.list_final_columns, open(os.path.join(self.outdir_pre, "list_final_columns.pkl"), "wb"))

        return data, doc_spacy_data, position_id

    def transform(self, data_test):

        if self.target is not None:
            for col in self.target:
                if col in data_test.columns:
                    data_test = data_test.drop([col], axis=1)

        if self.column_text is not None:
            list_texts, doc_spacy_data_test = self.transform_nlp(data_test[self.column_text])
            data_test = data_test.drop([self.column_text], axis=1)
        else:
            doc_spacy_data_test = None

        # Position ID for time_series objective
        if self.flags_parameters.position_id is not None and self.flags_parameters.position_id in data_test.columns and isinstance(
                    self.flags_parameters.position_id, str):
            position_id_test = data_test[[self.flags_parameters.position_id]]
        else:
            position_id_test = self.flags_parameters.position_id

        if self.apply_preprocessing_mandatory:
            data_test = self.preprocessing_mandatory_transform(data_test)
        if len(self.bin_numeric_features) > 0:
            data_test = self.build_feature_bin_numeric_transform(data_test)
        if self.polynomial_features:
            data_test = self.build_feature_polynomial_transform(data_test)
        if self.feature_interaction:
            data_test = self.build_feature_interaction_transform(data_test)
        if self.feature_ratio:
            data_test = self.build_feature_ratio_transform(data_test)
        if len(self.info_tsne.keys()) > 0:
            data_test = self.build_fe_tsne_transform(data_test)
        if len(self.info_pca.keys()) > 0:
            data_test = self.build_fe_pca_transform(data_test)
        if len(self.info_stats.keys()) > 0:
            data_test = self.build_fe_stats_transform(data_test)
        if self.remove_low_variance:
            data_test = self.feature_selection_VarianceThreshold_transform(data_test)
        if self.remove_multicollinearity:
            data_test = self.fct_remove_multicollinearity_transform(data_test)
        if self.feature_selection:
            data_test = self.select_feature_by_importance_transform(data_test)

        if self.column_text is not None:
            data_test[self.column_text] = list_texts

        for col in data_test.columns:
            if col not in self.list_final_columns:
                print(col, 'is not in original data')

        for col in self.list_final_columns:
            if col not in data_test.columns:
                print(col, 'is not in test data')

        return data_test, doc_spacy_data_test, position_id_test

    def load_parameters(self):
        try:
            self.enc = load(os.path.join(self.outdir_pre, "ordinalencoder.pkl"))
        except:
            pass
        try:
            self.start_data = pd.read_csv(os.path.join(self.outdir_pre, "start_data.csv"))
        except:
            pass
        try:
            self.information_feature_bin_numeric = pickle.load(open(os.path.join(self.outdir_pre, "information_feature_bin_numeric.pkl"), "rb"))
        except:
            pass
        try:
            self.col_drop_bin_numeric = pickle.load(open(os.path.join(self.outdir_pre, "col_drop_bin_numeric.pkl"), "rb"))
        except:
            pass
        try:
            self.col_power_2 = pickle.load(open(os.path.join(self.outdir_pre, "col_power_2.pkl"), "rb"))
        except:
            pass
        try:
            self.col_multi_interaction = pickle.load(open(os.path.join(self.outdir_pre, "col_multi_interaction.pkl"), "rb"))
        except:
            pass
        try:
            self.col_ratio = pickle.load(open(os.path.join(self.outdir_pre, "col_ratio.pkl"), "rb"))
        except:
            pass
        try:
            self.info_pca_for_transform = pickle.load(open(os.path.join(self.outdir_pre, "info_pca_for_transform.pkl"), "rb"))
        except:
            pass
        try:
            self.info_tsne_for_transform = pickle.load(open(os.path.join(self.outdir_pre, "info_tsne_for_transform.pkl"), "rb"))
        except:
            pass
        try:
            self.info_stats_for_transform = pickle.load(open(os.path.join(self.outdir_pre, "info_stats_for_transform.pkl"), "rb"))
        except:
            pass
        try:
            self.remove_column_correlation = pickle.load(open(os.path.join(self.outdir_pre, "remove_column_correlation.pkl"), "rb"))
        except:
            pass
        try:
            self.columns_to_drop_variance = pickle.load(open(os.path.join(self.outdir_pre, "columns_to_drop_variance.pkl"), "rb"))
        except:
            pass
        try:
            self.columns_to_drop_multicollinearity = pickle.load(open(os.path.join(self.outdir_pre, "columns_to_drop_multicollinearity.pkl"), "rb"))
        except:
            pass
        try:
            self.columns_to_drop_importance = pickle.load(open(os.path.join(self.outdir_pre, "columns_to_drop_importance.pkl"), "rb"))
        except:
            pass
        try:
            self.list_final_columns = pickle.load(open(os.path.join(self.outdir_pre, "list_final_columns.pkl"), "rb"))
        except:
            pass