from ...models.classifier.trainer import Model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from hyperopt import hp
import numpy as np
import os
import json


class ML_RandomForest(Model):
    name_classifier = 'RandomForest'
    is_NN = False

    def __init__(self, flags_parameters, name_model_full, class_weight=None, len_unique_value={},
                 time_series_features=None, scaler_info=None, position_id=None):
        Model.__init__(self, flags_parameters, name_model_full, class_weight, len_unique_value, time_series_features,
                       scaler_info, position_id)

    def hyper_params(self, size_params='small'):
        parameters = dict()
        if size_params == 'small':
            if self.flags_parameters.rf_n_estimators_min == self.flags_parameters.rf_n_estimators_max:
                parameters['n_estimators'] = hp.choice('n_estimators', [self.flags_parameters.rf_n_estimators_min])
            else:
                parameters['n_estimators'] = hp.randint('n_estimators', self.flags_parameters.rf_n_estimators_min,
                                                        self.flags_parameters.rf_n_estimators_max)
            if self.flags_parameters.rf_max_depth_min == self.flags_parameters.rf_max_depth_max:
                parameters['max_depth'] = hp.choice('max_depth', [self.flags_parameters.rf_max_depth_min])
            else:
                parameters['max_depth'] = hp.randint('max_depth', self.flags_parameters.rf_max_depth_min,
                                                     self.flags_parameters.rf_max_depth_max)
            if self.flags_parameters.rf_min_samples_split_min == self.flags_parameters.rf_min_samples_split_max:
                parameters['min_samples_split'] = hp.choice('min_samples_split', [self.flags_parameters.rf_min_samples_split_min])
            else:
                parameters['min_samples_split'] = hp.randint('min_samples_split', self.flags_parameters.rf_min_samples_split_min,
                                                             self.flags_parameters.rf_min_samples_split_max)
            if self.flags_parameters.rf_max_samples_min == self.flags_parameters.rf_max_samples_max:
                parameters['max_samples'] = hp.choice('max_samples', [self.flags_parameters.rf_max_samples_min])
            else:
                parameters['max_samples'] = hp.uniform('max_samples', self.flags_parameters.rf_max_samples_min,
                                                       self.flags_parameters.rf_max_samples_max)
        else:
            if self.flags_parameters.rf_n_estimators_min == self.flags_parameters.rf_n_estimators_max:
                parameters['n_estimators'] = hp.choice('n_estimators', [self.flags_parameters.rf_n_estimators_min])
            else:
                parameters['n_estimators'] = hp.randint('n_estimators', self.flags_parameters.rf_n_estimators_min,
                                                        self.flags_parameters.rf_n_estimators_max)
            if self.flags_parameters.rf_max_depth_min == self.flags_parameters.rf_max_depth_max:
                parameters['max_depth'] = hp.choice('max_depth', [self.flags_parameters.rf_max_depth_min])
            else:
                parameters['max_depth'] = hp.randint('max_depth', self.flags_parameters.rf_max_depth_min,
                                                     self.flags_parameters.rf_max_depth_max)
            if self.flags_parameters.rf_min_samples_split_min == self.flags_parameters.rf_min_samples_split_max:
                parameters['min_samples_split'] = hp.choice('min_samples_split',
                                                            [self.flags_parameters.rf_min_samples_split_min])
            else:
                parameters['min_samples_split'] = hp.randint('min_samples_split',
                                                             self.flags_parameters.rf_min_samples_split_min,
                                                             self.flags_parameters.rf_min_samples_split_max)
            if self.flags_parameters.rf_max_samples_min == self.flags_parameters.rf_max_samples_max:
                parameters['max_samples'] = hp.choice('max_samples', [self.flags_parameters.rf_max_samples_min])
            else:
                parameters['max_samples'] = hp.uniform('max_samples', self.flags_parameters.rf_max_samples_min,
                                                       self.flags_parameters.rf_max_samples_max)

            if 'regression' in self.objective:
                parameters['criterion'] = hp.choice('criterion', ['mse', 'mae'])
            else:
                parameters['criterion'] = hp.choice('criterion', ['gini', 'entropy'])

        return parameters

    def initialize_params(self, y, params):
        self.shape_y = y.shape[1]
        self.p = params

    def save_params(self, outdir_model):
        params_all = dict()

        p_model = self.p.copy()
        params_all['p_model'] = p_model
        params_all['name_classifier'] = self.name_classifier
        params_all['shape_y'] = self.shape_y

        self.params_all = {self.name_model_full: params_all}

        if self.apply_logs:
            with open(os.path.join(outdir_model, "parameters.json"), "w") as outfile:
                json.dump(self.params_all, outfile)

    def load_params(self, params_all, outdir):
        p_model = params_all['p_model']
        self.p = p_model
        self.shape_y = params_all['shape_y']

    def model(self, hyper_params_clf={}):
        if 'regression' in self.objective:
            return RandomForestRegressor(
                random_state=self.seed,
                **self.p
            )
        else:
            clf = RandomForestClassifier(
                random_state=self.seed,
                class_weight=self.class_weight,
                **self.p
            )
        return clf
