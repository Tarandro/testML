from ...models.classifier.trainer import Model
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from hyperopt import hp
import numpy as np
import os
import json


class ML_XGBoost(Model):
    name_classifier = 'XGBoost'
    is_NN = False

    def __init__(self, flags_parameters, name_model_full, class_weight=None, len_unique_value={},
                 time_series_features=None, scaler_info=None):
        Model.__init__(self, flags_parameters, name_model_full, class_weight, len_unique_value, time_series_features,
                       scaler_info)

    def hyper_params(self, size_params='small'):
        parameters = dict()
        if size_params == 'small':
            if self.flags_parameters.xgb_n_estimators_min == self.flags_parameters.xgb_n_estimators_max:
                parameters['n_estimators'] = hp.choice('n_estimators', [self.flags_parameters.xgb_n_estimators_min])
            else:
                parameters['n_estimators'] = hp.randint('n_estimators', self.flags_parameters.xgb_n_estimators_min,
                                                        self.flags_parameters.xgb_n_estimators_max)
            if self.flags_parameters.xgb_max_depth_min == self.flags_parameters.xgb_max_depth_max:
                parameters['max_depth'] = hp.choice('max_depth', [self.flags_parameters.xgb_max_depth_min])
            else:
                parameters['max_depth'] = hp.randint('max_depth', self.flags_parameters.xgb_max_depth_min,
                                                     self.flags_parameters.xgb_max_depth_max)
            if self.flags_parameters.xgb_learning_rate_min == self.flags_parameters.xgb_learning_rate_max:
                parameters['learning_rate'] = hp.choice('learning_rate', [self.flags_parameters.xgb_learning_rate_min])
            else:
                parameters['learning_rate'] = hp.loguniform('learning_rate', np.log(self.flags_parameters.xgb_learning_rate_min),
                                                            np.log(self.flags_parameters.xgb_learning_rate_max))
            if self.flags_parameters.xgb_subsample_min == self.flags_parameters.xgb_subsample_max:
                parameters['subsample'] = hp.choice('subsample', [self.flags_parameters.xgb_subsample_min])
            else:
                parameters['subsample'] = hp.uniform('subsample', self.flags_parameters.xgb_subsample_min,
                                                     self.flags_parameters.xgb_subsample_max)
        else:
            if self.flags_parameters.xgb_n_estimators_min == self.flags_parameters.xgb_n_estimators_max:
                parameters['n_estimators'] = hp.choice('n_estimators', [self.flags_parameters.xgb_n_estimators_min])
            else:
                parameters['n_estimators'] = hp.randint('n_estimators', self.flags_parameters.xgb_n_estimators_min,
                                                        self.flags_parameters.xgb_n_estimators_max)
            if self.flags_parameters.xgb_max_depth_min == self.flags_parameters.xgb_max_depth_max:
                parameters['max_depth'] = hp.choice('max_depth', [self.flags_parameters.xgb_max_depth_min])
            else:
                parameters['max_depth'] = hp.randint('max_depth', self.flags_parameters.xgb_max_depth_min,
                                                     self.flags_parameters.xgb_max_depth_max)
            if self.flags_parameters.xgb_learning_rate_min == self.flags_parameters.xgb_learning_rate_max:
                parameters['learning_rate'] = hp.choice('learning_rate', [self.flags_parameters.xgb_learning_rate_min])
            else:
                parameters['learning_rate'] = hp.loguniform('learning_rate',
                                                            np.log(self.flags_parameters.xgb_learning_rate_min),
                                                            np.log(self.flags_parameters.xgb_learning_rate_max))
            if self.flags_parameters.xgb_subsample_min == self.flags_parameters.xgb_subsample_max:
                parameters['subsample'] = hp.choice('subsample', [self.flags_parameters.xgb_subsample_min])
            else:
                parameters['subsample'] = hp.uniform('subsample', self.flags_parameters.xgb_subsample_min,
                                                     self.flags_parameters.xgb_subsample_max)

            parameters['min_child_weight'] = hp.loguniform('min_child_weight', np.log(1e-1), np.log(4))  # default 1
            parameters['eta'] = hp.uniform('eta', 0, 1)  # default 0.3
            parameters['gamma'] = hp.loguniform('gamma', np.log(1e-2), np.log(3))  # default 0
            parameters['reg_alpha'] = hp.loguniform('reg_alpha', np.log(1e-2), np.log(0.5))  # default 0
            parameters['reg_lambda'] = hp.loguniform('reg_lambda', np.log(1e-2), np.log(1))  # default 1

        return parameters

    def initialize_params(self, y, params):
        self.shape_y = y.shape[1]
        if self.shape_y > 1:
            params = {'estimator__'+k: v for k, v in params.items()}
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
            clf = xgb.XGBRegressor(
                random_state=self.seed,
                **self.p
            )
            if self.shape_y == 1:
                return clf
            else:
                return MultiOutputRegressor(clf)
        else:
            clf = xgb.XGBClassifier(
                random_state=self.seed,
                **self.p  # ,
                # scale_pos_weight = count(negative examples)/count(Positive examples)
            )
            if self.shape_y == 1:
                return clf
            else:
                return MultiOutputClassifier(clf)
