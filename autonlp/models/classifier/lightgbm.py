from ...models.classifier.trainer import Model
import lightgbm as lgb
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from hyperopt import hp
import numpy as np
import os
import json


class ML_LightGBM(Model):
    name_classifier = 'LightGBM'
    is_NN = False

    def __init__(self, flags_parameters, name_model_full, class_weight=None):
        Model.__init__(self, flags_parameters, name_model_full, class_weight)

    def hyper_params(self, size_params='small'):
        parameters = dict()
        if size_params == 'small':
            if self.flags_parameters.lgbm_n_estimators_min == self.flags_parameters.lgbm_n_estimators_max:
                parameters['n_estimators'] = hp.choice('n_estimators', [self.flags_parameters.lgbm_n_estimators_min])
            else:
                parameters['n_estimators'] = hp.randint('n_estimators', self.flags_parameters.lgbm_n_estimators_min,
                                                        self.flags_parameters.lgbm_n_estimators_max)
            if self.flags_parameters.lgbm_num_leaves_min == self.flags_parameters.lgbm_num_leaves_max:
                parameters['num_leaves'] = hp.choice('num_leaves', [self.flags_parameters.lgbm_num_leaves_min])
            else:
                parameters['num_leaves'] = hp.randint('num_leaves', self.flags_parameters.lgbm_num_leaves_min,
                                                      self.flags_parameters.lgbm_num_leaves_max)
            if self.flags_parameters.lgbm_learning_rate_min == self.flags_parameters.lgbm_learning_rate_max:
                parameters['learning_rate'] = hp.choice('learning_rate', [self.flags_parameters.lgbm_learning_rate_min])
            else:
                parameters['learning_rate'] = hp.loguniform('learning_rate', np.log(self.flags_parameters.lgbm_learning_rate_min),
                                                            np.log(self.flags_parameters.lgbm_learning_rate_max))
            if self.flags_parameters.lgbm_bagging_fraction_min == self.flags_parameters.lgbm_bagging_fraction_max:
                parameters['bagging_fraction'] = hp.choice('bagging_fraction', [self.flags_parameters.lgbm_bagging_fraction_min])
            else:
                parameters['bagging_fraction'] = hp.uniform('bagging_fraction', self.flags_parameters.lgbm_bagging_fraction_min,
                                                            self.flags_parameters.lgbm_bagging_fraction_max)
        else:
            if self.flags_parameters.lgbm_n_estimators_min == self.flags_parameters.lgbm_n_estimators_max:
                parameters['n_estimators'] = hp.choice('n_estimators', [self.flags_parameters.lgbm_n_estimators_min])
            else:
                parameters['n_estimators'] = hp.randint('n_estimators', self.flags_parameters.lgbm_n_estimators_min,
                                                        self.flags_parameters.lgbm_n_estimators_max)
            if self.flags_parameters.lgbm_num_leaves_min == self.flags_parameters.lgbm_num_leaves_max:
                parameters['num_leaves'] = hp.choice('num_leaves', [self.flags_parameters.lgbm_num_leaves_min])
            else:
                parameters['num_leaves'] = hp.randint('num_leaves', self.flags_parameters.lgbm_num_leaves_min,
                                                      self.flags_parameters.lgbm_num_leaves_max)
            if self.flags_parameters.lgbm_learning_rate_min == self.flags_parameters.lgbm_learning_rate_max:
                parameters['learning_rate'] = hp.choice('learning_rate', [self.flags_parameters.lgbm_learning_rate_min])
            else:
                parameters['learning_rate'] = hp.loguniform('learning_rate',
                                                            np.log(self.flags_parameters.lgbm_learning_rate_min),
                                                            np.log(self.flags_parameters.lgbm_learning_rate_max))
            if self.flags_parameters.lgbm_bagging_fraction_min == self.flags_parameters.lgbm_bagging_fraction_max:
                parameters['bagging_fraction'] = hp.choice('bagging_fraction',
                                                           [self.flags_parameters.lgbm_bagging_fraction_min])
            else:
                parameters['bagging_fraction'] = hp.uniform('bagging_fraction',
                                                            self.flags_parameters.lgbm_bagging_fraction_min,
                                                            self.flags_parameters.lgbm_bagging_fraction_max)

            parameters['min_child_weight'] = hp.loguniform('min_child_weight', np.log(1e-3), np.log(0.1))  # default 1e-3
            parameters['feature_fraction'] = hp.uniform('feature_fraction', 0.5, 1)  # default 1
            parameters['reg_alpha'] = hp.loguniform('reg_alpha', np.log(1e-2), np.log(0.5))  # default 0
            parameters['reg_lambda'] = hp.loguniform('reg_lambda', np.log(1e-2), np.log(0.5))  # default 0

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
            clf = lgb.LGBMRegressor(
                random_state=self.seed,
                class_weight=self.class_weight,
                **self.p
            )
            if self.shape_y == 1:
                return clf
            else:
                return MultiOutputRegressor(clf)
        else:
            clf = lgb.LGBMClassifier(
                random_state=self.seed,
                class_weight=self.class_weight,
                **self.p
            )
            if self.shape_y == 1:
                return clf
            else:
                return MultiOutputClassifier(clf)
