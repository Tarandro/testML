from ...models.classifier.trainer import Model
import catboost as cat
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.utils.class_weight import compute_class_weight
from hyperopt import hp
import numpy as np
import os
import json


class ML_CatBoost(Model):
    name_classifier = 'CatBoost'
    is_NN = False

    def __init__(self, flags_parameters, name_model_full, class_weight=None, len_unique_value={},
                 time_series_features=None, scaler_info=None):
        Model.__init__(self, flags_parameters, name_model_full, class_weight, len_unique_value, time_series_features,
                       scaler_info)

    def hyper_params(self, size_params='small'):
        parameters = dict()
        if size_params == 'small':
            if self.flags_parameters.cat_iterations_min == self.flags_parameters.cat_iterations_max:
                parameters['iterations'] = hp.choice('iterations', [self.flags_parameters.cat_iterations_min])
            else:
                parameters['iterations'] = hp.randint('iterations', self.flags_parameters.cat_iterations_min,
                                                      self.flags_parameters.cat_iterations_max)
            if self.flags_parameters.cat_depth_min == self.flags_parameters.cat_depth_max:
                parameters['depth'] = hp.choice('depth', [self.flags_parameters.cat_depth_min])
            else:
                parameters['depth'] = hp.randint('depth', self.flags_parameters.cat_depth_min,
                                                 self.flags_parameters.cat_depth_max)
            if self.flags_parameters.cat_learning_rate_min == self.flags_parameters.cat_learning_rate_max:
                parameters['learning_rate'] = hp.choice('learning_rate', [self.flags_parameters.cat_learning_rate_min])
            else:
                parameters['learning_rate'] = hp.loguniform('learning_rate', np.log(self.flags_parameters.cat_learning_rate_min),
                                                            np.log(self.flags_parameters.cat_learning_rate_max))
            if self.flags_parameters.cat_subsample_min == self.flags_parameters.cat_subsample_max:
                parameters['subsample'] = hp.choice('subsample', [self.flags_parameters.cat_subsample_min])
            else:
                parameters['subsample'] = hp.uniform('subsample', self.flags_parameters.cat_subsample_min,
                                                     self.flags_parameters.cat_subsample_max)
        else:
            if self.flags_parameters.cat_iterations_min == self.flags_parameters.cat_iterations_max:
                parameters['iterations'] = hp.choice('iterations', [self.flags_parameters.cat_iterations_min])
            else:
                parameters['iterations'] = hp.randint('iterations', self.flags_parameters.cat_iterations_min,
                                                      self.flags_parameters.cat_iterations_max)
            if self.flags_parameters.cat_depth_min == self.flags_parameters.cat_depth_max:
                parameters['depth'] = hp.choice('depth', [self.flags_parameters.cat_depth_min])
            else:
                parameters['depth'] = hp.randint('depth', self.flags_parameters.cat_depth_min,
                                                 self.flags_parameters.cat_depth_max)
            if self.flags_parameters.cat_learning_rate_min == self.flags_parameters.cat_learning_rate_max:
                parameters['learning_rate'] = hp.choice('learning_rate', [self.flags_parameters.cat_learning_rate_min])
            else:
                parameters['learning_rate'] = hp.loguniform('learning_rate',
                                                            np.log(self.flags_parameters.cat_learning_rate_min),
                                                            np.log(self.flags_parameters.cat_learning_rate_max))
            if self.flags_parameters.cat_subsample_min == self.flags_parameters.cat_subsample_max:
                parameters['subsample'] = hp.choice('subsample', [self.flags_parameters.cat_subsample_min])
            else:
                parameters['subsample'] = hp.uniform('subsample', self.flags_parameters.cat_subsample_min,
                                                     self.flags_parameters.cat_subsample_max)

            parameters['l2_leaf_reg'] = hp.randint('l2_leaf_reg', 1, 12)  # default None ?

        return parameters

    def initialize_params(self, y, params):
        self.shape_y = y.shape[1]
        if self.shape_y > 1:
            params = {'estimator__'+k: v for k, v in params.items()}
        self.p = params

        if self.class_weight == 'balanced' and self.shape_y == 1:
            try:
                self.class_weight = compute_class_weight(class_weight='balanced',
                                                         classes=np.unique(y.values.reshape(-1)),
                                                         y=y.values.reshape(-1))
            except:
                self.class_weight = compute_class_weight(class_weight='balanced',
                                                         classes=np.unique(y.reshape(-1)),
                                                         y=y.reshape(-1))
            # self.class_weight = dict(zip(np.unique(self.y_train[target]), weights))
        else:
            self.class_weight = None

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
            clf = cat.CatBoostRegressor(
                random_state=self.seed,
                verbose=False,
                **self.p
            )
            if self.shape_y == 1:
                return clf
            else:
                return MultiOutputRegressor(clf)
        else:

            clf = cat.CatBoostClassifier(
                random_state=self.seed,
                verbose=False,
                class_weights=self.class_weight,
                bootstrap_type='Bernoulli',
                **self.p
            )
            if self.shape_y == 1:
                return clf
            else:
                return MultiOutputClassifier(clf)
