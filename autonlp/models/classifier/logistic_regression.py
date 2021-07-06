from ...models.classifier.trainer import Model
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from hyperopt import hp
import numpy as np
import os
import json


class ML_Logistic_Regression(Model):
    name_classifier = 'Logistic_Regression'
    is_NN = False

    def __init__(self, flags_parameters, name_model_full, class_weight=None):
        Model.__init__(self, flags_parameters, name_model_full, class_weight)

    def hyper_params(self, size_params='small'):
        parameters = dict()
        if size_params == 'small':
            # parameters['clf__C'] = loguniform(self.flags_parameters.logr_C_min, self.flags_parameters.logr_C_max)
            # parameters['clf__penalty'] = self.flags_parameters.logr_penalty
            if self.flags_parameters.logr_C_min == self.flags_parameters.logr_C_max:
                parameters['C'] = hp.choice('C', [self.flags_parameters.logr_C_min])
            else:
                parameters['C'] = hp.loguniform('C', np.log(self.flags_parameters.logr_C_min),
                                                     np.log(self.flags_parameters.logr_C_max))
            parameters['penalty'] = hp.choice('penalty', self.flags_parameters.logr_penalty)
        else:
            # parameters['clf__C'] = loguniform(self.flags_parameters.logr_C_min, self.flags_parameters.logr_C_max)
            # parameters['clf__penalty'] = self.flags_parameters.logr_penalty  # ['l2', 'l1', 'elasticnet', 'None']
            # parameters['clf__max__iter'] = randint(50, 150)
            if self.flags_parameters.logr_C_min == self.flags_parameters.logr_C_max:
                parameters['C'] = hp.choice('C', [self.flags_parameters.logr_C_min])
            else:
                parameters['C'] = hp.loguniform('C', np.log(self.flags_parameters.logr_C_min),
                                                     np.log(self.flags_parameters.logr_C_max))
            parameters['penalty'] = hp.choice('penalty', self.flags_parameters.logr_penalty)
            parameters['max_iter'] = hp.randint('max_iter', 50, 150)

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
        clf = LogisticRegression(
            random_state=self.seed,
            class_weight=self.class_weight,
            solver="saga",
            **hyper_params_clf
        )

        clf.set_params(**self.p)

        if self.shape_y == 1:
            return clf
        else:
            return MultiOutputClassifier(clf)
