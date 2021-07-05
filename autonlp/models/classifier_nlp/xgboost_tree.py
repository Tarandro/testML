from ...models.classifier_nlp.trainer import Model
from hyperopt import hp
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
import xgboost as xgb
from sklearn.pipeline import Pipeline
import os
import json


class XGBoost(Model):
    name_classifier = 'XGBoost'
    dimension_embedding = "doc_embedding"
    is_NN = False

    def __init__(self, flags_parameters, embedding, name_model_full, column_text, class_weight=None):
        Model.__init__(self, flags_parameters, embedding, name_model_full, column_text, class_weight)

    def hyper_params(self, size_params='small'):
        parameters = dict()
        if size_params == 'small':
            # self.parameters = dict(n_estimators = randint(20,200), #[75, 100, 125, 150],
            #                                    max_depth = randint(3,10),    #[7,8,10,20,30]
            #                                    learning_rate = uniform(0.04,0.3),
            #                                   subsample = uniform(0.5,0.5))
            if self.flags_parameters.xgb_n_estimators_min == self.flags_parameters.xgb_n_estimators_max:
                parameters['clf__n_estimators'] = hp.choice('clf__n_estimators', [self.flags_parameters.xgb_n_estimators_min])
            else:
                parameters['clf__n_estimators'] = hp.choice('clf__n_estimators', [i for i in range(
                    self.flags_parameters.xgb_n_estimators_min, self.flags_parameters.xgb_n_estimators_max+1)])
            if self.flags_parameters.xgb_max_depth_min == self.flags_parameters.xgb_max_depth_max:
                parameters['clf__max_depth'] = hp.choice('clf__max_depth', [self.flags_parameters.xgb_max_depth_min])
            else:
                parameters['clf__max_depth'] = hp.choice('clf__max_depth', [i for i in range(
                    self.flags_parameters.xgb_max_depth_min, self.flags_parameters.xgb_max_depth_max+1)])
            if self.flags_parameters.xgb_learning_rate_min == self.flags_parameters.xgb_learning_rate_max:
                parameters['clf__learning_rate'] = hp.choice('clf__learning_rate',
                                                             [self.flags_parameters.xgb_learning_rate_min])
            else:
                parameters['clf__learning_rate'] = hp.uniform('clf__learning_rate',
                                                              self.flags_parameters.xgb_learning_rate_min,
                                                              self.flags_parameters.xgb_learning_rate_max)
            if self.flags_parameters.xgb_subsample_min == self.flags_parameters.xgb_subsample_max:
                parameters['clf__subsample'] = hp.choice('clf__subsample', [self.flags_parameters.xgb_subsample_min])
            else:
                parameters['clf__subsample'] = hp.uniform('clf__subsample', self.flags_parameters.xgb_subsample_min,
                                                          self.flags_parameters.xgb_subsample_max)
        else:
            if self.flags_parameters.xgb_n_estimators_min == self.flags_parameters.xgb_n_estimators_max:
                parameters['clf__n_estimators'] = hp.choice('clf__n_estimators',
                                                            [self.flags_parameters.xgb_n_estimators_min])
            else:
                parameters['clf__n_estimators'] = hp.choice('clf__n_estimators', [i for i in range(
                    self.flags_parameters.xgb_n_estimators_min, self.flags_parameters.xgb_n_estimators_max + 1)])
            if self.flags_parameters.xgb_max_depth_min == self.flags_parameters.xgb_max_depth_max:
                parameters['clf__max_depth'] = hp.choice('clf__max_depth', [self.flags_parameters.xgb_max_depth_min])
            else:
                parameters['clf__max_depth'] = hp.choice('clf__max_depth', [i for i in range(
                    self.flags_parameters.xgb_max_depth_min, self.flags_parameters.xgb_max_depth_max + 1)])
            if self.flags_parameters.xgb_learning_rate_min == self.flags_parameters.xgb_learning_rate_max:
                parameters['clf__learning_rate'] = hp.choice('clf__learning_rate',
                                                             [self.flags_parameters.xgb_learning_rate_min])
            else:
                parameters['clf__learning_rate'] = hp.uniform('clf__learning_rate',
                                                              self.flags_parameters.xgb_learning_rate_min,
                                                              self.flags_parameters.xgb_learning_rate_max)
            if self.flags_parameters.xgb_subsample_min == self.flags_parameters.xgb_subsample_max:
                parameters['clf__subsample'] = hp.choice('clf__subsample', [self.flags_parameters.xgb_subsample_min])
            else:
                parameters['clf__subsample'] = hp.uniform('clf__subsample', self.flags_parameters.xgb_subsample_min,
                                                          self.flags_parameters.xgb_subsample_max)

        if self.embedding.name_model in ['tf', 'tf-idf']:
            parameters_embedding = self.embedding.hyper_params()
            parameters.update(parameters_embedding)

        return parameters

    def initialize_params(self, y, params):
        self.shape_y = y.shape[1]
        self.p = params

    def save_params(self, outdir_model):
        params_all = dict()

        p_model = self.p.copy()
        # list of stop_words is transformed in boolean
        if 'vect__text__tf__stop_words' in p_model.keys() and p_model['vect__text__tf__stop_words'] is not None:
            p_model['vect__text__tf__stop_words'] = True
        if 'vect__tf__stop_words' in p_model.keys() and p_model['vect__tf__stop_words'] is not None:
            p_model['vect__tf__stop_words'] = True
        params_all['p_model'] = p_model
        params_all['name_classifier'] = self.name_classifier
        params_all['language_text'] = self.flags_parameters.language_text

        params_embedding = self.embedding.save_params(outdir_model)
        params_all.update(params_embedding)

        self.params_all = {self.name_model_full: params_all}

        if self.apply_logs:
            with open(os.path.join(outdir_model, "parameters.json"), "w") as outfile:
                json.dump(self.params_all, outfile)

    def load_params(self, params_all, outdir):
        if params_all['language_text'] == 'fr':
            stopwords = list(fr_stop)
        else:
            stopwords = list(en_stop)
        p_model = params_all['p_model']
        # list of stop_words need to be boolean
        if 'vect__text__tf__stop_words' in p_model.keys() and p_model['vect__text__tf__stop_words']:
            p_model['vect__text__tf__stop_words'] = stopwords
        if 'vect__tf__stop_words' in p_model.keys() and p_model['vect__tf__stop_words']:
            p_model['vect__tf__stop_words'] = stopwords
        if 'vect__text__tfidf__stop_words' in p_model.keys() and p_model['vect__text__tfidf__stop_words']:
            p_model['vect__text__tfidf__stop_words'] = stopwords
        if 'vect__tfidf__stop_words' in p_model.keys() and p_model['vect__tfidf__stop_words']:
            p_model['vect__tfidf__stop_words'] = stopwords

        self.p = p_model

        self.embedding.load_params(params_all, outdir)

    def model(self, hyper_params_clf={}):
        if 'regression' in self.flags_parameters.objective:
            clf = xgb.XGBRegressor(
                random_state=self.seed,
                **hyper_params_clf
            )
        else:
            clf = xgb.XGBClassifier(
                random_state=self.seed,
                **hyper_params_clf  # ,
                # scale_pos_weight = count(negative examples)/count(Positive examples)
            )
        if self.embedding.name_model in ['tf', 'tf-idf']:
            vect = self.embedding.model()
            pipeline = Pipeline(steps=[('vect', vect), ('clf', clf)])
            pipeline.set_params(**self.p)

        else:
            pipeline = Pipeline(steps=[('clf', clf)])
            pipeline.set_params(**self.p)
        return pipeline
