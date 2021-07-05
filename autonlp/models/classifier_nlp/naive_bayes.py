from ...models.classifier_nlp.trainer import Model
from sklearn.naive_bayes import MultinomialNB
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from hyperopt import hp
from sklearn.pipeline import Pipeline
import os
import json


class Naive_Bayes(Model):
    name_classifier = 'Naive_Bayes'
    dimension_embedding = "doc_embedding"
    is_NN = False

    def __init__(self, flags_parameters, embedding, name_model_full, column_text, class_weight=None):
        Model.__init__(self, flags_parameters, embedding, name_model_full, column_text, class_weight)

    def hyper_params(self, size_params='small'):
        parameters = dict()
        if size_params == 'small':
            # parameters['clf__alpha'] = uniform(self.flags_parameters.nb_alpha_min, self.flags_parameters.nb_alpha_max)
            if self.flags_parameters.nb_alpha_min == self.flags_parameters.nb_alpha_max:
                parameters['clf__alpha'] = hp.choice('clf__alpha', [self.flags_parameters.nb_alpha_min])
            else:
                parameters['clf__alpha'] = hp.uniform('clf__alpha', self.flags_parameters.nb_alpha_min,
                                                      self.flags_parameters.nb_alpha_max)
        else:
            # parameters['clf__alpha'] = uniform(self.flags_parameters.nb_alpha_min, self.flags_parameters.nb_alpha_max)
            if self.flags_parameters.nb_alpha_min == self.flags_parameters.nb_alpha_max:
                parameters['clf__alpha'] = hp.choice('clf__alpha', [self.flags_parameters.nb_alpha_min])
            else:
                parameters['clf__alpha'] = hp.uniform('clf__alpha', self.flags_parameters.nb_alpha_min,
                                                      self.flags_parameters.nb_alpha_max)

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
        clf = MultinomialNB(
            **hyper_params_clf
        )

        if self.embedding.name_model in ['tf', 'tf-idf']:
            vect = self.embedding.model()
            pipeline = Pipeline(steps=[('vect', vect), ('clf', clf)])
            pipeline.set_params(**self.p)

        else:
            pipeline = Pipeline(steps=[('clf', clf)])
            pipeline.set_params(**self.p)
        return pipeline
