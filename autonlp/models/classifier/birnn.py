from ...models.classifier.trainer import Model
from tensorflow.keras.layers import SimpleRNN, Dropout
from tensorflow.keras.layers import Bidirectional
import numpy as np
import tensorflow as tf
from hyperopt import hp
from tensorflow.keras.layers import Dense
import os
import json


class Birnn(Model):
    name_classifier = 'Birnn'
    dimension_embedding = "word_embedding"
    is_NN = True

    def __init__(self, flags_parameters, embedding, name_model_full, column_text, class_weight=None):
        Model.__init__(self, flags_parameters, embedding, name_model_full, column_text, class_weight)
        self.batch_size = self.flags_parameters.batch_size
        self.patience = self.flags_parameters.patience
        self.epochs = self.flags_parameters.epochs
        self.min_lr = self.flags_parameters.min_lr

    def hyper_params(self, size_params='small'):
        # Default : parameters = {'hidden_unit': hp.randint('hidden_unit_1', 120, 130),
        #                         'dropout_rate': hp.uniform('dropout_rate', 0, 0.5)}
        parameters = dict()
        if size_params == 'small':
            if self.flags_parameters.rnn_hidden_unit_min == self.flags_parameters.rnn_hidden_unit_max:
                parameters['hidden_unit'] = hp.choice('hidden_unit_1', [self.flags_parameters.rnn_hidden_unit_min])
            else:
                parameters['hidden_unit'] = hp.randint('hidden_unit_1', self.flags_parameters.rnn_hidden_unit_min,
                                                       self.flags_parameters.rnn_hidden_unit_max)
            if self.flags_parameters.rnn_dropout_rate_min == self.flags_parameters.rnn_dropout_rate_max:
                parameters['dropout_rate'] = hp.choice('dropout_rate', [self.flags_parameters.rnn_dropout_rate_min])
            else:
                parameters['dropout_rate'] = hp.uniform('dropout_rate', self.flags_parameters.rnn_dropout_rate_min,
                                                        self.flags_parameters.rnn_dropout_rate_max)
        else:
            if self.flags_parameters.rnn_hidden_unit_min == self.flags_parameters.rnn_hidden_unit_max:
                parameters['hidden_unit'] = hp.choice('hidden_unit_1', [self.flags_parameters.rnn_hidden_unit_min])
            else:
                parameters['hidden_unit'] = hp.randint('hidden_unit_1', self.flags_parameters.rnn_hidden_unit_min,
                                                       self.flags_parameters.rnn_hidden_unit_max)
            if self.flags_parameters.rnn_dropout_rate_min == self.flags_parameters.rnn_dropout_rate_max:
                parameters['dropout_rate'] = hp.choice('dropout_rate', [self.flags_parameters.rnn_dropout_rate_min])
            else:
                parameters['dropout_rate'] = hp.uniform('dropout_rate', self.flags_parameters.rnn_dropout_rate_min,
                                                        self.flags_parameters.rnn_dropout_rate_max)
        parameters_embedding = self.embedding.hyper_params()
        parameters.update(parameters_embedding)
        return parameters

    def initialize_params(self, y, params):
        self.shape_y = y.shape[1]
        if self.dimension_embedding == 'word_embedding':
            if self.shape_y == 1:
                if 'regression' in self.objective:
                    self.nb_classes = 1
                else:
                    self.nb_classes = len(np.unique(y))
            else:
                self.nb_classes = self.shape_y
        self.p = params

    def save_params(self, outdir_model):
        params_all = dict()
        params_all['p_model'] = self.p
        params_all['language_text'] = self.flags_parameters.language_text
        params_all['name_classifier'] = self.name_classifier

        params_all['nb_classes'] = self.nb_classes
        params_all['shape_y'] = self.shape_y

        params_embedding = self.embedding.save_params(outdir_model)
        params_all.update(params_embedding)

        self.params_all = {self.name_model_full: params_all}

        if self.apply_logs:
            with open(os.path.join(outdir_model, "parameters.json"), "w") as outfile:
                json.dump(self.params_all, outfile)

    def load_params(self, params_all, outdir):
        self.p = params_all['p_model']

        self.nb_classes = params_all['nb_classes']
        self.shape_y = params_all['shape_y']

        self.embedding.load_params(params_all, outdir)

    def model(self):

        x, inp = self.embedding.model()

        x = Bidirectional(SimpleRNN(int(self.p['hidden_unit']), return_sequences=False))(x)
        # x = Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
        x = Dropout(self.p['dropout_rate'])(x)

        if 'binary' in self.objective:
            out = Dense(1, 'sigmoid')(x)
        elif 'regression' in self.objective:
            out = Dense(self.nb_classes, 'linear')(x)
        else:
            if self.shape_y == 1:
                out = Dense(self.nb_classes, activation="softmax")(x)
            else:
                out = Dense(self.nb_classes, activation="sigmoid")(x)

        model = tf.keras.models.Model(inputs=inp, outputs=out)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.p['learning_rate'])
        if 'binary' in self.objective:
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        elif 'regression' in self.objective:
            model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
        else:
            if self.shape_y == 1:
                model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            else:
                model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=optimizer,
                              metrics=['binary_crossentropy'])
        return model