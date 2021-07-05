import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.python.keras.layers import Layer
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
import os
import json
import numpy as np

from ...models.classifier_nlp.trainer import Model
from hyperopt import hp


class Attention(Model):
    name_classifier = 'Attention'
    dimension_embedding = "word_embedding"
    is_NN = True

    def __init__(self, flags_parameters, embedding, name_model_full, column_text, class_weight=None):
        Model.__init__(self, flags_parameters, embedding, name_model_full, column_text, class_weight)
        self.batch_size = self.flags_parameters.batch_size
        self.patience = self.flags_parameters.patience
        self.epochs = self.flags_parameters.epochs
        self.min_lr = self.flags_parameters.min_lr

    def hyper_params(self, size_params='small'):
        # Default : parameters = {'dropout_rate': hp.uniform('dropout_rate', 0, 0.5)}
        parameters = dict()
        if size_params == 'small':
            if self.flags_parameters.att_dropout_rate_min == self.flags_parameters.att_dropout_rate_max:
                parameters['dropout_rate'] = hp.choice('dropout_rate', [self.flags_parameters.att_dropout_rate_min])
            else:
                parameters['dropout_rate'] = hp.uniform('dropout_rate', self.flags_parameters.att_dropout_rate_min,
                                                        self.flags_parameters.att_dropout_rate_max)
        else:
            if self.flags_parameters.lstm_dropout_rate_min == self.flags_parameters.att_dropout_rate_max:
                parameters['dropout_rate'] = hp.choice('dropout_rate', [self.flags_parameters.att_dropout_rate_min])
            else:
                parameters['dropout_rate'] = hp.uniform('dropout_rate', self.flags_parameters.att_dropout_rate_min,
                                                        self.flags_parameters.att_dropout_rate_max)
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

        x = Dropout(self.p['dropout_rate'])(x)
        x = Attention_layer(self.embedding.maxlen)(x)

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


#################
# Build Attention Layer tensorflow :
#################


class Attention_layer(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim