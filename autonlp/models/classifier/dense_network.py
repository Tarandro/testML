from ...models.classifier.trainer import Model
from hyperopt import hp
import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras.layers import Dense,concatenate, Flatten, Embedding
import logging
from ...utils.logging import get_logger, verbosity_to_loglevel

logger = get_logger(__name__)


class ML_DenseNetwork(Model):
    name_classifier = 'Dense_Network'
    is_NN = True

    def __init__(self, flags_parameters, name_model_full, class_weight=None, len_unique_value={}):
        Model.__init__(self, flags_parameters, name_model_full, class_weight, len_unique_value)
        self.batch_size = self.flags_parameters.batch_size
        self.patience = self.flags_parameters.patience
        self.epochs = self.flags_parameters.epochs
        self.min_lr = self.flags_parameters.min_lr

    def hyper_params(self, size_params='small'):
        parameters = dict()
        if size_params == 'small':
            if self.flags_parameters.dnn_hidden_unit_1_min == self.flags_parameters.dnn_hidden_unit_1_max:
                parameters['hidden_unit_1'] = hp.choice('hidden_unit_1', [self.flags_parameters.dnn_hidden_unit_1_min])
            else:
                parameters['hidden_unit_1'] = hp.randint('hidden_unit_1', self.flags_parameters.dnn_hidden_unit_1_min,
                                                      self.flags_parameters.dnn_hidden_unit_1_max)
            if self.flags_parameters.dnn_hidden_unit_2_min == self.flags_parameters.dnn_hidden_unit_2_max:
                parameters['hidden_unit_2'] = hp.choice('hidden_unit_2', [self.flags_parameters.dnn_hidden_unit_2_min])
            else:
                parameters['hidden_unit_2'] = hp.choice('hd_2', [0, hp.randint('hidden_unit_2', self.flags_parameters.dnn_hidden_unit_2_min,
                                                 self.flags_parameters.dnn_hidden_unit_2_max)])
            if self.flags_parameters.dnn_hidden_unit_3_min == self.flags_parameters.dnn_hidden_unit_3_max:
                parameters['hidden_unit_3'] = hp.choice('hidden_unit_3', [self.flags_parameters.dnn_hidden_unit_3_min])
            else:
                parameters['hidden_unit_3'] = hp.choice('hd_3', [0, hp.randint('hidden_unit_3', self.flags_parameters.dnn_hidden_unit_3_min,
                                                 self.flags_parameters.dnn_hidden_unit_3_max)])

            parameters['learning_rate'] = hp.choice('learning_rate', self.flags_parameters.dnn_learning_rate)

            if self.flags_parameters.dnn_dropout_rate_min == self.flags_parameters.dnn_dropout_rate_max:
                parameters['dropout_rate'] = hp.choice('dropout_rate', [self.flags_parameters.dnn_dropout_rate_min])
            else:
                parameters['dropout_rate'] = hp.uniform('dropout_rate', self.flags_parameters.dnn_dropout_rate_min,
                                                     self.flags_parameters.dnn_dropout_rate_max)
        else:
            if self.flags_parameters.dnn_hidden_unit_1_min == self.flags_parameters.dnn_hidden_unit_1_max:
                parameters['hidden_unit_1'] = hp.choice('hidden_unit_1', [self.flags_parameters.dnn_hidden_unit_1_min])
            else:
                parameters['hidden_unit_1'] = hp.randint('hidden_unit_1', self.flags_parameters.dnn_hidden_unit_1_min,
                                                         self.flags_parameters.dnn_hidden_unit_1_max)
            if self.flags_parameters.dnn_hidden_unit_2_min == self.flags_parameters.dnn_hidden_unit_2_max:
                parameters['hidden_unit_2'] = hp.choice('hidden_unit_2', [self.flags_parameters.dnn_hidden_unit_2_min])
            else:
                parameters['hidden_unit_2'] = hp.choice('hd_2', [0, hp.randint('hidden_unit_2',
                                                                               self.flags_parameters.dnn_hidden_unit_2_min,
                                                                               self.flags_parameters.dnn_hidden_unit_2_max)])
            if self.flags_parameters.dnn_hidden_unit_3_min == self.flags_parameters.dnn_hidden_unit_3_max:
                parameters['hidden_unit_3'] = hp.choice('hidden_unit_3', [self.flags_parameters.dnn_hidden_unit_3_min])
            else:
                parameters['hidden_unit_3'] = hp.choice('hd_3', [0, hp.randint('hidden_unit_3',
                                                                               self.flags_parameters.dnn_hidden_unit_3_min,
                                                                               self.flags_parameters.dnn_hidden_unit_3_max)])
            if self.flags_parameters.dnn_learning_rate_min == self.flags_parameters.dnn_learning_rate_max:
                parameters['learning_rate'] = hp.choice('learning_rate', [self.flags_parameters.dnn_learning_rate_min])
            else:
                parameters['learning_rate'] = hp.choice('learning_rate', [self.flags_parameters.dnn_learning_rate_min,
                                                                          self.flags_parameters.dnn_learning_rate_max])
            if self.flags_parameters.dnn_dropout_rate_min == self.flags_parameters.dnn_dropout_rate_max:
                parameters['dropout_rate'] = hp.choice('dropout_rate', [self.flags_parameters.dnn_dropout_rate_min])
            else:
                parameters['dropout_rate'] = hp.uniform('dropout_rate', self.flags_parameters.dnn_dropout_rate_min,
                                                        self.flags_parameters.dnn_dropout_rate_max)

            parameters['regularizer'] = hp.uniform('regularizer', 0.000001, 0.00001)

        return parameters

    def preprocessing_fit_transform(self, x):
        self.ordered_column = list(x.columns)
        self.name_numeric_features = [name for name in x.columns if name not in self.ordinal_features]
        self.shape_embedding = {}
        if len(self.name_numeric_features) == 0:
            x_preprocessed = {}
        elif len(self.name_numeric_features) == 1:
            x_preprocessed = {"inp": x[[self.name_numeric_features]].to_numpy()}
            self.shape_embedding["inp"] = x_preprocessed["inp"].shape
        else:
            x_preprocessed = {"inp": x[self.name_numeric_features].to_numpy()}
            self.shape_embedding["inp"] = x_preprocessed["inp"].shape

        for col in self.ordinal_features:
            self.shape_embedding[col] = (self.len_unique_value[col], 1)
            x_preprocessed[col] = x[[col]].to_numpy()

        logger.info("Input shapes :")
        logger.info(self.shape_embedding)
        return x_preprocessed

    def preprocessing_transform(self, x):

        if len(self.name_numeric_features) == 0:
            x_preprocessed = {}
        elif len(self.name_numeric_features) == 1:
            x_preprocessed = {"inp": x[[self.name_numeric_features]].to_numpy()}
        else:
            x_preprocessed = {"inp": x[self.name_numeric_features].to_numpy()}

        for col in self.ordinal_features:
            x_preprocessed[col] = x[[col]].to_numpy()

        return x_preprocessed

    def initialize_params(self, y, params):
        self.shape_y = y.shape[1]
        if self.shape_y == 1:
            if 'regression' in self.objective:
                self.nb_classes = 1
            else:
                self.nb_classes = len(np.unique(y))
        else:
            self.nb_classes = self.shape_y

        hu = [params['hidden_unit_1']] + [params['hidden_unit_' + str(i)] for i in range(2, 4) if
                                          params['hidden_unit_' + str(i)] != 0]

        try:
            if self.size_params == 'small':
                self.p = {'hidden_units': hu,
                          'learning_rate': params['learning_rate'],
                          'dropout_rate': params['dropout_rate']
                          }
            else:
                self.p = {'hidden_units': hu,
                          'learning_rate': params['learning_rate'],
                          'dropout_rate': params['dropout_rate'],
                          'regularizer': params['regularizer']
                          }
        except:
            self.p = {'hidden_units': hu,
                      'learning_rate': params['learning_rate'],
                      'dropout_rate': params['dropout_rate']
                      }
            self.size_params = 'small'
        #self.p = params

    def save_params(self, outdir_model):
        params_all = dict()

        p_model = self.p.copy()
        params_all['p_model'] = p_model
        params_all['name_classifier'] = self.name_classifier
        params_all['shape_y'] = self.shape_y
        params_all['nb_classes'] = self.nb_classes
        params_all['name_numeric_features'] = self.name_numeric_features
        params_all['shape_embedding'] = self.shape_embedding

        self.params_all = {self.name_model_full: params_all}

        if self.apply_logs:
            with open(os.path.join(outdir_model, "parameters.json"), "w") as outfile:
                json.dump(self.params_all, outfile)

    def load_params(self, params_all, outdir):
        p_model = params_all['p_model']
        self.p = p_model
        self.shape_y = params_all['shape_y']
        self.nb_classes = params_all['nb_classes']
        self.name_numeric_features = params_all['name_numeric_features']
        self.shape_embedding = params_all['shape_embedding']

    def model(self, hyper_params_clf={}):
        # Dense input
        if "inp" in self.shape_embedding.keys():
            dense = tf.keras.layers.Input(shape=(self.shape_embedding["inp"][1],), name="inp")
            INPUT_LIST = [dense]
            EMBEDDING_LIST = [dense]
            inp = {"inp": dense}
        else:
            INPUT_LIST = []
            EMBEDDING_LIST = []
            inp = {}

        for col in self.shape_embedding.keys():
            if col != "inp":
                a = tf.keras.layers.Input(shape=(1,), name=col)
                INPUT_LIST.append(a)
                inp[col] = a

        # Embedding input
        for i, col in enumerate(self.shape_embedding.keys()):
            if col != "inp":
                a = Flatten()(
                    Embedding(self.shape_embedding[col][0], self.shape_embedding[col][1], name="embedding_" + col)(
                        INPUT_LIST[i]))
                EMBEDDING_LIST.append(a)

        if len(EMBEDDING_LIST) >= 2:
            x = concatenate(EMBEDDING_LIST, axis=-1)
        else:
            x = EMBEDDING_LIST[0]

        # x = tf.keras.layers.BatchNormalization()(inp)

        for units in self.p['hidden_units']:
            if 'regularizer' not in self.p.keys():
                x = tf.keras.layers.Dense(int(units), activation='relu')(x)
            else:
                x = tf.keras.layers.Dense(int(units), activation='relu',
                                          kernel_regularizer=tf.keras.regularizers.l2(self.p['regularizer']))(x)
            x = tf.keras.layers.Dropout(self.p['dropout_rate'])(x)
            x = tf.keras.layers.BatchNormalization()(x)

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