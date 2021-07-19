from sklearn.preprocessing import MinMaxScaler

from ...models.classifier.trainer import Model
from hyperopt import hp
import numpy as np
import pandas as pd
import os
import json
import tensorflow as tf
from tensorflow.keras.layers import Dense,concatenate, Flatten, Embedding
import logging
from ...utils.logging import get_logger, verbosity_to_loglevel

logger = get_logger(__name__)


class ML_LSTM(Model):
    name_classifier = 'LSTM'
    is_NN = True

    def __init__(self, flags_parameters, name_model_full, class_weight=None, len_unique_value={},
                 time_series_features=None, scaler_info=None, position_id=None):
        Model.__init__(self, flags_parameters, name_model_full, class_weight, len_unique_value, time_series_features,
                       scaler_info, position_id)
        self.batch_size = self.flags_parameters.batch_size
        self.patience = self.flags_parameters.patience
        self.epochs = self.flags_parameters.epochs
        self.min_lr = self.flags_parameters.min_lr

        self.position_date = self.flags_parameters.position_date
        self.timesteps = self.flags_parameters.timesteps

    def hyper_params(self, size_params='small'):
        parameters = dict()
        if size_params == 'small':
            if self.flags_parameters.lstm_hidden_unit_1_min == self.flags_parameters.lstm_hidden_unit_1_max:
                parameters['hidden_unit_1'] = hp.choice('hidden_unit_1', [self.flags_parameters.lstm_hidden_unit_1_min])
            else:
                parameters['hidden_unit_1'] = hp.randint('hidden_unit_1', self.flags_parameters.lstm_hidden_unit_1_min,
                                                      self.flags_parameters.lstm_hidden_unit_1_max)
            if self.flags_parameters.lstm_hidden_unit_2_min == self.flags_parameters.lstm_hidden_unit_2_max:
                parameters['hidden_unit_2'] = hp.choice('hidden_unit_2', [self.flags_parameters.lstm_hidden_unit_2_min])
            else:
                parameters['hidden_unit_2'] = hp.choice('hd_2', [0, hp.randint('hidden_unit_2', self.flags_parameters.lstm_hidden_unit_2_min,
                                                 self.flags_parameters.lstm_hidden_unit_2_max)])
            if self.flags_parameters.lstm_hidden_unit_3_min == self.flags_parameters.lstm_hidden_unit_3_max:
                parameters['hidden_unit_3'] = hp.choice('hidden_unit_3', [self.flags_parameters.lstm_hidden_unit_3_min])
            else:
                parameters['hidden_unit_3'] = hp.choice('hd_3', [0, hp.randint('hidden_unit_3', self.flags_parameters.lstm_hidden_unit_3_min,
                                                 self.flags_parameters.lstm_hidden_unit_3_max)])

            parameters['learning_rate'] = hp.choice('learning_rate', self.flags_parameters.lstm_learning_rate)

            if self.flags_parameters.lstm_dropout_rate_min == self.flags_parameters.lstm_dropout_rate_max:
                parameters['dropout_rate'] = hp.choice('dropout_rate', [self.flags_parameters.lstm_dropout_rate_min])
            else:
                parameters['dropout_rate'] = hp.uniform('dropout_rate', self.flags_parameters.lstm_dropout_rate_min,
                                                     self.flags_parameters.lstm_dropout_rate_max)
        else:
            if self.flags_parameters.lstm_hidden_unit_1_min == self.flags_parameters.lstm_hidden_unit_1_max:
                parameters['hidden_unit_1'] = hp.choice('hidden_unit_1', [self.flags_parameters.lstm_hidden_unit_1_min])
            else:
                parameters['hidden_unit_1'] = hp.randint('hidden_unit_1', self.flags_parameters.lstm_hidden_unit_1_min,
                                                         self.flags_parameters.lstm_hidden_unit_1_max)
            if self.flags_parameters.lstm_hidden_unit_2_min == self.flags_parameters.lstm_hidden_unit_2_max:
                parameters['hidden_unit_2'] = hp.choice('hidden_unit_2', [self.flags_parameters.lstm_hidden_unit_2_min])
            else:
                parameters['hidden_unit_2'] = hp.choice('hd_2', [0, hp.randint('hidden_unit_2',
                                                                               self.flags_parameters.lstm_hidden_unit_2_min,
                                                                               self.flags_parameters.lstm_hidden_unit_2_max)])
            if self.flags_parameters.lstm_hidden_unit_3_min == self.flags_parameters.lstm_hidden_unit_3_max:
                parameters['hidden_unit_3'] = hp.choice('hidden_unit_3', [self.flags_parameters.lstm_hidden_unit_3_min])
            else:
                parameters['hidden_unit_3'] = hp.choice('hd_3', [0, hp.randint('hidden_unit_3',
                                                                               self.flags_parameters.lstm_hidden_unit_3_min,
                                                                               self.flags_parameters.lstm_hidden_unit_3_max)])

            parameters['learning_rate'] = hp.choice('learning_rate', self.flags_parameters.lstm_learning_rate)

            if self.flags_parameters.lstm_dropout_rate_min == self.flags_parameters.lstm_dropout_rate_max:
                parameters['dropout_rate'] = hp.choice('dropout_rate', [self.flags_parameters.lstm_dropout_rate_min])
            else:
                parameters['dropout_rate'] = hp.uniform('dropout_rate', self.flags_parameters.lstm_dropout_rate_min,
                                                        self.flags_parameters.lstm_dropout_rate_max)

            parameters['regularizer'] = hp.uniform('regularizer', 0.000001, 0.00001)

        return parameters

    def preprocessing_fit_transform(self, x, y, size_train_prc=0.9):
        self.ordered_column = list(x.columns)
        size_train = int(y.shape[0] * size_train_prc)
        if x is not None:
            dt = pd.concat([x, y], axis=1)
        else:
            dt = y.copy()

        value_series = list(y.columns)
        self.index_pivot_fit = []
        if self.position_id is None or self.position_date is None:
            dt_date_id = dt[value_series]
            self.size_y_train_preprocessed = size_train - self.timesteps - 1
        elif self.position_date not in dt.columns:
            dt_date_id = dt[value_series]
            self.size_y_train_preprocessed = size_train - self.timesteps - 1
        else:
            try:
                dt = pd.concat([self.position_id.loc[list(dt.index), :], dt], axis=1)
                dt_date_id = dt.pivot(index=self.position_date, columns=self.position_id.columns[0],
                                      values=value_series[0])
                for col in value_series[1:]:
                    dt_date_id = pd.concat([dt_date_id,
                                            dt.pivot(index=self.position_date, columns=self.position_id.columns[0],
                                                     values=col)], axis=1)

                list_index = list(dt_date_id.index)
                list_column = list(dt_date_id.columns)
                for i, row in dt[size_train:].iterrows():
                    index_row = list_index.index(row[self.position_date])
                    index_column = list_column.index(row[self.position_id.columns[0]])
                    self.index_pivot_fit.append(index_row * len(list_column) + index_column - size_train)
                    if i == size_train:
                        self.size_y_train_preprocessed = index_row - self.timesteps - 1
                        delate = index_column
            except:
                dt_date_id = dt[value_series]
                self.size_y_train_preprocessed = size_train - self.timesteps - 1

        for col in dt_date_id.columns:
            if dt_date_id[col].isnull().sum() > 0:
                dt_date_id[col] = dt_date_id[col].fillna(method='ffill').fillna(method='bfill')

        #########
        # Scaling
        #########

        self.scaler_ts = MinMaxScaler(feature_range=(0, 1))
        self.features_id = list(dt_date_id.columns)
        dt_date_id[self.features_id] = self.scaler_ts.fit_transform(dt_date_id)

        #########

        self.LSTM_date_features = [col for col in self.LSTM_date_features if col in dt.columns]
        if len(self.LSTM_date_features) > 0:
            add_features_to_date = dt[self.LSTM_date_features + [self.position_date]].drop_duplicates(
                self.position_date).drop([self.position_date], axis=1)
            self.add_features_to_date_input_last_timesteps = add_features_to_date.iloc[-(self.timesteps - 1):, :]

            x_array = dt_date_id.values
            x_preprocessed = {'inp': []}
            add_features_array = {}
            for col in self.LSTM_date_features:
                #if list(x.columns).index(col) != self.column_text:
                x_preprocessed[col] = []
                add_features_array[col] = add_features_to_date[[col]].values
                #else:
                #    token_info = self.preprocessing_text_fit_transform(x[col], size_params, method_embedding)
                #    add_features_array["tok"] = token_info["tok"]
                #    x_preprocessed["tok"] = []
            y_preprocessed = []
            self.shape_embedding = {}

            for i in range(self.timesteps, x_array.shape[0] - 1):
                x_preprocessed['inp'].append(x_array[i - self.timesteps:i])
                y_preprocessed.append(self.scaler_ts.inverse_transform(x_array[i].reshape(1, -1)).reshape(-1))

                for col in add_features_array.keys():
                    x_preprocessed[col].append(
                        add_features_array[col][(i - self.timesteps + 1):(i + 1), :])  # features add for the next date

                # if i > x_array.shape[0]-10:
                #    print('inp :')
                #    print(x_array[i-self.timesteps:i])
                #    print('y :')
                #    print(x_array[i])

            self.input_last_timesteps = {"inp": [x_array[-self.timesteps:]]}

            logger.info("Preprocess time_series, shapes :")
            for col in x_preprocessed.keys():
                x_preprocessed[col] = np.array(x_preprocessed[col])
            y_preprocessed = np.array(y_preprocessed)
            logger.info("X_train shape :", x_preprocessed["inp"].shape)
            logger.info("y_train shape :", y_preprocessed.shape)
            for col in add_features_array.keys():
                #if col not in ['tok']:
                self.shape_embedding[col] = (self.len_unique_value[col], 1)
                #else:
                #    self.shape_embedding[col] = (self.maxlen,)
            logger.info(self.shape_embedding)

            del add_features_to_date

        else:
            x_array = dt_date_id.values
            x_preprocessed = []
            y_preprocessed = []
            for i in range(self.timesteps, x_array.shape[0]):
                x_preprocessed.append(x_array[i - self.timesteps:i])
                y_preprocessed.append(x_array[i])
                if (i + 1) == x_array.shape[0]:
                    self.input_last_timesteps = np.array([x_array[i - self.timesteps:i]])

            logger.info("Preprocess time_series, shapes :")
            x_preprocessed = np.array(x_preprocessed)
            y_preprocessed = np.array(y_preprocessed)
            logger.info("X_train shape :", x_preprocessed.shape)
            logger.info("y_train shape :", y_preprocessed.shape)
        del dt, x_array, dt_date_id

        if isinstance(x_preprocessed, dict):
            self.shape_x_1 = x_preprocessed['inp'].shape[1]
            self.shape_x_2 = x_preprocessed['inp'].shape[2]
        else:
            self.shape_x_1 = x_preprocessed.shape[1]
            self.shape_x_2 = x_preprocessed.shape[2]

        return x_preprocessed, y_preprocessed

    def preprocessing_transform(self, x, y=None):

        if y is not None:
            if x is not None:
                dt = pd.concat([x, y], axis=1)
            else:
                dt = y.copy()

            value_series = list(y.columns)
            if self.position_id is None or self.position_date is None:
                dt_date_id = dt[value_series]
            elif self.position_date not in dt.columns:
                dt_date_id = dt[value_series]
            else:
                try:
                    dt = pd.concat([self.position_id.loc[list(dt.index), :], dt], axis=1)
                    dt_date_id = dt.pivot(index=self.position_date, columns=self.position_id.columns[0],
                                          values=value_series[0])
                    for col in value_series[1:]:
                        dt_date_id = pd.concat([dt_date_id,
                                                dt.pivot(index=self.position_date, columns=self.position_id.columns[0],
                                                         values=col)], axis=1)
                except:
                    dt_date_id = dt[value_series]

            for col in dt_date_id.columns:
                if dt_date_id[col].isnull().sum() > 0:
                    dt_date_id[col] = dt_date_id[col].fillna(method='ffill').fillna(method='bfill')

            y_preprocessed = dt_date_id
        else:
            y_preprocessed = None

        #################################################

        add_features_to_date = x[self.LSTM_date_features + [self.position_date]].drop_duplicates(self.position_date)
        list_index = list(add_features_to_date[self.position_date])
        add_features_to_date.drop([self.position_date], axis=1)
        add_features_to_date = pd.concat([self.add_features_to_date_input_last_timesteps, add_features_to_date], axis=0)

        self.index_pivot_pred = []
        if self.position_id is not None:
            list_column = list(self.features_id)
            dt = pd.concat([self.position_id.loc[list(x.index), :], x], axis=1)
            for i, row in dt.iterrows():
                index_row = list_index.index(row[self.position_date])
                index_column = list_column.index(row[self.position_id.columns[0]])
                self.index_pivot_pred.append(index_row * len(list_column) + index_column)

        add_features_array = {}
        for col in self.LSTM_date_features:
            #if list(x.columns).index(col) != self.column_text:
            add_features_array[col] = add_features_to_date[[col]].values
            self.input_last_timesteps[col] = []
            #else:
            #    token_info = self.preprocessing_text_transform(add_features_to_date[col])
            #    add_features_array["tok"] = token_info["tok"]
            #    self.input_last_timesteps["tok"] = []

        for i in range(self.timesteps, add_features_to_date.shape[0] + 1):
            for col in add_features_array.keys():
                self.input_last_timesteps[col].append(add_features_array[col][i - self.timesteps:i, :])

        for col in self.input_last_timesteps.keys():
            self.input_last_timesteps[col] = np.array(self.input_last_timesteps[col])

        if isinstance(self.input_last_timesteps, dict):
            self.shape_x_1 = self.input_last_timesteps['inp'].shape[1]
            self.shape_x_2 = self.input_last_timesteps['inp'].shape[2]
        else:
            self.shape_x_1 = self.input_last_timesteps.shape[1]
            self.shape_x_2 = self.input_last_timesteps.shape[2]

        return self.input_last_timesteps, y_preprocessed

    def initialize_params(self, y, params):
        #if isinstance(x, dict):
        #    self.shape_x_1 = x['inp'].shape[1]
        #    self.shape_x_2 = x['inp'].shape[2]
        #else:
        #    self.shape_x_1 = x.shape[1]
        #    self.shape_x_2 = x.shape[2]

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
        params_all['shape_embedding'] = self.shape_embedding

        df_to_dict = self.add_features_to_date_input_last_timesteps.to_dict()
        params_all['add_features_to_date_input_last_timesteps'] = df_to_dict

        self.params_all = {self.name_model_full: params_all}

        if self.apply_logs:
            with open(os.path.join(outdir_model, "parameters.json"), "w") as outfile:
                json.dump(self.params_all, outfile)

    def load_params(self, params_all, outdir):
        p_model = params_all['p_model']
        self.p = p_model
        self.shape_y = params_all['shape_y']
        self.nb_classes = params_all['nb_classes']
        self.shape_embedding = params_all['shape_embedding']

        self.add_features_to_date_input_last_timesteps = pd.DataFrame(params_all['add_features_to_date_input_last_timesteps'])

    def model(self, hyper_params_clf={}):
        dense = tf.keras.layers.Input(shape=(self.shape_x_1, self.shape_x_2,), name="inp")

        if len(self.LSTM_date_features) > 0:
            INPUT_LIST = [dense]
            EMBEDDING_LIST = [dense]
            inp = {"inp": dense}

            for col in self.shape_embedding.keys():
                if col not in ['inp', 'tok']:
                    a = tf.keras.layers.Input(shape=(self.shape_x_1, 1,), name=col)
                    INPUT_LIST.append(a)
                    inp[col] = a

            # Embedding input
            indice = 0
            for i, col in enumerate(self.shape_embedding.keys()):
                if col not in ['inp', 'tok']:
                    if col in self.ordinal_features:
                        a = tf.keras.layers.TimeDistributed(Embedding(self.shape_embedding[col][0], self.shape_embedding[col][1],
                                                      name="embedding_" + col))(
                            INPUT_LIST[1 + indice])  # shape (batch, timesteps,1, 1)
                        a = tf.squeeze(a, axis=-1)  # shape (batch, timesteps,1)
                        EMBEDDING_LIST.append(a)
                    else:
                        EMBEDDING_LIST.append(INPUT_LIST[1 + indice])
                    indice += 1

            #if "tok" in self.shape_embedding.keys():
            #    token = tf.keras.layers.Input(shape=(self.shape_x_1, self.shape_embedding["tok"][0],), name="tok")
            #    INPUT_LIST.append(token)
            #    inp["tok"] = token
            #
            #    # Embedding
            #    a = TimeDistributed(
            #        Embedding(len(self.word_index) + 1, self.embed_size, weights=[self.embedding_matrix],
            #                  trainable=True))(token)
            #    a = tf.math.reduce_mean(a, axis=-2)  # mean word vectors to get document vector
            #    EMBEDDING_LIST.append(a)

            if len(EMBEDDING_LIST) >= 2:
                x = concatenate(EMBEDDING_LIST,
                                axis=-1)  # shape (batch, timesteps,self.shape_x_2+len(self.LSTM_date_features))
            else:
                x = EMBEDDING_LIST[0]
        else:
            x = tf.keras.layers.BatchNormalization()(dense)

        for i, units in enumerate(self.p['hidden_units']):

            if (i + 1) == len(self.p['hidden_units']):
                x = tf.keras.layers.LSTM(int(units))(x)
            else:
                x = tf.keras.layers.LSTM(int(units), return_sequences=True)(x)
            x = tf.keras.layers.Dropout(self.p['dropout_rate'])(x)
            # x = tf.keras.layers.BatchNormalization()(x)

        if 'binary' in self.objective:
            out = Dense(1, 'sigmoid')(x)
        elif 'regression' in self.objective:
            out = Dense(self.nb_classes, 'linear')(x)
        else:
            if self.shape_y == 1:
                out = Dense(self.nb_classes, activation="softmax")(x)
            else:
                out = Dense(self.nb_classes, activation="sigmoid")(x)

        if len(self.LSTM_date_features) > 0:
            model = tf.keras.models.Model(inputs=inp, outputs=out)
        else:
            model = tf.keras.models.Model(inputs=dense, outputs=out)

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