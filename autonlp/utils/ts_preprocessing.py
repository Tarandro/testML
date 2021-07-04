import pandas as pd


def build_lag_features_transform(data_test, y_test, step_lags, position_id):
    target = list(y_test.columns)
    if position_id is None:
        for col in target:
            for i in step_lags:
                data_test[col + '_lag_ ' + str(i)] = y_test[col].transform(lambda x: x.shift(i))
    else:
        if isinstance(position_id, str):
            dt = pd.concat([data_test[[position_id]].reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
        else:
            dt = pd.concat([position_id[:len(y_test)].reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

        for i in step_lags:
            new_name_column = [col + '_lag_ ' + str(i) for col in target]
            data_test[new_name_column] = dt.groupby(dt.columns[0], as_index=False)[target].shift(i).fillna(0)
    return data_test


def build_rolling_features_transform(data_test, y_test, step_rolling, win_type, position_id):
    target = list(y_test.columns)

    if position_id is None:
        for col in target:
            for i in step_rolling:
                data_test[col + '_rolling_mean_ ' + str(i)] = y_test[col].transform \
                    (lambda x: x.rolling(i, win_type=win_type).mean())
                data_test[col + '_rolling_std_ ' + str(i)] = y_test[col].transform \
                    (lambda x: x.rolling(i, win_type=win_type).std())
    else:
        if isinstance(position_id, str):
            dt = pd.concat([data_test[[position_id]].reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
        else:
            dt = pd.concat([position_id[:len(y_test)].reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
        for i in step_rolling:
            new_name_column = [col + '_rolling_mean_ ' + str(i) for col in target]
            data_test[new_name_column] = dt.groupby(dt.columns[0], as_index=False)[target].transform \
                (lambda x: x.rolling(window=i, win_type=win_type).mean()).fillna(0)
            new_name_column = [col + '_rolling_std_ ' + str(i) for col in target]
            data_test[new_name_column] = dt.groupby(dt.columns[0], as_index=False)[target].transform \
                (lambda x: x.rolling(window=i, win_type=win_type).std()).fillna(0)
    return data_test
