import pandas as pd
from sklearn.cluster import KMeans


def interpolate_missing_data_categorical(data, method):
    """Interpolate missing data and return as pandas data frame."""

    if method == "constant":
        return data.fillna('not_available')
    elif method == 'ffill':
        return data.fillna(method='ffill').fillna(method='bfill')
    else:
        return data.fillna(data.mode())


def interpolate_missing_data_numeric(data, method):
    """Interpolate missing data and return as pandas data frame."""

    def is_number(string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    if method == "mean":
        return data.fillna(data.mean())
    elif method == 'ffill':
        return data.fillna(method='ffill').fillna(method='bfill')
    elif method == 'interpolate':
        return data.interpolate()
    elif is_number(str(method)):
        return data.fillna(float(method))
    else:
        return data.fillna(data.median())


def remove_outliers(data, real):
    """Remove outliers from data and return as a pandas data frame."""

    # get field mean and std for real-valued fields
    mean = data.describe().iloc[1, :]
    std = data.describe().iloc[2, :]

    # remove outliers
    for (real, mean, std) in zip(real, mean, std):
        data = data[data[real] < 3 * std + mean]

    return data


def one_hot_encode(data):
    """Perform a one-hot encoding and return only n-1 columns (avoid multicorrelation) """
    return pd.get_dummies(data).iloc[:, :-1]


def ordinal_encoding(data):
    dict_map_ordinal = dict(zip(data.astype('category').cat.codes, data))
    return data.astype('category').cat.codes, dict_map_ordinal
    # enc = OrdinalEncoder(dtype="int")
    # enc.fit_transform(data_)


def is_number_tryexcept(s):
    """ Returns True is string is a number. """
    try:
        if s == 'nan':
            return False
        float(s)
        return True
    except ValueError:
        return False


def remove_not_numeric(data):
    if data.shape[1] > 0:
        bool_numeric = data.apply(lambda x: x.apply(str).apply(is_number_tryexcept))
        drop_index, drop_col = [], []
        # row with only not numeric values
        bool_index = (~bool_numeric).all(axis=1)
        for index in data.index:
            if bool_index[index]:
                data = data.drop([index], axis=0)
                drop_index.append(index)

        # column with only not numeric values
        bool_col = (~bool_numeric).all(axis=0)
        for col in data.columns:
            if bool_col[col]:
                data = data.drop([col], axis=1)
                drop_col.append(col)

        if len(drop_index) > 0:
            print('index remove because not numeric :', drop_index)
        if len(drop_col) > 0:
            print('columns remove because not numeric :', drop_col)
    return data


def find_optimal_number(X):
    sil_score_max = 0
    from sklearn.metrics import silhouette_score
    for n_clusters in range(3, 15):
        model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1, random_state=0)
        labels = model.fit_predict(X)
        sil_score = silhouette_score(X, labels)
        if sil_score > sil_score_max:
            sil_score_max = sil_score
            best_n_clusters = n_clusters
    return best_n_clusters
