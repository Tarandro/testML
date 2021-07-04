import random as rd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def get_features_importance(data, Y, subsample, class_weight):
    index_sample = rd.sample(list(data.index), int(len(data) * subsample))
    try:
        clf = RandomForestClassifier(random_state=15, class_weight=class_weight, n_estimators=50, max_samples=0.8)
        clf.fit(data.loc[index_sample, :], Y.loc[index_sample, :])
    except:
        clf = RandomForestRegressor(random_state=15, n_estimators=50, max_samples=0.8)
        clf.fit(data.loc[index_sample, :], Y.loc[index_sample, :])
    return clf.feature_importances_
