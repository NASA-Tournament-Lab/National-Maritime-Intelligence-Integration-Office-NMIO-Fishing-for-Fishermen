import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
from sklearn import model_selection, metrics, pipeline, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals.joblib import Parallel, delayed
import xgboost as xgb
import lightgbm as lgb

train = pd.read_csv('../input/training.txt', header=None, names=['id', 'type'])
test = pd.read_csv('../input/testing.txt', header=None, names=['id'])
ship_classes = ['trawler', 'longliner', 'seiner', 'other', 'support']
type_to_idx = {t: i for i, t in enumerate(ship_classes)}
train.type = train.type.transform(lambda x: type_to_idx[x])

kfold = model_selection.StratifiedKFold(4, shuffle=True, random_state=123)
folds = []
for train_idxs, test_idxs in kfold.split(train, train.type):
    folds.append((train.iloc[train_idxs], train.iloc[test_idxs]))

def read_track(idx):
    return pd.read_csv('../input/VesselTracks/{}.csv'.format(idx))

def save_result(ids, probas, filename):
    import csv
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for row_num in range(ids.shape[0]):
            for cls_num in range(5):
                writer.writerow((ids[row_num], ship_classes[cls_num], probas[row_num, cls_num]))

percentiles = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]

def extract_percentiles_row(track):
    track.columns = ['TrackNumber', 'Time(seconds)', 'Latitude', 'Longitude', 'SOG',
       'Oceanic Depth', 'Chlorophyll Concentration', 'Salinity',
       'Water Surface Elevation', 'Sea Temperature', 'Thermocline Depth',
       'Eastward Water Velocity', 'Northward Water Velocity']
    return np.hstack([np.percentile(track[c], percentiles)] for c in [
        'SOG',
        'Oceanic Depth',
        'Chlorophyll Concentration',
        'Salinity',
        'Water Surface Elevation',
        'Sea Temperature',
        'Thermocline Depth',
        'Eastward Water Velocity',
        'Northward Water Velocity',
    ])

def extract_percentiles(idxs):
    return np.vstack([extract_percentiles_row(read_track(idx)) for idx in idxs])

train_all_percentiles, test_all_percentiles = extract_percentiles(train.id), extract_percentiles(test.id)

def load_fold(name):
    return np.load('../input/{}_train.npy'.format(name)), np.load('../input/{}_test.npy'.format(name))

# Calculate distance features

def calc_distances(idx, columns, group_columns, h_percentiles, v_percentiles):
    idx = str(idx)
    data = pd.read_csv('../input/TrackDistances/{}.csv'.format(idx), engine='c')
    parts = []
    for key in range(5):
        columns = group_columns[key]
        if idx in columns:
            columns = list(columns)
            columns.remove(idx)
        col_data = data[columns]
        h_data = np.percentile(col_data.values, h_percentiles, axis=1)
        result = np.percentile(h_data, v_percentiles, axis=1)
        parts.append(result.ravel())
    return np.hstack(parts)

def get_part_distances(part, columns, group_columns, percentiles, n_jobs=-1):
    parts = Parallel(n_jobs=n_jobs)(
        delayed(calc_distances)(idx, columns, group_columns, percentiles, percentiles)
        for idx in part.id)
    return np.vstack(parts)

def get_distance_features(train, test, percentiles):
    groups = {}
    group_columns = {}
    for key, group in train.groupby('type'):
        columns = [str(i) for i in group.id]
        group_columns[key] = columns
    train_features = get_part_distances(train, columns, group_columns, percentiles)
    test_features = get_part_distances(test, columns, group_columns, percentiles)
    return train_features, test_features


percentiles = list(range(0, 101, 10))
for idx, fold in enumerate(folds):
    fold_train, fold_test = get_distance_features(fold[0], fold[1], percentiles)
    np.save('../input/fold_{}_distances_train.npy'.format(idx), fold_train)
    np.save('../input/fold_{}_distances_test.npy'.format(idx), fold_test)
    print('DONE', idx)

full_train, full_test = get_distance_features(train, test, percentiles)
np.save('../input/full_distances_train.npy', full_train)
np.save('../input/full_distances_test.npy', full_test)


# Load calculated distance features

fold_distances = [load_fold('fold_{}_distances'.format(i)) for i in range(4)]
all_fold_distances = np.vstack([f[1] for f in fold_distances])
all_fold_types = np.hstack([f[1].type.values for f in folds])

train_speed_matrices, test_speed_matrices = np.load('../input/train_speed_matrices.npy'), np.load('../input/test_speed_matrices.npy') 
train_speed_matrices = train_speed_matrices / np.sum(train_speed_matrices, axis=1)[:, None]
test_speed_matrices = test_speed_matrices / np.sum(test_speed_matrices, axis=1)[:, None]

train_speed_percentiles, test_speed_percentiles = np.load('../input/train_speed_percentiles.npy'), np.load('../input/test_speed_percentiles.npy') 
extended_features = np.hstack([train_speed_matrices, train_speed_percentiles])
extended_features = np.hstack([
    all_fold_distances,
    np.vstack([extended_features[test_idxs] for _, test_idxs in kfold.split(train, train.type)]),
])

extended_features_with_percentiles = np.hstack([
    extended_features,
    np.vstack([train_all_percentiles[test_idxs] for _, test_idxs in kfold.split(train, train.type)]),
])

gbm_args = dict(
    base_score=0.5,
    colsample_bylevel=0.9,
    colsample_bytree=0.9,
    gamma=0.65,
    learning_rate=0.075,
    max_delta_step=0,
    max_depth=7,
    min_child_weight=1,
    missing=None,
    n_estimators=250,
    nthread=-1,
    objective='multi:softprob',
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    silent=True,
    subsample=0.7
)

models = [xgb.XGBClassifier(seed=123 + i, **gbm_args) for i in range(15)]

gbm_args_1 = dict(
    base_score=0.5,
    colsample_bylevel=0.5,
    colsample_bytree=0.5,
    gamma=0.65,
    learning_rate=0.065,
    max_delta_step=0,
    max_depth=7,
    min_child_weight=1,
    missing=None,
    n_estimators=200,
    nthread=-1,
    objective='multi:softprob',
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    silent=True,
    subsample=0.7
)

models_1 = [xgb.XGBClassifier(seed=123 + i, **gbm_args_1) for i in range(7)]

gbm_args_2 = dict(
    base_score=0.5,
    colsample_bylevel=0.5,
    colsample_bytree=0.5,
    gamma=0.65,
    learning_rate=0.07,
    max_delta_step=0,
    max_depth=10,
    min_child_weight=1,
    missing=None,
    n_estimators=200,
    nthread=-1,
    objective='multi:softprob',
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    silent=True,
    subsample=0.6
)

models_2 = [xgb.XGBClassifier(seed=1234 + i, **gbm_args_1) for i in range(7)]

lgm_args_1 = {
    'learning_rate': 0.075,
    'n_estimators': 100,
    'max_depth': 5,
}

lgm_models_1 = [lgb.LGBMClassifier(seed=123 + i, **lgm_args_1) for i in range(7)]

lgm_args_2 = {
    'learning_rate': 0.075,
    'n_estimators': 100,
    'max_depth': 8,
}

lgm_models_2 = [lgb.LGBMClassifier(seed=123 + i, **lgm_args_2) for i in range(7)]

best_knn = pipeline.Pipeline([
    ('scaler', preprocessing.StandardScaler()),
    ('clf', KNeighborsClassifier(n_neighbors=11)),
])

mlp_args = {'hidden_layer_sizes': (200,), 'solver': 'sgd', 'max_iter': 1200, 'alpha': 1.75}
mlp_models = [pipeline.Pipeline([
    ('scaler', preprocessing.StandardScaler()),
    ('clf', MLPClassifier(random_state=234+i, **mlp_args)),
]) for i in range(7)]

for m in models:
    m.fit(extended_features, all_fold_types)

for m in models_1:
    m.fit(extended_features_with_percentiles, all_fold_types)
    
for m in models_2:
    m.fit(extended_features_with_percentiles, all_fold_types)

for m in lgm_models_1:
    m.fit(extended_features, all_fold_types)

for m in lgm_models_2:
    m.fit(extended_features_with_percentiles, all_fold_types)

best_knn.fit(extended_features_with_percentiles, all_fold_types)
    
for m in mlp_models:
    m.fit(extended_features_with_percentiles, all_fold_types)

knn_models = [best_knn]

train_dist, test_dist = load_fold('full_distances')
extended_features_test = np.hstack([test_dist, test_speed_matrices, test_speed_percentiles])
extended_features_with_all_percentiles_test = np.hstack([test_dist, test_speed_matrices, test_speed_percentiles, test_all_percentiles])
parts_1 = [[m.predict_proba(extended_features_test) for m in ms] for ms in [models, lgm_models_1]]
parts_2 = [[m.predict_proba(extended_features_with_all_percentiles_test) for m in ms] for ms in [models_1, models_2, lgm_models_2, knn_models, mlp_models]]
result = np.mean([np.mean(p, axis=0) for p in parts_1 + parts_2], axis=0)
save_result(test.id.values, result, '../submission.csv')

