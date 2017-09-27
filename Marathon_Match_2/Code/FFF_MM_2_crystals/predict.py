# coding: utf-8

import math
import os
import datetime
import pandas as pd
import numpy as np
import gc

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from mlxtend.classifier import StackingCVClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileTransformer
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

from prepare import prepare
  
seed = 42
classes = ['trawler', 'seiner', 'longliner', 'support', 'other']
operations = ['mean', 'median', 'q25', 'q75', 'std', 'min', 'max']
data_path = 'combined.csv'


def read_train(path, rows=None, names=['Track_Number', 'Fishing_Type']):
    print "Reading the train data"
    start = datetime.datetime.now()
    data = pd.read_csv(path, names=names, nrows=rows)
    print datetime.datetime.now()-start
    return data


def read_X_test(path, names=['Track_Number']):
    print "read X_test", path
    return pd.read_csv(path, names=names)


def read_all(path):
    start = datetime.datetime.now()
    dtype = {
        'TrackNumber': np.int32,
        'Time(seconds)': np.int32,
        'Latitude': np.float32,
        'Longitude': np.float32,
        'SOG': np.float32,
        'oceanic depth': np.int32, 
        'Chlorophyll Concentration': np.float32,
        'Salinity': np.float32,
        'Water Surface Elevation': np.float32,
        'Sea Temperature': np.float32,
        'Thermocline Depth': np.float32,
        'Eastward Water Velocity': np.float32,
        'Northward Water Velocity': np.float32,
        'Water Velocity': np.float32
    }
    dtypes = {'%s_%s' % (op ,dt.replace(' ','_')):v for dt,v in dtype.iteritems() for op in operations }
    dtypes['Sog_lt5_median'] = np.float32
    dtypes['Sog_gte5_median'] = np.float32

    data = pd.read_csv(path, header=0, dtype=dtype)
    print datetime.datetime.now()-start
    return data


def clean_data(data, columns=['TrackNumber', 'Fishing_Type', 'Track_Number']):
    for col in columns:
        if col in data.columns:
            data.drop(col, inplace=True, axis=1)
    return data


def select_x(train_data, train, i_fold):
    if i_fold is not None:
        data_x = pd.DataFrame(train_data[train_data['TrackNumber'].isin(train['Track_Number'].iloc[i_fold])])
    else:
        data_x = pd.DataFrame(train_data[train_data['TrackNumber'].isin(train['Track_Number'])])
    return clean_data(data_x) 


def select_y(train_data, train, i_fold, binarize=True):
    if i_fold is not None:
        data = train_data[train_data['TrackNumber'].isin(train['Track_Number'].iloc[i_fold])]['Fishing_Type']
    else:
        data = train_data[train_data['TrackNumber'].isin(train['Track_Number'])]['Fishing_Type']
    if binarize:
        return label_binarize(data, classes=classes)
    else:
        data = pd.Series(data)
        for i_col, col in enumerate(classes):
             data.loc[data==col] = str(i_col)
    return data.values


def select_track(train_data, train, i_fold):
    if i_fold is not None:
        return train_data[train_data['TrackNumber'].isin(train['Track_Number'].iloc[i_fold])]['TrackNumber']
    else:
        return train_data[train_data['TrackNumber'].isin(train['Track_Number'])]['TrackNumber']


def collapse(data, track_numbers, use_other=False, normalize=True):
    a = pd.DataFrame(data, columns=classes)
    b = pd.Series(track_numbers, name='TrackNumber', dtype=np.int32)
    a.index = pd.RangeIndex(len(a.index))
    b.index = pd.RangeIndex(len(b.index))
    df = pd.concat([
        a,
        b, 
    ], axis=1)
    means = df.groupby('TrackNumber').mean()
    if normalize:
        if not use_other:
            other = means.other
            means.drop('other', axis=1, inplace=True)
        norm = means.sum(axis=1)
        means['trawler'] =  means['trawler']/norm
        means['seiner'] =  means['seiner']/norm
        means['longliner'] =  means['longliner']/norm
        means['support'] =  means['support']/norm
        if use_other:
            means['other'] =  means['other']/norm
        else:
            means.loc[:,'other'] = other.values/norm
    print means.head()
    return means


def purge_data(data):
    drop_cols = [
        'min_Sog',
        'min_Time(Seconds)',
        'min_Salinity',
        'max_Salinity', 
        'min_Distance',
        'min_Total_Distance',
        'min_Speed',
        'min_Oceanic_Depth',
        'min_Chlorophyll_Concentration',
        'min_Time_Diff',
        'q25_Time(Seconds)',
        'min_Water_Velocity',
        'q25_Northward_Water_Velocity',
        'max_Northward_Water_Velocity',
        'median_Eastward_Water_Velocity',
        'min_Northward_Water_Velocity',
        'mean_Eastward_Water_Velocity',
        'q75_Northward_Water_Velocity',
        'std_Northward_Water_Velocity',
        'q75_Chlorophyll_Concentration',
        'median_Northward_Water_Velocity',
        'q25_Chlorophyll_Concentration',
        'q25_Sea_Temperature',
        'mean_Northward_Water_Velocity',
        'q25_Water_Velocity',
        'q75_Water_Velocity',
        'median_Salinity',
        'std_Water_Velocity',
        'gte5_median_Salinity',
        'median_Water_Surface_Elevation',
        'mean_Total_Distance',
        'mean_Sea_Temperature',
        'median_Water_Velocity',
        'q75_Sea_Temperature',
        'mean_Speed',
        'median_Sea_Temperature',
        'q25_Time_Diff',
        'std_Eastward_Water_Velocity',
        'q75_Latitude',
        'lt5_median_Longitude',
        'q25_Latitude',
        'q75_Longitude',
        'q25_Longitude',
    ]

    for col in drop_cols:
        data = data.drop(col, axis=1)

    for col in data.columns:
        pass 
        if 'Latitude' in col or 'Longitude' in col:
            data.loc[:, col] = data[col].round(0)
        if 'Time(Seconds)' in col:
            data = data.drop(col, axis=1)

    data = data.round(2)
    return data 


def pred_cv(clean=True, N=3000000, print_importances=False, simple_input=True, classifier='knn', smote=True):
    binarize = False
 
    print 'using %s classifer  with %s examples, binarize? %s' %(classifier, N, binarize)

    train_path = 'training.txt'
    train = read_train(train_path, rows=N)
    print "Train data:"
    print train.head()
 
    test_path = 'testing.txt'
    test_x = read_X_test(test_path)
    print "Test data:"
    print test_x.head()
    
    train_data = read_all(data_path)
    train_data = train_data.merge(train, left_on='TrackNumber', right_on='Track_Number', how='inner')

    if clean:
        print "Before clean", train_data.shape
        train_data = purge_data(train_data)
        print "after clean", train_data.shape
        print train_data.head()

    cv =  RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    auc_score = dict()
    fold_i = -1
    importances = dict()

    for fold_train, fold_test in cv.split(train['Track_Number'].values, train['Fishing_Type'].values):
        fold_i += 1
        print 'Starting Fold', fold_i

        X_train = select_x(train_data, train, fold_train)
        X_train_columns = X_train.columns
        x = X_train.as_matrix()
        gc.collect()

        y = select_y(train_data, train, fold_train, binarize=binarize)
        gc.collect()

        X_test = select_x(train_data, train, fold_test)
        Y_test = select_y(train_data, train, fold_test)
        Y_test_track = select_track(train_data, train, fold_test)
        gc.collect()

        if classifier == 'knn':
            clf = KNeighborsClassifier(n_neighbors=15, p=1, weights='distance', n_jobs=-1)
        elif classifier == 'ovrrf':
            clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=125, max_features=0.25, n_jobs=-1, random_state=seed, class_weight='balanced'))
        elif classifier == 'rf':
            clf = RandomForestClassifier(n_estimators=199, max_features=0.18, n_jobs=-1, random_state=seed, class_weight='balanced')
        elif classifier == 'lgbm':
            clf = lgb.LGBMClassifier(seed=seed, nthread=8, num_leaves=64, learning_rate=0.09,
                   n_estimators=139, min_child_weight=4, min_child_samples=2, min_split_gain=0, subsample=1,
                   colsample_bytree=0.58, scale_pos_weight=1, silent=True)
        elif classifier == 'lrrf':
            np.random.seed(seed)
            knn = KNeighborsClassifier(n_neighbors=15, p=1, weights='distance', n_jobs=-1)
            clf1 = RandomForestClassifier(n_estimators=199, max_features=0.18, n_jobs=-1, random_state=seed, class_weight='balanced')
            clf2 = lgb.LGBMClassifier(seed=seed, nthread=8, num_leaves=64, learning_rate=0.09,
                   n_estimators=139, min_child_weight=4, min_child_samples=2, min_split_gain=0, subsample=1,
                   colsample_bytree=0.58, scale_pos_weight=1, silent=True)
            meta = lgb.LGBMClassifier(seed=seed, nthread=8, num_leaves=64, learning_rate=0.09, 
                   n_estimators=139, min_child_weight=4, min_child_samples=2, min_split_gain=0, subsample=1,
                   colsample_bytree=0.58, scale_pos_weight=1, silent=True)
            clf = StackingCVClassifier(classifiers=[knn, clf1, clf2], use_probas=True, cv=10, meta_classifier=meta, use_features_in_secondary=True)

        inputer = Imputer(strategy="mean", axis=0)
        x = inputer.fit_transform(x)
        X_test = inputer.transform(X_test)
        print "NaN's: ", np.sum(np.isnan(x))

        scaler = QuantileTransformer(output_distribution='uniform')
        x = scaler.fit_transform(x)
        X_test = scaler.transform(X_test)

        if smote:
            print 'BEFORE SMOTE', x.shape
            sm = SMOTE(random_state=seed, ratio=0.82, n_jobs=-1, k_neighbors=5, m_neighbors=15)
            x, y = sm.fit_sample(x, y)
            print 'AFTER SMOTE: ', x.shape
       
        print 'fit'
        start = datetime.datetime.now()
        clf.fit(x, y) 
        print 'fit took %s s, size %s' % ((datetime.datetime.now()-start).total_seconds(), x.shape)

        start = datetime.datetime.now()
        print 'predict'
        probas = clf.predict_proba(X_test)
        print 'predict took %s s, size %s' % ((datetime.datetime.now()-start).total_seconds(), X_test.shape)

        for i, cls in enumerate(classes):
            name = "%s" % (cls)
            fpr.setdefault(name, [])
            tpr.setdefault(name, [])
            roc_auc.setdefault(name, [])
            
            fpr[name].append(0)
            tpr[name].append(0)
            roc_auc[name].append(0)
            
            fpr[name][fold_i], tpr[name][fold_i], _ = roc_curve(Y_test[:, i], probas[:, i])
            roc_auc[name][fold_i] = auc(fpr[name][fold_i], tpr[name][fold_i])

            print 'AUC', name, fold_i, roc_auc[name][fold_i]
        if print_importances:
            for colum, imp in sorted(zip(X_train_columns, clf.feature_importances_), key=lambda x: x[1] * -1):
                importances.setdefault(colum, [])
                importances[colum].append(imp)
    
    if print_importances:
        print '======================='      
        for colum, imp in sorted(importances.iteritems(), key=lambda x: -sum(x[1])):
            print "%.5f: %s" % (sum(imp)/len(imp), colum)
        print '=======================' 

    score = {}
    for i, cls in enumerate(classes):
        score_i = sum(roc_auc[cls])/len(roc_auc[cls])
        score[cls] = score_i
        print 'SUBSCORE', cls, score_i

    sum_a = sum(score.values())/len(classes)
    print 'SCORE %.5f(%.5f)' % (sum_a, math.sqrt(sum(((x-sum_a)*(x-sum_a) for x in score.values()))/(len(classes)-1)))
    print 'WGH %.5f' % (score['trawler']*0.4 + score['seiner']*0.3 + score['longliner']*0.2 + score['support']*0.1)


def extract_column(probas_collapse, name):
    out = pd.DataFrame()
    out['TrackNumber'] = probas_collapse.index.get_values()
    out.index = probas_collapse.index.get_values()
    out['FishingType'] = name
    out.loc[:,'Prob'] = probas_collapse[name].values
    return out


def prepare_out_multi(probas_collapse, with_other=True):
    print probas_collapse.head()
    out_tr = extract_column(probas_collapse, 'trawler')
    out_se = extract_column(probas_collapse, 'seiner')
    out_lo = extract_column(probas_collapse, 'longliner')
    out_su = extract_column(probas_collapse, 'support')
    if with_other:
        out_ot = extract_column(probas_collapse, 'other')

    out = pd.DataFrame()
    out = out_tr.append(out_se, ignore_index=True)
    out = out.append(out_lo, ignore_index=True)
    out = out.append(out_su, ignore_index=True)
    if with_other:
        out = out.append(out_ot, ignore_index=True)

    print probas_collapse.shape, out.shape
    print out.head()
    out.to_csv('result.csv', index=False, header=False)


def make_pred(classifier='rf', clean=True, smote=False):
    binarize = True
    if classifier=='rf' or classifier=='lgbm' or classifier=='lrrf':
        binarize = False

    train_path = 'training.txt'
    train = read_train(train_path)
    print "Train data:"
    print train.head()
    test_path = 'testing.txt'  
    test_x = read_X_test(test_path)

    print "Test data:"
    print test_x.head()
    train_data = read_all(data_path)
    train_data = train_data.merge(train, left_on='TrackNumber', right_on='Track_Number', how='inner')

    if clean:
        print "Before clean", train_data.shape
        train_data = purge_data(train_data)
        print "after clean", train_data.shape
        print train_data.head()

    test_data = read_all(data_path)
    test_data = test_data.merge(test_x, left_on='TrackNumber', right_on='Track_Number', how='inner')

    if clean:
        print "Before clean", test_data.shape
        test_data = purge_data(test_data)
        print "after clean", test_data.shape
        print test_data.head()

    X_train = select_x(train_data, train, None)
    print 'X_train head'
    print X_train.head()
    x = X_train.as_matrix()
    X_train = None
    gc.collect()

    Y_train = select_y(train_data, train, None, binarize=binarize)
    print 'Y_train'
    print Y_train
    y = Y_train
    Y_train = None
    gc.collect()

    X_test = select_x(test_data, test_x, None)
    print "X_test head"
    print X_test.head()
    Y_test_track = select_track(test_data, test_x, None)

    print 'fit clf'
    if classifier == 'knn':   
        clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=9, p=1, weights='distance', n_jobs=-1), n_jobs=-1)
    elif classifier == 'ovrrf':
        clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=125, max_features=0.25, n_jobs=-1, random_state=seed, class_weight='balanced'))
    elif classifier == 'rf':
        clf = RandomForestClassifier(n_estimators=200, max_features=0.15, n_jobs=-1, random_state=seed, class_weight='balanced')
    elif classifier == 'lrrf':
        np.random.seed(seed)
        knn = KNeighborsClassifier(n_neighbors=15, p=1, weights='distance', n_jobs=-1)
        clf1 = RandomForestClassifier(n_estimators=199, max_features=0.18, n_jobs=-1, random_state=seed, class_weight='balanced')
        clf2 = lgb.LGBMClassifier(seed=seed, nthread=8, num_leaves=64, learning_rate=0.09,
               n_estimators=139, min_child_weight=4, min_child_samples=2, min_split_gain=0, subsample=1,
               colsample_bytree=0.58, scale_pos_weight=1, silent=True)
        meta = lgb.LGBMClassifier(seed=seed, nthread=8, num_leaves=64, learning_rate=0.09,
               n_estimators=139, min_child_weight=4, min_child_samples=2, min_split_gain=0, subsample=1,
               colsample_bytree=0.58, scale_pos_weight=1, silent=True)
        clf = StackingCVClassifier(classifiers=[clf1, clf2, knn], use_probas=True, cv=10, meta_classifier=meta, use_features_in_secondary=True)

    inputer = Imputer(strategy="mean", axis=0)
    x = inputer.fit_transform(x)
    X_test = inputer.transform(X_test)
    print "NaN's: ", np.sum(np.isnan(x))

    scaler = QuantileTransformer(output_distribution='uniform')
    x = scaler.fit_transform(x)
    X_test = scaler.transform(X_test)

    if smote:
        print 'BEFORE SMOTE', x.shape
        sm = SMOTE(random_state=seed, ratio=0.80, n_jobs=-1, k_neighbors=5, m_neighbors=15)
        x, y = sm.fit_sample(x, y)
        print 'AFTER SMOTE: ', x.shape
         
    print x.shape
    print y
    start = datetime.datetime.now()
    clf.fit(x, y)
    print 'fit took %s s, size %s' % ((datetime.datetime.now()-start).total_seconds(), x.shape)

    start = datetime.datetime.now()
    probas = clf.predict_proba(X_test)
    print 'predict took %s s, size %s' % ((datetime.datetime.now()-start).total_seconds(), X_test.shape)

    probas_collapse = collapse(probas, Y_test_track, normalize=False)
    prepare_out_multi(probas_collapse, with_other=True)


if __name__ == "__main__":
    prepare(data_path)
    #pred_cv(N=1209, classifier='lrrf', simple_input=False, print_importances=False, smote=False)
    make_pred(classifier='lrrf', smote=True)
