import os
import numpy as np
import pandas as pd
from sklearn.model_selection import LeavePGroupsOut

from src.eval import type_order


COLUMNS = ['TrackNumber', 'Time(seconds)', 'Latitude', 'Longitude', 'SOG', 'Oceanic Depth', 'Chlorophyll Concentration',
           'Salinity', 'Water Surface Elevation', 'Sea Temperature', 'Thermocline Depth', 'Eastward Water Velocity', 'Northward Water Velocity']

COLS = ['TrackNumber_NEW', 'Latitude', 'Longitude', 'SOG', 'Oceanic Depth', 'diff_lat', 'diff_lon', 'diff_lat_norm', 'diff_lon_norm']
QUANTILES = [0, .01, .1, .25, .5, .75, .9, .99, 1]


def load_data_train_raw(dataset_dir):
    df_train_y_unique = pd.read_csv(os.path.join(dataset_dir, 'training.txt'), names=['TrackNumber', 'Type'])

    dfs_raw = []

    for i, _file in enumerate(df_train_y_unique['TrackNumber'].tolist()):
        df_tmp = pd.read_csv(os.path.join(dataset_dir, 'VesselTracks', '{}.csv'.format(_file)))
        df_tmp.columns = COLUMNS
        dfs_raw.append(df_tmp)


    df_train_x = pd.concat(dfs_raw).reset_index(drop=True)

    df_train_y = df_train_x[['TrackNumber']]
    df_train_y = pd.merge(df_train_y, df_train_y_unique, left_on='TrackNumber', right_on='TrackNumber', how='left')

    return df_train_x, df_train_y, df_train_y_unique

def load_data_test_raw(dataset_dir):
    df_test_y_unique = pd.read_csv(os.path.join(dataset_dir, 'testing.txt'), names=['TrackNumber'])

    dfs_raw = []

    for i, _file in enumerate(df_test_y_unique['TrackNumber'].tolist()):
        df_tmp = pd.read_csv(os.path.join(dataset_dir, 'VesselTracks', '{}.csv'.format(_file)))
        df_tmp.columns = COLUMNS
        dfs_raw.append(df_tmp)

    df_test_x = pd.concat(dfs_raw).reset_index(drop=True)

    df_test_y = df_test_x[['TrackNumber']]
    df_test_y = pd.merge(df_test_y, df_test_y_unique, left_on='TrackNumber', right_on='TrackNumber', how='left')

    return df_test_x, df_test_y, df_test_y_unique


def split_data(df_x, df_y, pct_test=.2):
    lpgo = LeavePGroupsOut(int(len(df_y['TrackNumber'].unique().tolist()) * pct_test))

    # Simply take the first split
    for train_ids, test_ids in lpgo.split(df_x, df_y['Type'], groups=df_y['TrackNumber']):
        break

    df_eval_train_x = df_x.loc[train_ids.tolist()].fillna(0)
    df_eval_test_x = df_x.loc[test_ids.tolist()].fillna(0)

    df_eval_train_y = df_y.loc[train_ids.tolist()][['Type']]
    df_eval_test_y = df_y.loc[test_ids.tolist()][['Type']]

    df_eval_train_y = pd.get_dummies(df_eval_train_y, prefix='', prefix_sep='')[type_order]
    df_eval_test_y = pd.get_dummies(df_eval_test_y, prefix='', prefix_sep='')[type_order]

    df_eval_train_tracks = df_y.loc[train_ids.tolist()][['TrackNumber']]
    df_eval_test_tracks = df_y.loc[test_ids.tolist()][['TrackNumber']]

    return df_eval_train_x, df_eval_test_x, df_eval_train_y, df_eval_test_y, df_eval_train_tracks, df_eval_test_tracks


def pre_process_data(df_x):
    df_x['Week'] = df_x['Time(seconds)'].apply(lambda x: int(np.floor(x / (60 * 60 * 24 * 31))))
    df_x['TrackNumber_NEW'] = df_x['Week'] * 1000000000 + df_x['TrackNumber']

    df_x['diff_lat'] = df_x.groupby('TrackNumber_NEW')['Latitude'].diff(1).fillna(0)
    df_x['diff_lon'] = df_x.groupby('TrackNumber_NEW')['Longitude'].diff(1).fillna(0)

    df_x['diff_lat_norm'] = df_x['diff_lat'].apply(lambda x: min(x % 360, -x % 360))
    df_x['diff_lon_norm'] = df_x['diff_lon'].apply(lambda x: min(x % 180, -x % 180))

    df_x_tmp = df_x.groupby('TrackNumber_NEW').quantile(QUANTILES).fillna(0).stack().reset_index()
    df_x_tmp['key'] = df_x_tmp.apply(lambda x: str(x['level_1']) + '_' + x['level_2'], axis=1)
    df_x_new = pd.pivot_table(df_x_tmp, index='TrackNumber_NEW', columns='key', values=0)

    track_number_counts = pd.DataFrame(df_x['TrackNumber_NEW'].value_counts()).sort_index()
    df_x_new['TrackCount'] = track_number_counts

    df_x_tmp_2 = df_x[df_x['Oceanic Depth'] > -50][COLS].groupby('TrackNumber_NEW').quantile(QUANTILES).fillna(0).stack().reset_index()
    df_x_tmp_2['key'] = df_x_tmp_2.apply(lambda x: str(x['level_1']) + '_' + x['level_2'], axis=1)
    df_x_new_2 = pd.pivot_table(df_x_tmp_2, index='TrackNumber_NEW', columns='key', values=0)
    df_x_new = pd.merge(df_x_new, df_x_new_2, left_index=True, right_index=True, suffixes=('', '_not_deep_1'), how='left')

    df_x_tmp_2 = df_x[(df_x['Oceanic Depth'] < -50) & (df_x['Oceanic Depth'] > -250)][COLS].groupby('TrackNumber_NEW').quantile(QUANTILES).fillna(0).stack().reset_index()
    df_x_tmp_2['key'] = df_x_tmp_2.apply(lambda x: str(x['level_1']) + '_' + x['level_2'], axis=1)
    df_x_new_2 = pd.pivot_table(df_x_tmp_2, index='TrackNumber_NEW', columns='key', values=0)
    df_x_new = pd.merge(df_x_new, df_x_new_2, left_index=True, right_index=True, suffixes=('', '_not_deep_2'), how='left')

    df_x_tmp_2 = df_x[(df_x['Oceanic Depth'] < -250) & (df_x['Oceanic Depth'] > -500)][COLS].groupby('TrackNumber_NEW').quantile(QUANTILES).fillna(0).stack().reset_index()
    df_x_tmp_2['key'] = df_x_tmp_2.apply(lambda x: str(x['level_1']) + '_' + x['level_2'], axis=1)
    df_x_new_2 = pd.pivot_table(df_x_tmp_2, index='TrackNumber_NEW', columns='key', values=0)
    df_x_new = pd.merge(df_x_new, df_x_new_2, left_index=True, right_index=True, suffixes=('', '_not_deep_3'), how='left')

    df_x_tmp_2 = df_x[df_x['Oceanic Depth'] < -500][COLS].groupby('TrackNumber_NEW').quantile(QUANTILES).fillna(0).stack().reset_index()
    df_x_tmp_2['key'] = df_x_tmp_2.apply(lambda x: str(x['level_1']) + '_' + x['level_2'], axis=1)
    df_x_new_2 = pd.pivot_table(df_x_tmp_2, index='TrackNumber_NEW', columns='key', values=0)
    df_x_new = pd.merge(df_x_new, df_x_new_2, left_index=True, right_index=True, suffixes=('', '_not_deep_4'), how='left')

    df_x_tmp_2 = df_x[df_x['SOG'] > 2][COLS].groupby('TrackNumber_NEW').quantile(QUANTILES).fillna(0).stack().reset_index()
    df_x_tmp_2['key'] = df_x_tmp_2.apply(lambda x: str(x['level_1']) + '_' + x['level_2'], axis=1)
    df_x_new_2 = pd.pivot_table(df_x_tmp_2, index='TrackNumber_NEW', columns='key', values=0)
    df_x_new = pd.merge(df_x_new, df_x_new_2, left_index=True, right_index=True, suffixes=('', '_sog_1'), how='left')

    df_x_tmp_2 = df_x[df_x['SOG'] < 2][COLS].groupby('TrackNumber_NEW').quantile(QUANTILES).fillna(0).stack().reset_index()
    df_x_tmp_2['key'] = df_x_tmp_2.apply(lambda x: str(x['level_1']) + '_' + x['level_2'], axis=1)
    df_x_new_2 = pd.pivot_table(df_x_tmp_2, index='TrackNumber_NEW', columns='key', values=0)
    df_x_new = pd.merge(df_x_new, df_x_new_2, left_index=True, right_index=True, suffixes=('', '_sog_2'), how='left')

    df_x_tmp = df_x.fillna(0)[COLS].groupby('TrackNumber_NEW').std()
    df_x_new = pd.merge(df_x_new, df_x_tmp, left_index=True, right_index=True, suffixes=('', '_std'))

    df_x_tmp = df_x.fillna(0)[COLS].groupby('TrackNumber_NEW').mean()
    df_x_new = pd.merge(df_x_new, df_x_tmp, left_index=True, right_index=True, suffixes=('', '_mean'))

    return df_x_new


def save_submission(df_test_preds, df_test_y, file_path):
    df_test_preds.index = df_test_y[['TrackNumber']]
    df_test_preds = df_test_preds.reset_index()

    ret = df_test_preds.groupby('index').mean().stack().reset_index()
    ret.columns = ['Track#', 'FishingType', 'Prob']

    if os.path.exists(file_path):
        print("File already exists !!!")
        return

    ret.to_csv(file_path, index=False, header=False)
    