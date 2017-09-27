import collections
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.externals.joblib import Parallel, delayed


train = pd.read_csv('../input/training.txt', header=None, names=['id', 'type'])
test = pd.read_csv('../input/testing.txt', header=None, names=['id'])

type_to_idx = {t: i for i, t in enumerate(['trawler', 'longliner', 'seiner', 'other', 'support'])}
train.type = train.type.transform(lambda x: type_to_idx[x])


def read_track(idx):
    return pd.read_csv('../input/VesselTracks/{}.csv'.format(idx))


def calc_distances(idx, kdtrees):
    data = read_track(idx)[['Latitude', 'Longitude']]
    columns = []
    for tree in kdtrees.values():
        distances, indices = query = tree.query(data, k=1)
        columns.append(distances)
    columns_np = np.hstack(columns)
    df = pd.DataFrame(columns_np, columns=kdtrees.keys())
    df['TrackNumber'] = idx
    df.to_csv('../input/TrackDistances/{}.csv'.format(idx, kdtrees), index=None)
    print('DONE', idx)


kdtrees = collections.OrderedDict()

for idx in train.id:
    data = read_track(idx)
    data = data[['Latitude', 'Longitude']]
    data_mid = data
    data_left = data_mid[data_mid.Longitude >= 0].copy()
    data_left.Longitude -= 360
    data_right = data_mid[data_mid.Longitude < 0].copy()
    data_right.Longitude += 360
    data = pd.concat([data_left, data_mid, data_right])
    kdtrees[idx] = KDTree(data, leaf_size=100)

print('FINISHED BUILDING TREES')

Parallel(n_jobs=-1)(delayed(calc_distances)(idx, kdtrees) for idx in train.id)
Parallel(n_jobs=-1)(delayed(calc_distances)(idx, kdtrees) for idx in test.id)
