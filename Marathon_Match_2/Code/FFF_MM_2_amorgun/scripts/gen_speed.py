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


percentiles = [0,1,5,10,20,30,40,50,60,70,80,90,95,99,100]

speed_bins = np.array([
        0.00000000e+00,   2.06001643e-03,   5.90728896e-03,
        1.87525583e-02,   1.56496151e-01,   1.08085267e+00,
        2.05829826e+00,   4.08043981e+00,   5.65426157e+00,
        6.88369766e+00,   9.07828686e+00,   1.35635645e+07])

def get_deltas(part):
    R = 6371e3
    part = part.sort_values('Time(seconds)')
    first, second = part[:-1], part[1:]
    result = pd.DataFrame()
    result['d_t'] = second['Time(seconds)'].values - first['Time(seconds)'].values
    phi_1, phi_2 = np.radians(first['Latitude'].values), np.radians(second['Latitude'].values)
    d_phi = phi_2 - phi_1
    d_lambda = np.radians(second['Longitude'].values - first['Longitude'].values)
    a = np.sin(d_phi / 2) ** 2 + np.cos(phi_1) * np.cos(phi_2) * np.sin(d_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    result['d_L'] = c * R
    result['speed'] = result['d_L'] / result['d_t']
    return result

def get_speed_stats(deltas):
    dd = np.digitize(deltas['speed'], speed_bins)
    s = speed_bins.shape[0]
    flat_coords = np.ravel_multi_index((dd[:-1], dd[1:]), (s, s))
    return np.bincount(flat_coords, minlength=s*s).reshape((s, s))

def get_speed_matrix(idx):
    track = read_track(idx)
    if track.shape[0] <= 1:
        return np.zeros((speed_bins.shape[0], speed_bins.shape[0]))
    deltas = get_deltas(track)
    return get_speed_stats(deltas).ravel()
    

def get_speed_percentiles(idx):
    track = read_track(idx)
    if track.shape[0] <= 1:
        return np.zeros((speed_bins.shape[0]))
    deltas = get_deltas(track)
    return np.percentile(deltas['speed'], percentiles)

def gen_speed_matrices(filename, ids):
    parts = Parallel(n_jobs=-1)(delayed(get_speed_matrix)(idx) for idx in ids)
    result = np.vstack(parts)
    np.save('../input/{}.npy'.format(filename), result)

def gen_speed_percentiles(filename, ids):
    parts = Parallel(n_jobs=-1)(delayed(get_speed_percentiles)(idx) for idx in ids)
    result = np.vstack(parts)
    np.save('../input/{}.npy'.format(filename), result)

gen_speed_matrices('train_speed_matrices', train.id)
gen_speed_matrices('test_speed_matrices', test.id)
gen_speed_matrices('train_speed_percentiles', train.id)
gen_speed_matrices('test_speed_percentiles', test.id)
