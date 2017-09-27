# coding: utf-8

import math
import os
import glob
import datetime
import pandas as pd
import numpy as np

def read(path):
    print "Reading", path
    start = datetime.datetime.now()
    dtype = {
        'TrackNumber': np.int32,
        'Time(seconds)': np.int32,
        'Latitude': np.float32,
        'Longitude': np.float32,
        'SOG': np.float32,
        'Oceanic Depth': np.int32,
        'Chlorophyll Concentration': np.float32,
        'Salinity': np.float32,
        'Water Surface Elevation': np.float32,
        'Sea Temperature': np.float32,
        'Thermocline Depth': np.float32,
        'Eastward Water Velocity': np.float32,
        'Northward Water Velocity': np.float32
    }
    data = pd.read_csv(path, header=0, dtype=dtype)
    print datetime.datetime.now()-start
    return data


def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def calc_dist(data, total=False):
    if total:
        out = haversine_np(data['Longitude'].values[0], data['Latitude'].values[0], data.loc[0:, 'Longitude'], data.loc[0:, 'Latitude'])
    else:
        out = haversine_np(data.Longitude.shift(), data.Latitude.shift(), data.loc[1:, 'Longitude'], data.loc[1:, 'Latitude'])
    out.fillna(0.0, inplace=True)
    return out 


def calc_time_diff(data):
    out = data.loc[1:, 'Time(seconds)']-data['Time(seconds)'].shift()
    out.loc[out<1] = 1
    return out


def calc_speed(data):
    t_diff = data.loc[1:, 'Time(seconds)']-data['Time(seconds)'].shift()
    t_diff.loc[t_diff<1] = 1
    t_diff = t_diff/3600
    out = data['Distance']/t_diff
    out.fillna(0.0, inplace=True)
    return out


def clean(data_nan, track_number):
    print track_number, 'in shape:', data_nan.shape
    data_nan[data_nan == -99999] = np.nan

    data_nan = data_nan.ix[data_nan['Latitude'] >= -90.0]
    data_nan = data_nan.ix[data_nan['Latitude'] <= 90.0]
    data_nan = data_nan.ix[data_nan['Longitude'] >= -180]
    data_nan = data_nan.ix[data_nan['Longitude'] <= 180]

    if data_nan.shape[0] == 0:
        print 'ZERO SIZE'
        return pd.DataFrame()

    for col in data_nan.columns:
        b_data_nan = pd.DataFrame(data_nan.loc[:,col])
        b_data_nan.fillna(method='bfill', inplace=True)
        f_data_nan = pd.DataFrame(data_nan.loc[:,col])
        f_data_nan.fillna(method='ffill', inplace=True)

        mask = data_nan.loc[:,col].isnull()
        data_nan.loc[mask,col] = f_data_nan

        mask = data_nan.loc[:,col].isnull()
        data_nan.loc[mask,col] = b_data_nan

    data_nan.loc[:, 'Distance'] = calc_dist(data_nan)
    data_nan.loc[:, 'Total Distance'] = calc_dist(data_nan, total=True)
    data_nan.loc[:, 'Speed'] = calc_speed(data_nan)   
    data_nan.loc[:, 'Time Diff'] = calc_time_diff(data_nan)
   
    data_nan['Water Velocity'] = np.sqrt(data_nan['Northward Water Velocity']*data_nan['Northward Water Velocity']+data_nan['Eastward Water Velocity']*data_nan['Eastward Water Velocity'])
    if 'Oceanic Depth' in data_nan.columns:
        data_nan.loc[:,'Oceanic Depth'] = data_nan['Oceanic Depth'].abs().values
    if 'oceanic depth' in data_nan.columns:
        data_nan.loc[:,'oceanic depth'] = data_nan['oceanic depth'].abs().values

    data_nan.loc[:, 'Salinity'] = np.round(data_nan['Salinity'].values, 2)
    data_nan.loc[:, 'Thermocline Depth'] = np.round(data_nan['Thermocline Depth'].values, 0)
    data_nan.loc[:, 'Sea Temperature'] = np.round(data_nan['Sea Temperature'].values, 2)
    data_nan.loc[:, 'Distance'] = np.round(data_nan['Distance'].values, 2)
    data_nan.loc[:, 'Total Distance'] = np.round(data_nan['Total Distance'].values, 2)
    data_nan.loc[:, 'Speed'] = np.round(data_nan['Speed'].values, 2)
    data_nan.loc[:, 'Water Velocity'] = np.round(data_nan['Water Velocity'].values, 2)
    
    data_nan = data_nan.ix[data_nan['Time Diff'] > 60]
    data_nan = data_nan.ix[data_nan['Speed'] < 60]
    data_nan = data_nan.ix[(np.isnan(data_nan['SOG'])) | ((data_nan['SOG'] < 25))]

    if data_nan.loc[:, 'Speed'].max() > 40:
        print 'high speed detected', track_number
        print data_nan[data_nan.loc[:, 'Speed'] > data_nan.loc[:, 'SOG'].max()*3].head()
    
    if data_nan.loc[:, 'SOG'].max() > 22:
        print 'high SOG detected', track_number
        print data_nan[data_nan.loc[:, 'SOG'] > 22].head()

    return data_nan
    
def calc_features(data_nan, track_number):
    sog_lt5_median = data_nan[(data_nan['SOG']<5) & (data_nan['SOG']>0.2)].groupby('TrackNumber').median()
    sog_gte5_median = data_nan[data_nan['SOG']>=5].groupby('TrackNumber').median()
    sog_lt5_std = data_nan[(data_nan['SOG']<5) & (data_nan['SOG']>0.2)].groupby('TrackNumber').std()
    sog_gte5_std = data_nan[data_nan['SOG']>=5].groupby('TrackNumber').std()
    if sog_lt5_median.shape[0] == 0:
        sog_lt5_median.loc[0,:] = (len(data_nan.columns)-1)*[np.nan]
    if sog_gte5_median.shape[0] == 0:
        sog_gte5_median.loc[0,:] = (len(data_nan.columns)-1)*[np.nan]
    if sog_lt5_std.shape[0] == 0:
        sog_lt5_std.loc[0,:] = (len(data_nan.columns)-1)*[np.nan]
    if sog_gte5_std.shape[0] == 0:
        sog_gte5_std.loc[0,:] = (len(data_nan.columns)-1)*[np.nan]
 
    data_nan = data_nan.groupby('TrackNumber')
    data_mean = data_nan.mean()
    data_median = data_nan.median()
    data_q25 = data_nan.quantile(q=0.25)
    data_q75 = data_nan.quantile(q=0.75)
    data_std = data_nan.std()
    data_min = data_nan.min()
    data_max = data_nan.max()

    df = pd.concat([
        pd.DataFrame(data_mean.values),
        pd.DataFrame(data_median.values),
        pd.DataFrame(data_q25.values),
        pd.DataFrame(data_q75.values),
        pd.DataFrame(data_std.values),
        pd.DataFrame(data_min.values),
        pd.DataFrame(data_max.values),
        pd.DataFrame(sog_lt5_median.values),
        pd.DataFrame(sog_gte5_median.values),
        pd.DataFrame(sog_lt5_std.values),
        pd.DataFrame(sog_gte5_std.values),
        pd.DataFrame([track_number]),
    ], axis=1, ignore_index=True)

    cols =  ["%s_%s" % (op, col.title().replace(' ','_')) for op in ['mean', 'median', 'q25', 'q75', 'std', 'min', 'max', 'lt5_median', 'gte5_median', 'lt5_std', 'gte5_std'] for col in data_mean.columns]
    cols.append('TrackNumber')
    df.columns = cols
    return df


def prepare(filename):
    if not os.path.exists(filename):
        data_all = [] 

        for i, name in enumerate(sorted(glob.glob(os.path.join('VesselTracks','1*.csv')))):
            df = read(name)
            if df.shape[0] < 2:
                continue
            track_number = int(name.replace('VesselTracks/', '').replace('.csv',''))
            df = clean(df, track_number)
            if df.shape[0] > 0:
                df = calc_features(df, track_number)
                if df.shape[0] > 0:    
                    data_all.append(df)    
            
        data_all = pd.DataFrame().append(data_all, ignore_index=True)
        print data_all  
        data_all.to_csv('combined.csv', index=False, float_format='%.3f')
