import pandas as pd
from sklearn.metrics import roc_auc_score


type_order = ['longliner', 'other', 'seiner', 'support', 'trawler']
type_weights = [0.2, 0.0, 0.3, 0.1, 0.4]

def get_individual_scores_raw_predictions(y_true, y_pred, y_tracks):
    ret_true = y_true.copy()
    ret_true['TrackNumber'] = y_tracks['TrackNumber']
    ret_true = ret_true.groupby('TrackNumber').mean()

    ret_pred = y_pred.copy()
    ret_pred['TrackNumber']= y_tracks['TrackNumber']
    ret_pred = ret_pred.groupby('TrackNumber').mean()

    return get_individual_scores(ret_true, ret_pred)


def get_score_raw_predictions(y_true, y_pred, y_tracks):
    ret_true = y_true.copy()
    ret_true['TrackNumber'] = y_tracks['TrackNumber']
    ret_true = ret_true.groupby('TrackNumber').mean()

    ret_pred = y_pred.copy()
    ret_pred['TrackNumber']= y_tracks['TrackNumber']
    ret_pred = ret_pred.groupby('TrackNumber').mean()

    return get_score(ret_true, ret_pred)


def get_individual_scores(y_true, y_pred):
    return roc_auc_score(y_true, y_pred, average=None)


def get_score(y_true, y_pred):
    scores = get_individual_scores(y_true, y_pred)
    weighted_average = (scores * type_weights).sum()

    return 1000 * 1000 * (2 * weighted_average - 1)
