import os
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

class MinMaxScalar:
    def __init__(self, min, max):
        self._min = min
        self._max = max

    def __int__(self):
        pass

    def transform(self, data):
        return (data - self._min) / (self._max - self._min)

    def fit_transform(self, data):
        self._min = np.min(data)
        self._max = np.max(data)
        return (data - self._min) / (self._max - self._min)

    def inverse_transform(self, data):
        return (data * (self._max - self._min)) + self._min

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = (labels!=null_val)

    mask = mask.astype('float32')
    mask /= np.mean(mask)
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    #modified
    loss = np.abs((preds-labels)/labels)
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return np.mean(loss)


def mean_absolute_percentage_error(y_true, y_pred):
    '''
    caculate mape
    :param y_true:
    :param y_pred:
    :return: mape ∈ [0,+ꝏ]
    '''
    return masked_mape(y_true,y_pred)
    # return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def evaluate(pred, targets, scaler):
    '''
    evaluate model with rmse, mae and mape
    :param flow_pred: [n_samples, 2, hegiht, width]
    :param flow_targets: [n_samples, 2, hegiht, width]
    :param od_pred: [n_samples, 2*height*width, hegiht, width]
    :param od_targets: [n_samples, 2*height*width, hegiht, width]
    :return:
    '''
    metrics = defaultdict(dict)
    # inverse_transform
    pred = scaler.inverse_transform(pred)
    targets = scaler.inverse_transform(targets)

    pred = np.reshape(pred, (pred.shape[0], -1))
    targets = np.reshape(targets, (targets.shape[0], -1))

    rmse = np.sqrt(mean_squared_error(targets, pred))
    mae = mean_absolute_error(targets, pred)
    mape = mean_absolute_percentage_error(targets, pred)

    metrics['rmse'] = rmse
    metrics['mae'] = mae
    metrics['mape'] = mape
    return metrics


def save_model(path, **save_dict):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    torch.save(save_dict,path)


def normalize_adj_mats(adj_mats):
    mask = (adj_mats > 1e-3).float()
    adj_mats = torch.softmax(adj_mats, dim=1) * mask
    adj_mats = (1.0 / (adj_mats.sum(dim=1, keepdim=True) + 1e-8)) * adj_mats
    return adj_mats
