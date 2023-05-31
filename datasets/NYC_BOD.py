import os
import pickle
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


# Z_score归一化
class Z_score_scaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class MaxMin_scaler:
    def __init__(self, max, min):
        self.max = max
        self.min = min

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min


#
def get_feats(data: np.array, params):
    x, y = [], []
    for i in range(params.obs_len, data.shape[0] - params.pred_len):
        x.append(data[i - params.obs_len:i])
        y.append(data[i:i + params.pred_len])
    return x, y


def split2len(data_len: int, params):
    mode_len = dict()
    mode_len['train'] = int(params.split_ratio[0] / sum(params.split_ratio) *
                            data_len)
    mode_len['validate'] = int(params.split_ratio[1] /
                               sum(params.split_ratio) * data_len)
    mode_len['test'] = data_len - mode_len['train'] - mode_len['validate']
    return mode_len


def split_data(X, data_dict, mode_len):
    data_dict.update({
        'train':
        X[:mode_len['train']],
        'val':
        X[mode_len['train']:mode_len['train'] + mode_len['validate']],
        'test':
        X[mode_len['train'] + mode_len['validate']:]
    })


#读取NYC数据集
def load_nyc_data(path, params):
    # 邻接矩阵
    adj = np.load(path + 'Geo_adj_Grid300m.npy')
    adj_mx_ts = torch.from_numpy(adj).float()
    edge_index, edge_attr = dense_to_sparse(adj_mx_ts)
    # NYC-BOD数据、外部数据
    raw = np.load(path + 'NYC_BOD_2019-07-2019-07_Grid300m_60T.npy')
    FEATURE_START, FEATURE_END = 0, raw.shape[1]
    ATTR_START = raw.shape[1]
    # 根据预测步长生成数据集
    X, Y = get_feats(raw, params)
    X = np.asarray(X)
    Y = np.asarray(Y)
    # Z-score归一化
    feature_scaler = Z_score_scaler(mean=np.mean(X[...,
                                                   FEATURE_START:FEATURE_END]),
                                    std=np.std(X[...,
                                                 FEATURE_START:FEATURE_END]))
    attr_scaler = Z_score_scaler(mean=np.mean(X[..., ATTR_START:]),
                                 std=np.std(X[..., ATTR_START:]))
    # 最大最小归一化
    # feature_scaler = MaxMin_scaler(max=np.max(X[...,
    #                                             FEATURE_START:FEATURE_END]),
    #                                min=np.min(X[...,
    #                                             FEATURE_START:FEATURE_END]))
    # attr_scaler = MaxMin_scaler(max=np.max(X[..., ATTR_START:]),
    #                             min=np.min(X[..., ATTR_START:]))
    # 划分训练集
    mode_len = split2len(X.shape[0], params)
    print(mode_len)
    data_x = {}
    data_y = {}
    split_data(X, data_x, mode_len)
    del X
    split_data(Y, data_y, mode_len)
    del Y

    loaded_data = {}
    for name in ['train', 'val', 'test']:
        data = data_x.pop(name)
        x = data[..., FEATURE_START:FEATURE_END]
        x_attr = data[..., ATTR_START:]
        data = data_y.pop(name)
        y = data[..., FEATURE_START:FEATURE_END]
        y_attr = data[..., ATTR_START:]
        edge_index, edge_attr = edge_index, edge_attr
        data_dict = {}
        print("{0}_data.shape:{1}".format(name, x.shape))
        print("{0}_data_attr.shape:{1}".format(name, x_attr.shape))
        x = feature_scaler.transform(x)
        y = feature_scaler.transform(y)
        x_attr = attr_scaler.transform(x_attr)
        y_attr = attr_scaler.transform(y_attr)
        loaded_data.update(feature_scaler=feature_scaler)
        data_dict.update(
            x=torch.from_numpy(x).float(),
            y=torch.from_numpy(y).float(),
            x_attr=torch.from_numpy(x_attr).float(),
            y_attr=torch.from_numpy(y_attr).float(),
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        loaded_data[name] = data_dict
    return loaded_data
