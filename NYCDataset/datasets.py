import datetime
import os
import pickle
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


def load_nyc_data(path, data_name):
    #读取自制的NYC数据集
    adj = np.load(path + 'Geo_adj_Grid300m.npy')
    adj_mx_ts = torch.from_numpy(adj).float()
    geo_adj = adj_mx_ts
    edge_index, edge_attr = dense_to_sparse(geo_adj)

    raw_data = {}
    #划分数据集载入
    for name in ['train', 'val', 'test']:
        file = 'NYC_BOD_' + name + '_' + data_name + '.npz'
        raw_data[name] = np.load(os.path.join(path, file))
    FEATURE_START, FEATURE_END = 0, raw_data['train']['x'].shape[2]
    ATTR_START = raw_data['train']['x'].shape[2]
    loaded_data = {}
    for name in ['train', 'val', 'test']:
        nyc_data = raw_data[name]['x'][..., FEATURE_START:FEATURE_END]
        x = (nyc_data - np.min(nyc_data)) / (np.max(nyc_data) -
                                             np.min(nyc_data))
        nyc_data = raw_data[name]['y'][..., FEATURE_START:FEATURE_END]
        y = (nyc_data - np.min(nyc_data)) / (np.max(nyc_data) -
                                             np.min(nyc_data))
        del nyc_data
        print('{0}的x的shape为{1}'.format(name, x.shape))
        nyc_attr = raw_data[name]['x'][..., ATTR_START:]
        x_attr = (nyc_attr - np.min(nyc_attr)) / (np.max(nyc_attr) -
                                                  np.min(nyc_attr))
        nyc_attr = raw_data[name]['y'][..., ATTR_START:]
        y_attr = (nyc_attr - np.min(nyc_attr)) / (np.max(nyc_attr) -
                                                  np.min(nyc_attr))
        del nyc_attr
        data = {}
        data.update(x=torch.from_numpy(x).float(),
                    y=torch.from_numpy(y).float(),
                    x_attr=torch.from_numpy(x_attr).float(),
                    y_attr=torch.from_numpy(y_attr).float(),
                    edge_index=edge_index,
                    edge_attr=edge_attr)
        loaded_data[name] = data
    return loaded_data


def get_day_index(date):
    # 根据日期返回时序OD矩阵索引
    date = datetime.strptime(date, '%Y-%m-%d')
    base_date = datetime.strptime('2019-07-01', '%Y-%m-%d')
    index = (date - base_date).days * 24 - 1
    return index


# OD矩阵划分
def OD_split(data):
    day = data.shape[0] / 24
    M = data.reshape((int(day), 24, 535, 535))
    I = np.sum(M, axis=2)
    I_reshaped = I[:, :, np.newaxis, :]
    I_nonzero = np.where(I_reshaped != 0, I_reshaped, 1)
    S = M / I_nonzero
    return I, S


# 转换数据格式
def prepare_inputs_and_targets(I_data):

    week_before = I_data[:-7]
    prev_interval = np.roll(I_data[7:], shift=1, axis=1)
    # Set the first hour of each day to zero
    prev_interval[:, 0, :] = 0

    week_before = week_before.reshape(-1, 535)[:-1]
    prev_interval = prev_interval.reshape(-1, 535)[:-1]
    X_I = np.stack((week_before, prev_interval), axis=1)
    # y_I = I_data[7:].reshape(-1, 535)[1:]

    return X_I


# 划分数据集
def split_data(X, y, X_attr, Y_attr, S, train_days, val_days):
    train_len = train_days * 24
    val_len = val_days * 24

    S_train = S[:train_len]
    x_attr_train = X_attr[:train_len]
    y_attr_train = Y_attr[:train_len]
    X_train = X[:train_len]
    y_train = y[:train_len]

    S_val = S[train_len:train_len + val_len]
    X_val = X[train_len:train_len + val_len]
    y_val = y[train_len:train_len + val_len]
    x_attr_val = X_attr[train_len:train_len + val_len]
    y_attr_val = Y_attr[train_len:train_len + val_len]

    S_test = S[train_len + val_len:]
    X_test = X[train_len + val_len:]
    y_test = y[train_len + val_len:]
    x_attr_test = X_attr[train_len + val_len:]
    y_attr_test = Y_attr[train_len + val_len:]

    data = {
        'train': {
            'x': torch.from_numpy(X_train).float(),
            'y': torch.from_numpy(y_train).float(),
            'x_attr': torch.from_numpy(x_attr_train).float(),
            'y_attr': torch.from_numpy(y_attr_train).float(),
            'S': torch.from_numpy(S_train).float()
        },
        'val': {
            'x': torch.from_numpy(X_val).float(),
            'y': torch.from_numpy(y_val).float(),
            'x_attr': torch.from_numpy(x_attr_val).float(),
            'y_attr': torch.from_numpy(y_attr_val).float(),
            'S': torch.from_numpy(S_val).float()
        },
        'test': {
            'x': torch.from_numpy(X_test).float(),
            'y': torch.from_numpy(y_test).float(),
            'x_attr': torch.from_numpy(x_attr_test).float(),
            'y_attr': torch.from_numpy(y_attr_test).float(),
            'S': torch.from_numpy(S_test).float(),
        }
    }

    return data


def ODsplit_nyc_data(path, data_name):
    #读取自制的NYC数据集
    adj = np.load(path + 'Geo_adj_Grid300m.npy')  # O x D
    adj_mx_ts = torch.from_numpy(adj).float()
    edge_index, edge_attr = dense_to_sparse(adj_mx_ts)
    data = np.load(path +
                   'NYC_BOD_2019-07-2019-07_Grid300m_60T.npy')  # T x O x D
    bike_data = data[:, :, :-3].copy()
    # 天气数据
    weather_data = data[:, :, -3:].copy()
    del data
    # OD矩阵划分
    I, S = OD_split(bike_data)
    # 使用前一周的分离率作为预测的分离率
    S = S[:-7, ...]
    S = S.reshape(-1, 535, 535)[:-1]
    # 归一化
    od_max = np.max(I)
    od_min = np.min(I)
    I = (I - od_min) / (od_max - od_min)
    weather_data = (weather_data - np.min(weather_data)) / (
        np.max(weather_data) - np.min(weather_data))
    X_I = prepare_inputs_and_targets(I)
    x_attr, y_attr = weather_data[24 * 7:][:-1], weather_data[24 * 7:][1:]
    X_I = X_I[:, :, :, np.newaxis]
    x_attr = np.repeat(x_attr[:, np.newaxis, :, :], X_I.shape[1], axis=1)
    # 划分数据集
    Y = bike_data[24 * 7:-1]
    data_dict = split_data(X_I,
                           Y,
                           x_attr,
                           y_attr,
                           S,
                           train_days=20,
                           val_days=2)

    for name in ['train', 'val', 'test']:
        print('{0}的x的shape为{1}'.format(name, data_dict[name]['x'].shape))
        data_dict[name].update(edge_index=edge_index, edge_attr=edge_attr)
    data_dict.update(od_max=od_max, od_min=od_min)
    return data_dict