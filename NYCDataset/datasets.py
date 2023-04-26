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
                    I=
                    x_attr=torch.from_numpy(x_attr).float(),
                    y_attr=torch.from_numpy(y_attr).float(),
                    edge_index=edge_index,
                    edge_attr=edge_attr)
        loaded_data[name] = data
    return loaded_data