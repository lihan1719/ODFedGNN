# 处理NYC-bike数据集，包括OD矩阵转换，合并天气数据，邻接矩阵提取等等

# 导入包
from argparse import ArgumentParser
from functools import partial
import geopandas as gpd
import torch
from scipy.spatial.distance import cdist
from haversine import haversine, Unit
import os
import multiprocessing as mp
import transbigdata as tbd
import pandas as pd
import numpy as np
from .TO_OD import *


# cpu并行OD矩阵转化
def parallel_convert(chunk, grid_rec, params):
    oddata = tbd.odagg_grid(chunk, params, arrow=True)
    od = process_hourly_data(
        oddata, grid_rec)
    return od


# OD矩阵
def Bike2OD(bike_data, grid_rec, params, time_granularity):
    # 将数据分块，做并行处理
    chunks = [group for _, group in bike_data.resample(rule=time_granularity)]
    partial_func = partial(parallel_convert, grid_rec=grid_rec, params=params)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(
            partial_func,
            chunks,
        )
        pool.close()  # Close the pool to prevent new tasks
        pool.join()  # Wait for all processes to finish
        bike_od = np.stack(results)
    print("bike_od的shape:{}".format(bike_od.shape))
    return bike_od


# Geo 邻居矩阵
# def get_adj(mapping_grid, accuracy):
#     # 获取邻接矩阵

#     distance_matrix = cdist(mapping_grid[['SHBLON', 'SHBLAT']],
#                             mapping_grid[['SHBLON', 'SHBLAT']],
#                             metric=haversine)
#     distance_threshold = accuracy / 1000 + 0.1  # km
#     adjacency_matrix = np.where(distance_matrix <= distance_threshold, 1, 0)
#     adj = pd.DataFrame(adjacency_matrix,
#                        index=mapping_grid['s_index'],
#                        columns=mapping_grid['s_index'])
#     adj = adj.sort_index(axis=0).sort_index(axis=1)
#     print('邻接矩阵的shape:{}'.format(adj.shape))
#     return adj


def haversine_wrapper(u, v):
    return haversine(u, v, unit=Unit.KILOMETERS)


def get_adj(grid_rec, accuracy):
    # 获取邻接矩阵
    grid_count = len(grid_rec)
    adj_matrix = np.zeros((grid_count, grid_count))
    # 计算栅格中心的经纬度坐标
    centroids = mapping_grid.geometry.centroid
    lon_lat = centroids.apply(lambda p: (p.x, p.y)).tolist()

    # 计算距离矩阵
    distance_matrix = cdist(lon_lat, lon_lat, metric=haversine_wrapper)

    # 设置距离阈值
    distance_threshold = 0.1  # km

    # 根据距离阈值创建邻接矩阵
    adjacency_matrix = np.where(distance_matrix <= distance_threshold, 1, 0)

    # 将邻接矩阵转换为pandas DataFrame
    adj = pd.DataFrame(adjacency_matrix,
                       index=mapping_grid['s_index'],
                       columns=mapping_grid['s_index'])

    # 按照索引排序
    adj = adj.sort_index(axis=0).sort_index(axis=1)

    print('邻接矩阵的shape:{}'.format(adj.shape))
    return adj


def split_data(data, train_size, val_size, step):
    print('正在划分数据集')
    X_list = list(range(step, 0, -1))
    Y_list = list(range(step))

    X_ = np.array([[data[i - j] for j in X_list]
                   for i in range(max(X_list), data.shape[0] - max(Y_list))],
                  dtype=np.float32)
    Y_ = np.array([[data[i + j] for j in Y_list]
                   for i in range(max(X_list), data.shape[0] - max(Y_list))],
                  dtype=np.float32)
    del data
    if train_size == None:
        train_ratio, test_ratio, val_ratio = 0.7, 0.2, 0.1
        train_size, val_size = int(train_ratio * X_.shape[0]), int(val_ratio *
                                                                   X_.shape[0])
    train_X, val_X, test_X = X_[:train_size], X_[train_size:train_size +
                                                 val_size], X_[train_size +
                                                               val_size:]
    del X_
    train_Y, val_Y, test_Y = Y_[:train_size], Y_[train_size:train_size +
                                                 val_size], Y_[train_size +
                                                               val_size:]
    del Y_
    train_X, val_X, test_X = torch.from_numpy(train_X), torch.from_numpy(
        val_X), torch.from_numpy(test_X)
    train_Y, val_Y, test_Y = torch.from_numpy(train_Y), torch.from_numpy(
        val_Y), torch.from_numpy(test_Y)
    print('trainx_shape:{},valx_shape:{},testx_shape:{}'.format(
        train_X.shape, val_X.shape, test_X.shape))
    print('trainy_shape:{},valy_shape:{},testy_shape:{}'.format(
        train_Y.shape, val_Y.shape, test_Y.shape))
    train = {}
    val = {}
    test = {}
    train.update({'x': train_X, 'y': train_Y})
    del train_X, train_Y
    val.update({'x': val_X, 'y': val_Y})
    del val_X, val_Y
    test.update({'x': test_X, 'y': test_Y})
    del test_X, test_Y
    return train, val, test


# 剔除研究区域外、行驶距离过长过短的共享单车记录
def rm_out(data, geo_bounds):
    bike_data = tbd.clean_outofshape(data,
                                     geo_bounds,
                                     col=['slon', 'slat'],
                                     accuracy=100)
    bike_data = tbd.clean_outofshape(bike_data,
                                     geo_bounds,
                                     col=['elon', 'elat'],
                                     accuracy=100)
    print(len(bike_data) / len(data))
    #剔除过长与过短的共享单车出行
    bike_data['distance'] = tbd.getdistance(bike_data['slon'],
                                            bike_data['slat'],
                                            bike_data['elon'],
                                            bike_data['elat'])
    #清洗骑行数据，删除过长与过短的出行（小于100，大于10000）
    bike_data = bike_data[(bike_data['distance'] > 100)
                          & (bike_data['distance'] < 10000)]
    return bike_data
