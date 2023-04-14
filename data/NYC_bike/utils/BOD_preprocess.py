# 处理NYC-bike数据集，包括OD矩阵转换，合并天气数据，邻接矩阵提取等等

# 导入包
from argparse import ArgumentParser
from functools import partial
from utils.Mapping_Grid import *
import torch
from scipy.spatial.distance import cdist
from haversine import haversine
import os
import multiprocessing as mp


# cpu并行OD矩阵转化
def parallel_convert(chunk, params=None, od_final=None):
    od_gdf = tbd.odagg_grid(chunk.copy(deep=True), params, arrow=True)
    #快速给栅格编号并进行合并
    od_data = clean_grid(od_gdf)
    od_now = od_data.pivot_table(index='s_index',
                                 columns='e_index',
                                 values=['count'],
                                 fill_value=0,
                                 aggfunc=np.sum)

    od_now = od_now.droplevel(level=0, axis=1)
    result = od_final.add(od_now).fillna(0)

    return result.values


# OD矩阵
def Bike2OD(bike_data, mapping_grid, params, time_granularity):
    od_final = pd.DataFrame(0,
                            index=mapping_grid['s_index'],
                            columns=mapping_grid['s_index'])
    od_final.columns.name = 'e_index'
    od_final.index.name = 's_index'
    od_final.index = od_final.index.astype('int')
    od_final.columns = od_final.columns.astype('int')
    # 将数据分块，做并行处理
    chunks = [group for _, group in bike_data.resample(rule=time_granularity)]
    partial_func = partial(parallel_convert, params=params, od_final=od_final)
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
def get_adj(mapping_grid, accuracy):
    # 获取邻接矩阵

    distance_matrix = cdist(mapping_grid[['SHBLON', 'SHBLAT']],
                            mapping_grid[['SHBLON', 'SHBLAT']],
                            metric=haversine)
    distance_threshold = accuracy / 1000 + 0.1  # km
    adjacency_matrix = np.where(distance_matrix <= distance_threshold, 1, 0)
    adj = pd.DataFrame(adjacency_matrix,
                       index=mapping_grid['s_index'],
                       columns=mapping_grid['s_index'])
    adj = adj.sort_index(axis=0).sort_index(axis=1)
    print('邻接矩阵的shape:{}'.format(adj.shape))
    return adj


def split_data(data, train_size, val_size):
    print('正在划分数据集')
    X_list = list(range(4, 0, -1))
    Y_list = list(range(4))

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
    bike_data = tbd.clean_outofshape(data, geo_bounds, col=['slon', 'slat'])
    bike_data = tbd.clean_outofshape(bike_data,
                                     geo_bounds,
                                     col=['elon', 'elat'])
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
