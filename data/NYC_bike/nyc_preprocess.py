# 处理NYC-bike数据集，包括OD矩阵转换，合并天气数据，邻接矩阵提取等等

# 导入包
from argparse import ArgumentParser
from functools import partial
from utils.Data_Loader import *
from utils.Mapping_Grid import *
import torch
from scipy.spatial.distance import cdist
from haversine import haversine
import os


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


def main(args):

    #划分数据集之后保存数据位置
    train_file = args.out_dir + 'bike_od_train.npz'
    val_file = args.out_dir + 'bike_od_val.npz'
    test_file = args.out_dir + 'bike_od_test.npz'
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    out_file = args.out_dir + args.out_name
    # 生成区域栅格
    # 读取数据
    geo_bounds, params = get_bounds(args.region_file, accuracy=args.accuracy)
    data, W_data = get_data(args=args, geo_bounds=geo_bounds)
    Weather_data = W_data[args.time_start:args.time_end].copy(deep=True).values
    if not os.path.isfile(out_file):
        bike_data = data[args.time_start:args.time_end].copy(deep=True)

        od_gdf = tbd.odagg_grid(bike_data.copy(deep=True), params, arrow=True)
        mapping_grid = clean_grid(od_gdf, grid=True)
        # 生成包含每个i栅格的OD空表
        od_final = pd.DataFrame(0,
                                index=mapping_grid['s_index'],
                                columns=mapping_grid['s_index'])
        od_final.columns.name = 'e_index'
        od_final.index.name = 's_index'
        od_final.index = od_final.index.astype('int')
        od_final.columns = od_final.columns.astype('int')
        # 将数据分块，做并行处理
        chunks = [
            group
            for _, group in bike_data.resample(rule=args.time_granularity)
        ]
        partial_func = partial(parallel_convert,
                               params=params,
                               od_final=od_final)
        with mp.Pool(processes=args.num_processes) as pool:
            results = pool.map(
                partial_func,
                chunks,
            )
            pool.close()  # Close the pool to prevent new tasks
            pool.join()  # Wait for all processes to finish
            bike_od = np.stack(results)
        print("bike_od的shape:{}".format(bike_od.shape))

        # 获取邻接矩阵

        distance_matrix = cdist(mapping_grid[['SHBLON', 'SHBLAT']],
                                mapping_grid[['SHBLON', 'SHBLAT']],
                                metric=haversine)
        distance_threshold = args.accuracy / 1000 + 0.1  # km
        adjacency_matrix = np.where(distance_matrix <= distance_threshold, 1,
                                    0)
        adj = pd.DataFrame(adjacency_matrix,
                           index=mapping_grid['s_index'],
                           columns=mapping_grid['s_index'])
        adj = adj.sort_index(axis=0).sort_index(axis=1)
        print('邻接矩阵的shape:{}'.format(adj.shape))

        # 合并天气数据
        Weather_data = np.repeat(Weather_data[:, np.newaxis, :],
                                 repeats=bike_od.shape[1],
                                 axis=1)
        Weather_data = Weather_data.reshape(
            (bike_od.shape[0], bike_od.shape[1], Weather_data.shape[-1]))
        data = np.concatenate((bike_od, Weather_data), axis=-1)
        print('模型训练的数据集shape:{}'.format(data.shape))

        # 写入文件
        print('合并成功，写入文件')
        np.savez(out_file, adj=adj, bike_od=bike_od)
    else:
        bike_od = np.load(out_file)['bike_od']
        Weather_data = np.repeat(Weather_data[:, np.newaxis, :],
                                 repeats=bike_od.shape[1],
                                 axis=1)
        Weather_data = Weather_data.reshape(
            (bike_od.shape[0], bike_od.shape[1], Weather_data.shape[-1]))
        data = np.concatenate((bike_od, Weather_data), axis=-1)
    del W_data
    del Weather_data
    del bike_od
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
    train_ratio, test_ratio, val_ratio = 0.7, 0.2, 0.1
    train_size, val_size = int(train_ratio * X_.shape[0]), int(val_ratio *
                                                               X_.shape[0])
    test_size = X_.shape[0] - train_size - val_size

    train_X, val_X, test_X = X_[:train_size], X_[train_size:train_size +
                                                 val_size], X_[train_size +
                                                               val_size:]
    train_Y, val_Y, test_Y = Y_[:train_size], Y_[train_size:train_size +
                                                 val_size], Y_[train_size +
                                                               val_size:]

    train_X, val_X, test_X = torch.from_numpy(train_X), torch.from_numpy(
        val_X), torch.from_numpy(test_X)
    train_Y, val_Y, test_Y = torch.from_numpy(train_Y), torch.from_numpy(
        val_Y), torch.from_numpy(test_Y)
    print('trainx_shape:{},valx_shape:{},testx_shape:{}'.format(
        train_X.shape, val_X.shape, test_X.shape))
    print('trainy_shape:{},valy_shape:{},testy_shape:{}'.format(
        train_Y.shape, val_Y.shape, test_Y.shape))
    # 保存文件
    np.savez(train_file, x=train_X, y=train_Y)
    np.savez(val_file, x=val_X, y=val_Y)
    np.savez(test_file, x=test_X, y=test_Y)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--time_start', type=str, default='2019-07')
    parser.add_argument('--time_end', type=str, default='2019-07')
    parser.add_argument('--time_granularity', type=str, default='15T')
    parser.add_argument('--out_dir',
                        type=str,
                        default='data/NYC_bike/processed_bike_data/')
    parser.add_argument('--out_name',
                        type=str,
                        default='2019-07_2019-07_grid_300_15T.npz')
    parser.add_argument('--nyc_data',
                        type=str,
                        default='data/NYC_bike/raw_bike_data/nyc_data2019.h5')
    parser.add_argument('--accuracy', type=int, default=300)
    parser.add_argument('--region_file',
                        type=str,
                        default='data/NYC_bike/raw_bike_data/Manhattan.json')
    parser.add_argument('--num_processes', type=int, default=mp.cpu_count())
    args = parser.parse_args()
    main(args)