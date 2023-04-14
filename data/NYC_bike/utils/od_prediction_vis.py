# 可视化OD预测结果

import pandas as pd
import numpy as np
from utils.Mapping_Grid import *
import geopandas as gpd
from utils import *


#----------------------------------------------------------------------------------------------
# 实现地理可视化
def od2map(od_metric, OD_map, mapping_grid):
    od_metric = pd.DataFrame(od_metric,
                             index=mapping_grid.index,
                             columns=mapping_grid.index)
    od_metric.reset_index(inplace=True)
    od_metric = pd.melt(od_metric,
                        id_vars=['index'],
                        value_vars=list(od_metric.columns),
                        var_name='e_index',
                        value_name='demand')
    od_metric.columns = ['s_index', 'e_index', 'demand']
    od_metric = pd.merge(OD_map,
                         od_metric,
                         on=['s_index', 'e_index'],
                         how='inner')
    odmapped = gpd.GeoDataFrame(od_metric.loc[:, ['demand', 'geometry']])
    # od_metric.to_csv('data/NYC_bike/raw_bike_data/od_prediction.csv',
    #                         index=False)
    # odmapped.to_file('data/NYC_bike/raw_bike_data/od_prediction.json',
    #                  driver='GeoJSON')
    return odmapped


if __name__ == '__main__':
    # for debug
    t_s = '2019-07'
    t_e = '2019-07'
    Bike_file = 'data/NYC_bike/raw_bike_data/NYC_Bike.csv'
    region_file = 'data/NYC_bike/raw_bike_data/Manhattan_v2.json'
    grid_size = 300
    time_granularity = '15T'
    predict_file = './output/2019-07_2019-07_grid_300_15T_od_prediction.npy'
    true_file = 'data/NYC_bike/processed_bike_data/NYC_BOD_Test_2019-07-2019-07_300m_60T.npz'

    # 读取数据
    od_prediction = np.load(predict_file)
    od_true = np.load(true_file)['y']
    # 取整
    od_true = np.round(np.clip(od_true, a_min=0, a_max=None)).astype(int)
    od_prediction = np.round(np.clip(od_prediction, a_min=0,
                                     a_max=None)).astype(int)
    bike_data = pd.read_csv(Bike_file)
    mapping_grid = get_map(bike_data, region_file, grid_size, t_s, t_e)
    _, params = gird_vis(grid_size=grid_size, region_file=region_file)
    od_grid = tbd.odagg_grid(bike_data.copy(deep=True), params, arrow=True)
    OD_map = clean_grid(od_grid, grid=False).drop(['count'], axis=1)
    _ = od2map(od_true[0, 0, :, :], OD_map, mapping_grid)
    print(_)