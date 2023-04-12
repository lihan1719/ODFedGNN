# 可视化OD预测结果

import pandas as pd
import numpy as np
from utils.Data_Loader import *
from utils.Mapping_Grid import *
import geopandas as gpd

time_start = '2019-07'
time_end = '2019-07'
nyc_data = 'data/NYC_bike/raw_bike_data/nyc_data2019.h5'
region_file = 'data/NYC_bike/raw_bike_data/Manhattan.json'
accuracy = 300
time_granularity = '15T'
# 读取数据
# od_prediction = np.load(
#     'baseline/ODCRN-main/model/ODCRN/output/ODCRN_prediction.npy')
od_prediction = np.load(
    './output/2019-07_2019-07_grid_300_15T_od_prediction.npy')
od_true = np.load('./data/NYC_bike/processed_bike_data/bike_od_test.npz')['y'][
    ..., 0:od_prediction.shape[2]]
# 取整
od_true = np.round(np.clip(od_true, a_min=0, a_max=None)).astype(int)
od_prediction = np.round(np.clip(od_prediction, a_min=0,
                                 a_max=None)).astype(int)
geo_bounds, params = get_bounds(region_file, accuracy=accuracy)
data = get_data(geo_bounds,
                nyc_data=nyc_data,
                time_start=time_start,
                time_end=time_end,
                time_granularity=time_granularity)
bike_data = data[time_start:time_end].copy(deep=True)
od_gdf = tbd.odagg_grid(bike_data.copy(deep=True), params, arrow=True)
OD_map = clean_grid(od_gdf, grid=False).drop(['count'], axis=1)
mapping_grid = clean_grid(od_gdf, grid=True)
# for od in [od_prediction, od_true]:
#     od = pd.DataFrame(od[0, 0, :, :],
#                       index=mapping_grid.index,
#                       columns=mapping_grid.index)
#     #变为长格式
#     od.reset_index(inplace=True)
#     od = pd.melt(od,
#                  id_vars=['index'],
#                  value_vars=list(od.columns),
#                  var_name='e_index',
#                  value_name='demand')
#     od.columns = ['s_index', 'e_index', 'demand']
#     od = pd.merge(OD_map, od, on=['s_index', 'e_index'], how='inner')
#     grid_data = gpd.GeoDataFrame(od.loc[:, ['demand', 'geometry']])
#     for name in ['od_prediction', 'od_true']:
#         od.to_csv('data/NYC_bike/raw_bike_data/' + name + '.csv', index=False)
#         grid_data.to_file('data/NYC_bike/raw_bike_data/' + name + '.json',
#                           driver='GeoJSON')
# 处理od_prediction
od_prediction_df = pd.DataFrame(od_prediction[0, 0, :, :],
                                index=mapping_grid.index,
                                columns=mapping_grid.index)
od_prediction_df.reset_index(inplace=True)
od_prediction_df = pd.melt(od_prediction_df,
                           id_vars=['index'],
                           value_vars=list(od_prediction_df.columns),
                           var_name='e_index',
                           value_name='demand')
od_prediction_df.columns = ['s_index', 'e_index', 'demand']
od_prediction_df = pd.merge(OD_map,
                            od_prediction_df,
                            on=['s_index', 'e_index'],
                            how='inner')
od_prediction_gdf = gpd.GeoDataFrame(
    od_prediction_df.loc[:, ['demand', 'geometry']])
od_prediction_df.to_csv('data/NYC_bike/raw_bike_data/od_prediction.csv',
                        index=False)
od_prediction_gdf.to_file('data/NYC_bike/raw_bike_data/od_prediction.json',
                          driver='GeoJSON')

# 处理od_true
od_true_df = pd.DataFrame(od_true[0, 0, :, :],
                          index=mapping_grid.index,
                          columns=mapping_grid.index)
od_true_df.reset_index(inplace=True)
od_true_df = pd.melt(od_true_df,
                     id_vars=['index'],
                     value_vars=list(od_true_df.columns),
                     var_name='e_index',
                     value_name='demand')
od_true_df.columns = ['s_index', 'e_index', 'demand']
od_true_df = pd.merge(OD_map,
                      od_true_df,
                      on=['s_index', 'e_index'],
                      how='inner')
od_true_gdf = gpd.GeoDataFrame(od_true_df.loc[:, ['demand', 'geometry']])
od_true_df.to_csv('data/NYC_bike/raw_bike_data/od_true.csv', index=False)
od_true_gdf.to_file('data/NYC_bike/raw_bike_data/od_true.json',
                    driver='GeoJSON')
