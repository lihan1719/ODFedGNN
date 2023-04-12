# 存储栅格映射
import pandas as pd
import numpy as np
import transbigdata as tbd
import multiprocessing as mp
import geopandas as gpd
import pandas as pd
import os
from utils.Data_Loader import *


def clean_grid(od_gdf, grid=False):
    # 将栅格集记做OD标记
    od_gdf['SLONCOL'] = od_gdf['SLONCOL'].astype('str')
    od_gdf['SLATCOL'] = od_gdf['SLATCOL'].astype('str')
    od_gdf['sid'] = od_gdf['SLONCOL'] + '-' + od_gdf['SLATCOL']
    od_gdf['ELONCOL'] = od_gdf['ELONCOL'].astype('str')
    od_gdf['ELATCOL'] = od_gdf['ELATCOL'].astype('str')
    od_gdf['eid'] = od_gdf['ELONCOL'] + '-' + od_gdf['ELATCOL']
    if grid == False:
        od_data = od_gdf.drop(['SLONCOL', 'SLATCOL', 'ELONCOL', 'ELATCOL'],
                              axis=1).copy(deep=True)
    else:
        od_data = od_gdf.drop(
            ['geometry', 'SLONCOL', 'SLATCOL', 'ELONCOL', 'ELATCOL'],
            axis=1).copy(deep=True)
    od_data['sid'] = od_data['sid'].astype('category')
    od_data['eid'] = od_data['eid'].astype('category')
    od_data['s_index'] = od_data['sid'].cat.codes
    od_data['e_index'] = od_data['eid'].cat.codes
    od_data.drop(['sid', 'eid'], inplace=True, axis=1)
    if grid == True:
        od_data = od_data.drop_duplicates(subset=['s_index'], keep='first')
    od_data['s_index'] = od_data['s_index'].astype('int')
    od_data['e_index'] = od_data['e_index'].astype('int')
    od_data.reset_index(inplace=True)
    od_data.drop(['index'], axis=1, inplace=True)
    return od_data


if __name__ == '__main__':
    # 需要修改的参数
    time_start = '201907'  #时间格式，必须全为数字
    time_end = '201909'
    accuracy = 500
    region_file = 'data/NYC_bike/raw_bike_data/Manhattan_half.json'

    h5_file_name = 'nyc_bike_data_07-09_experiment1.h5'
    # h5_file_name = 'nyc_bike_data' + '_' + time_start + '-' + time_end + '.h5'  # h5 文件名称
    h5_path = 'data/NYC_bike/raw_bike_data/'  # 目标路径
    h5_file = h5_path + h5_file_name

    geo_bounds, params = get_bounds(region_file, accuracy=accuracy)
    bike_data = get_data(h5_file, geo_bounds)
    od_gdf = tbd.odagg_grid(bike_data.copy(deep=True), params, arrow=True)
    od_data = clean_grid(od_gdf, grid=True)
    # 保存到文件
    print('写入文件...')
    with pd.HDFStore(h5_file, mode='a') as store:
        store.put('mapping_grid_500', od_data, data_columns=True)
