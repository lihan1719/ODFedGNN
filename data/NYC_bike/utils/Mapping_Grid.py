# 存储栅格映射
import pandas as pd
import numpy as np
import transbigdata as tbd
import multiprocessing as mp
import geopandas as gpd
import pandas as pd
import os
from utils.BOD_preprocess import rm_out
from utils.Mapping_Grid import *
from utils import *


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


# 得到mapping_grid
def get_map(bike_data, region_file, grid_size, t_s, t_e):
    _, params = gird_vis(grid_size=grid_size, region_file=region_file)
    area = gpd.read_file(region_file)
    bike_data = rm_out(bike_data, area)
    bike_data['stime'] = pd.to_datetime(bike_data['stime'])
    bike_data.set_index('stime', inplace=True)
    bike_data = bike_data[t_s:t_e]
    od_grid = tbd.odagg_grid(bike_data.copy(deep=True), params,
                             arrow=True)  # OD集记
    mapping_grid = clean_grid(od_grid, grid=True)
    return mapping_grid
