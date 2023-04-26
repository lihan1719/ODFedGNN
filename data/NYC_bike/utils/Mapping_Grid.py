# 存储栅格映射
import pandas as pd
import numpy as np
import transbigdata as tbd
import multiprocessing as mp
import geopandas as gpd
import pandas as pd
import os


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
