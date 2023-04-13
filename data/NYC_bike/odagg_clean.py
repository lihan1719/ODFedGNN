# 将共享单车数据进行OD栅格集记,并清洗天气数据

import pandas as pd
from utils.Grid_split_Vis import *

# OD栅格集记
region_name = 'Manhattan_v2'
region = 'raw_bike_data/' + region_name + '.json'
grid_size = 300
data_file = 'data/NYC_bike/raw_bike_data/NYC_2019.h5'

# 读取数据
_, params = gird_vis(grid_size=1000, region_file=region)
data = pd.read_hdf(data_file, key='bike_data')

bike_data = tbd.clean_outofshape(data, region, col=['slon', 'slat'])
bike_data = tbd.clean_outofshape(bike_data, region, col=['elon', 'elat'])
print(len(bike_data) / len(data))
#剔除过长与过短的共享单车出行
bike_data['distance'] = tbd.getdistance(bike_data['slon'], bike_data['slat'],
                                        bike_data['elon'], bike_data['elat'])
#清洗骑行数据，删除过长与过短的出行（小于100，大于10000）
bike_data = bike_data[(bike_data['distance'] > 100)
                      & (bike_data['distance'] < 10000)]
bike_data['stime'] = pd.to_datetime(bike_data['stime'])
bike_data.set_index('stime', inplace=True)
if args == None:
    return bike_data
return bike_data, weather_data
od_grid = tbd.odagg_grid(bike_data.copy(deep=True), params, arrow=True)

clean_grid(od_gdf, grid=True)
