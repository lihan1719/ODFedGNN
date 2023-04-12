# 数据读取，并删除研究区域外的数据

import pandas as pd
import transbigdata as tbd
import geopandas as gpd


# 读取研究区域及栅格参数
def get_bounds(region_file, accuracy):
    #研究区域
    geo_bounds = gpd.read_file(region_file)
    #数据栅格化
    params = tbd.area_to_params(geo_bounds, accuracy=accuracy, method='rect')
    params['theta'] = 30
    return geo_bounds, params


# 读取数据(弃)
# def get_data(
#     geo_bounds,
#     nyc_data=None,
#     time_start='',
#     time_end='',
#     time_granularity='',
#     args=None,
# ):
#     if args == None:
#         data = pd.read_hdf(nyc_data, key='bike_data')
#     else:
#         weather_data = pd.read_hdf(args.nyc_data,
#                                    key='weather_data_{0}_{1}_{2}'.format(
#                                        args.time_start, args.time_end,
#                                        args.time_granularity))
#         data = pd.read_hdf(args.nyc_data, key='bike_data')
#     #数据预处理
#     #剔除研究范围外的数据点
#     bike_data = tbd.clean_outofshape(data, geo_bounds, col=['slon', 'slat'])
#     bike_data = tbd.clean_outofshape(bike_data,
#                                      geo_bounds,
#                                      col=['elon', 'elat'])
#     print(len(bike_data) / len(data))
#     #剔除过长与过短的共享单车出行
#     bike_data['distance'] = tbd.getdistance(bike_data['slon'],
#                                             bike_data['slat'],
#                                             bike_data['elon'],
#                                             bike_data['elat'])
#     #清洗骑行数据，删除过长与过短的出行（小于100，大于10000）
#     bike_data = bike_data[(bike_data['distance'] > 100)
#                           & (bike_data['distance'] < 10000)]
#     bike_data['stime'] = pd.to_datetime(bike_data['stime'])
#     bike_data.set_index('stime', inplace=True)
#     if args == None:
#         return bike_data
#     return bike_data, weather_data
