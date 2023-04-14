# 该python文件用于将收集的原始数据格式转化为模型训练格式,h5-->numpy

from utils import *
import pandas as pd
import transbigdata as tbd
from utils.Mapping_Grid import *
import os

# 初始化
Bike_file = 'data/NYC_bike/raw_bike_data/NYC_Bike.csv'
Weather_file = 'data/NYC_bike/raw_bike_data/NYC_Weather.csv'
t_s = '2019-07'
t_e = '2019-07'
drop_col = [
    'Dew Point', 'Humidity', 'Wind', 'Wind Gust', 'Pressure', 'Precip.'
]
region_file = 'data/NYC_bike/raw_bike_data/Manhattan_v2.json'
grid_size = 300
out_dir = 'data/NYC_bike/processed_bike_data/'
train_date = '2019-07-20'
val_date = '2019-07-24'
time_granularity = '60T'
nyc_bod = out_dir + 'NYC_BOD_{0}-{1}_Grid{2}m.npy'.format(t_s, t_e, grid_size)
geo_adj = out_dir + 'Geo_adj_Grid{0}m.npy'.format(grid_size)
nyc_w = out_dir + 'NYC_Weather_{0}_{1}_{2}.csv'.format(t_s, t_e,
                                                       time_granularity)
#-----------------------------------------------------------------------------

if not (os.path.isfile(nyc_bod) and os.path.isfile(geo_adj)
        and os.path.isfile(nyc_w)):
    # 读取数据
    bike_data = pd.read_csv(Bike_file)
    weather_data = pd.read_csv(Weather_file)
    weather_data['Time'] = pd.to_datetime(weather_data['Time'])
    weather_data.set_index('Time', inplace=True)
    weather_data = weather_data[t_s:t_e]

    #-----------------------------------------------------------------------------
    # 天气数据格式调整
    weather_data = unit_unify(raw=weather_data)  #统一格式
    weather_data = time_agg(weather_data, frep=time_granularity)  # 时间聚合
    weather_data = tune_col(weather_data, drop_col)  # 调整col
    weather_data.to_csv(nyc_w)
    #-----------------------------------------------------------------------------

    # Bike OD数据转换
    area = gpd.read_file(region_file)
    _, params = gird_vis(grid_size=grid_size, region_file=region_file)
    bike_data = rm_out(bike_data, area)
    bike_data['stime'] = pd.to_datetime(bike_data['stime'])
    bike_data.set_index('stime', inplace=True)
    bike_data = bike_data[t_s:t_e]
    od_grid = tbd.odagg_grid(bike_data.copy(deep=True), params,
                             arrow=True)  # OD集记
    mapping_grid = clean_grid(od_grid, grid=True)
    del od_grid
    Bike_od = Bike2OD(bike_data, mapping_grid, params, time_granularity)
    adj = get_adj(mapping_grid, grid_size)
    np.save(nyc_bod, Bike_od)
    np.save(geo_adj, adj)
else:
    Bike_od = np.load(nyc_bod)
    adj = np.load(geo_adj)
    weather_data = pd.read_csv(nyc_w)
#-----------------------------------------------------------------------------
# 数据集划分
train_file = out_dir + 'NYC_BOD_Train_{}-{}_{}m_{}.npz'.format(
    t_s, t_e, grid_size, time_granularity)
val_file = out_dir + 'NYC_BOD_Val_{}-{}_{}m_{}.npz'.format(
    t_s, t_e, grid_size, time_granularity)
test_file = out_dir + 'NYC_BOD_Test_{}-{}_{}m_{}.npz'.format(
    t_s, t_e, grid_size, time_granularity)
if not (os.path.isfile(train_file) and os.path.isfile(val_file)
        and os.path.isfile(test_file)):
    weather_data['Time'] = pd.to_datetime(weather_data['Time'])
    weather_data.set_index('Time', inplace=True)
    weather_data = weather_data[t_s:t_e]
    train_size = len(weather_data.loc[weather_data.index < train_date])
    val_size = len(weather_data.loc[(weather_data.index >= train_date)
                                    & (weather_data.index < val_date)])
    # test_size = len(weather_data.loc[weather_data.index >= val_date])
    train_data, val_data, test_data = split_data(Bike_od, train_size, val_size)
    np.savez(train_file, x=train_data['x'], y=train_data['y'])
    np.savez(val_file, x=val_data['x'], y=val_data['y'])
    np.savez(test_file, x=test_data['x'], y=test_data['y'])
print('处理完毕')