from utils import *
import pandas as pd
import transbigdata as tbd
import os
import geopandas as gpd
from torch_geometric.utils import dense_to_sparse

Bike_file = 'data/NYC_bike/raw_bike_data/NYC_Bike.csv'
Weather_file = 'data/NYC_bike/raw_bike_data/NYC_Weather.csv'
t_s = '2019-07'
t_e = '2019-07'
drop_col = [
    'Dew Point', 'Humidity', 'Wind', 'Wind Gust', 'Pressure', 'Precip.'
]
region_file = 'data/NYC_bike/raw_bike_data/Manhattan.json'
grid_size = 300
step = 4
out_dir = 'data/NYC_bike/processed_bike_data/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
train_date = '2019-07-20'
val_date = '2019-07-24'
time_granularity = '60T'
nyc_bod = out_dir + 'NYC_BOD_{0}-{1}_Grid{2}m_{3}.npy'.format(
    t_s, t_e, grid_size, time_granularity)
geo_adj = out_dir + 'Geo_adj_Grid{0}m.npy'.format(grid_size)
nyc_w = out_dir + 'NYC_Weather_{0}_{1}_{2}.csv'.format(t_s, t_e,
                                                       time_granularity)

weather_data = pd.read_csv(Weather_file)
weather_data['Time'] = pd.to_datetime(weather_data['Time'])
weather_data.set_index('Time', inplace=True)
weather_data = weather_data[t_s:t_e]
weather_data = unit_unify(raw=weather_data)
weather_data = weather_data.resample('H').first().fillna(method='ffill')
weather_data = tune_col(weather_data, drop_col, trian=False)

weather_data.to_csv(nyc_w)