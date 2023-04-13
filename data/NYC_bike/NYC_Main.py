# 该python文件用于将收集的原始数据格式转化为模型训练格式,h5-->numpy

from utils import *
import pandas as pd

h5_file = 'data/NYC_bike/raw_bike_data/NYC_2019.h5'
t_s = '2019-07'
t_e = '2019-07'
# 读取数据
bike_data = pd.read_hdf(h5_file, key='bike_data')
weather_data = pd.read_hdf(h5_file, key='weather_data')

# 天气数据格式调整
weather_data = unit_unify(raw=weather_data)
weather_data = time_agg(weather_data, frep='H')
_, _ = gird_vis()
