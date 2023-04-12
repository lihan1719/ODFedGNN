# 用于实验2：时间粒度分析，时间按15分钟、30分钟、1小时划分。

# 导入包
import pandas as pd
import numpy as np
import os

time_start = '2019-07'
time_end = '2019-09'
time_granularity = '60T'
data = r'data/NYC_bike/raw_bike_data/nyc_data2019.h5'

# 读取原始数据
weather_data = pd.read_hdf('data/NYC_bike/raw_bike_data/nyc_data2019.h5',
                           key='weather_data')
# 天气数据重采样，并使用前向填充的方式进行插值
weather_data = weather_data.resample(
    time_granularity).ffill()[time_start:time_end]
print("重采样后的数据shape为:{}".format(weather_data.shape))
with pd.HDFStore(data, mode='a') as store:
    store.put(
        'weather_data_{0}_{1}_{2}'.format(time_start, time_end,
                                          time_granularity), weather_data)

# 执行nyc_preprocess进行合并
os.system(
    'python data/NYC_bike/nyc_preprocess.py --time_start {0} --time_end {1} --out_dir data/NYC_bike/processed_bike_data/ --out_name nyc_{0}_{1}_{2}.npz --accuracy 1200 --time_granularity {2} --region_file data/NYC_bike/raw_bike_data/Manhattan.json'
    .format(time_start, time_end, time_granularity))
