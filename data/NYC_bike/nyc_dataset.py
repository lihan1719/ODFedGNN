# 生成不同栅格尺寸，不同时间粒度的nyc数据集

# 导入包
import pandas as pd
import numpy as np
import os

time_start = '2019-07'
time_end = '2019-07'
data = r'data/NYC_bike/raw_bike_data/nyc_data2019.h5'
time_granularity = '60T'
accuracy = 300

# 读取原始数据
weather_data = pd.read_hdf('data/NYC_bike/raw_bike_data/nyc_data2019.h5',
                           key='weather_data')
# 重采样，并使用前向填充的方式进行插值
weather_data.drop(columns=['Condition', 'is_weekday', 'Period'],
                  axis=1,
                  inplace=True)
weather_data = weather_data.resample(
    time_granularity).ffill()[time_start:time_end]
print("重采样后的数据shape为:{}".format(weather_data.shape))
with pd.HDFStore(data, mode='a') as store:
    store.put(
        'weather_data_{0}_{1}_{2}'.format(time_start, time_end,
                                          time_granularity), weather_data)

# 执行nyc_preprocess进行合并
os.system(
    'python data/NYC_bike/nyc_preprocess.py --time_start {0} --time_end {1} --out_dir data/NYC_bike/processed_bike_data/ --out_name {0}_{1}_grid_{2}_{3}.npz --accuracy {2} --time_granularity {3} --region_file data/NYC_bike/raw_bike_data/Manhattan.json'
    .format(time_start, time_end, accuracy, time_granularity))
