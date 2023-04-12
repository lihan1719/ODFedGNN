# 进一步清理爬取下来的天气数据

# 导入包

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import math
from sklearn.decomposition import PCA

input_path = 'data/NYC_bike/raw_bike_data/nyc_bike_data_201901-201912.h5'
output_path = 'data/NYC_bike/raw_bike_data/nyc_data2019.h5'
# 读取数据
weather_data = pd.read_hdf(input_path, key='weather_data_clean')
#共享单车数据
#订单的开始时间作为需求产生的时间点
bike_data = pd.read_hdf(input_path, key='bike_data')
bike_data.drop(['etime', 'slon', 'slat', 'elat', 'elon'], inplace=True, axis=1)
bike_data['Demand'] = 1
bike_data['stime'] = pd.to_datetime(bike_data['stime'])
bike_data = bike_data.set_index('stime')
bike_data = bike_data.resample(rule='H', label='right').sum()
#将共享单车需求与天气数据合并
raw_data = weather_data.join(bike_data, how='inner')

#数据预处理
#1.更改天气条件字段

data_process = raw_data.copy(deep=True)
weather_mapping = {
    'Fair': '晴朗',
    'Partly Cloudy': '晴朗',
    'Partly Cloudy / Windy': '晴朗',
    'Mostly Cloudy': '多云',
    'Mostly Cloudy / Windy': '多云',
    'Cloudy / Windy': '多云',
    'Cloudy': '多云',
    'Haze': '雾霾',
    'Thunder': '雷暴',
    'T-Storm': '雷暴',
    'Thunder in the Vicinity': '雷暴',
    'Thunder / Windy': '雷暴',
    'Light Rain': '小雨',
    'Light Rain with Thunder': '小雨',
    'Light Rain / Windy': '小雨',
    'Rain': '大雨',
    'Heavy Rain': '大雨',
    'Heavy T-Storm': '大雨',
    'Heavy T-Storm / Windy': '大雨',
    'Fog': '雾'
}
data_process['Condition'] = data_process['Condition'].replace(weather_mapping)
data_process.drop(['Dew Point', 'Wind Gust', 'Precip.', 'Wind', 'Pressure'],
                  axis=1,
                  inplace=True)
#构造时间特征
data_process = data_process.reset_index(names='Time')

# 定义节假日和周末日期
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start='2019-01-01', end='2019-12-31')
weekends = CustomBusinessDay(weekmask='Sat Sun')

holiday_hours = []
for holiday in holidays:
    holiday_hours.extend(
        pd.date_range(start=holiday,
                      end=holiday + pd.Timedelta(hours=23),
                      freq='H'))

# 构造日期索引
date_rng = pd.date_range(start='2019-01-01 01:00:00',
                         end='2020-01-01 00:00:00',
                         freq='H')

# 构造tem表
tem = pd.DataFrame(date_rng, columns=['Time'])
tem['is_weekday'] = np.where(
    (tem['Time'].dt.weekday < 5) & ~(tem['Time'].isin(holiday_hours)), 1, 0)

data_process = pd.merge(data_process,
                        tem[['Time', 'is_weekday']],
                        on='Time',
                        how='left')

# 构造新字段Period用于表示时间段
periods = ['凌晨', '早上', '上午', '中午', '下午', '傍晚', '晚上']
period_ranges = [(0, 4), (5, 7), (8, 11), (12, 12), (13, 17), (18, 19),
                 (20, 23)]
data_process['Period'] = None

for i in range(len(periods)):
    start_hour = period_ranges[i][0]
    end_hour = period_ranges[i][1]
    data_process.loc[(data_process.Time.dt.hour >= start_hour) &
                     (data_process.Time.dt.hour <= end_hour),
                     'Period'] = periods[i]

# 原始特征矩阵
X = data_process[['Temperature', 'Humidity']].values
# PCA降维
pca = PCA(n_components=1)
new_feature = pca.fit_transform(X)
# 将新特征与其他特征进行组合
data_process['Temp_Humidity'] = new_feature
#删除之前的特征
data_process.drop(['Temperature', 'Humidity'], axis=1, inplace=True)

#保存文件
data_process.set_index('Time', inplace=True)
data_process.drop(columns=['Demand'], inplace=True, axis=1)
data_process['Condition'] = data_process['Condition'].astype(
    'category').cat.codes
data_process['Period'] = data_process['Period'].astype('category').cat.codes

try:
    with pd.HDFStore(output_path, mode='a') as store:
        store.put('weather_data', data_process)
except:
    with pd.HDFStore(output_path, mode='w') as store:
        store.put('weather_data', data_process)
