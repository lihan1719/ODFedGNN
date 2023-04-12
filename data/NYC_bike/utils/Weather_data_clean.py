# 对weather_data 进行clean
import pandas as pd


# 修改单位
def unit_unify(raw=None):
    assert type(raw) != None
    #修改文本格式
    for i in raw.columns:
        raw.loc[:, i] = raw.loc[:, i].map(lambda x: x.replace('\xa0°', ''))
    #修改公式单位
    #公式1 C = (F-32)/1.8
    wendu = ['Temperature', 'Dew Point']
    for i in wendu:
        raw.loc[:, i] = raw.loc[:, i].map(lambda x: x.replace('F', ''))
        raw.loc[:, i] = raw.loc[:, i].astype(float)
        raw.loc[:, i] = (raw.loc[:, i] - 32) / 1.8
    #公式2 英里 = 1.609344千米
    sudu = ['Wind Speed', 'Wind Gust']
    for i in sudu:
        raw.loc[:, i] = raw.loc[:, i].map(lambda x: x.replace('mph', ''))
        raw.loc[:, i] = raw.loc[:, i].astype(float)
        raw.loc[:, i] = raw.loc[:, i] * 1.609344
    #公式3 英寸 = 25.4 毫米(mm)
    jiangshui = ['Pressure', 'Precip.']
    for i in jiangshui:
        raw.loc[:, i] = raw.loc[:, i].map(lambda x: x.replace('in', ''))
        raw.loc[:, i] = raw.loc[:, i].astype(float)
        raw.loc[:, i] = raw.loc[:, i] * 25.4
        # 修改%
        raw.loc[:, 'Humidity'] = raw.loc[:, 'Humidity'].astype(str)
        raw.loc[:, 'Humidity'] = raw.loc[:, 'Humidity'].map(
            lambda x: x.replace('%', ''))
        raw.loc[:, 'Humidity'] = raw.loc[:, 'Humidity'].astype(float)
    return raw


# 时间聚合
def time_agg(data, frep='H'):
    start_date = '2019-1-1 01:00:00'
    end_date = '2019-12-1'

    # 生成时间索引
    time_index = pd.DataFrame(pd.date_range(start=start_date,
                                            end=end_date,
                                            freq=frep),
                              columns=['Time'])
    time_index.set_index('Time', inplace=True)
    weather_data = time_index.join(
        data, how='left').fillna(method='ffill').fillna(method='bfill')
    return weather_data