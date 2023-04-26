# 对weather_data 进行clean
import pandas as pd


def classify_condition(condition):
    if condition in ['Fair', 'Fair / Windy']:
        return 'Sunny'
    elif condition in [
            'Mostly Cloudy', 'Partly Cloudy', 'Cloudy',
            'Mostly Cloudy / Windy', 'Cloudy / Windy', 'Partly Cloudy / Windy'
    ]:
        return 'Cloudy'
    elif condition in [
            'Rain', 'Light Rain', 'Heavy Rain', 'Rain / Windy',
            'Light Rain / Windy', 'Heavy Rain / Windy'
    ]:
        return 'Rainy'
    elif condition in [
            'Light Snow', 'Snow', 'Heavy Snow / Windy', 'Light Snow / Windy',
            'Snow / Windy', 'Snow and Sleet', 'Light Snow and Sleet',
            'Wintry Mix', 'Wintry Mix / Windy'
    ]:
        return 'Snowy'
    elif condition in [
            'T-Storm', 'Thunder', 'T-Storm / Windy', 'Heavy T-Storm / Windy',
            'Heavy T-Storm', 'Thunder in the Vicinity', 'Thunder / Windy',
            'Light Rain with Thunder'
    ]:
        return 'Thunderstorm'
    else:
        return 'Other'


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


# 删除无用字段,整理成模型能识别的数据格式
def tune_col(wethear_data, col, train=True):
    wethear_data.drop(col, inplace=True, axis=1)
    wethear_data['Condition'] = wethear_data['Condition'].apply(
        classify_condition)
    if train != False:
        wethear_data['Condition'] = wethear_data['Condition'].astype(
            'category').cat.codes
    return wethear_data