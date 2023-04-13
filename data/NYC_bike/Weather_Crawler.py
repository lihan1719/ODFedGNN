#导入包
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import numpy as np
from datetime import datetime, timedelta
from selenium import webdriver

#相关设置
import warnings

warnings.filterwarnings("ignore")

file_path = 'data/NYC_bike/raw_bike_data/NYC_2019.h5'


def weather_crawler(date):
    option = webdriver.ChromeOptions()
    option.add_argument('headless')
    option.binary_location = "C:/Program Files/Google/Chrome/Application/chrome.exe"
    url = 'https://www.wunderground.com/history/daily/KLGA/date/' + date
    #     print(url)
    while True:
        driver = webdriver.Chrome(options=option)
        driver.get(url)
        tables = WebDriverWait(driver, 100).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table")))
        try:
            table = tables[1]
            newTable = pd.read_html(table.get_attribute('outerHTML'))
            weather_raw_data = newTable[0].dropna()
            break
        except:
            time.sleep(3)
    return weather_raw_data


results = []
#获取天气数据
# 定义起始日期和结束日期
date_last = '2018-12-31'
start_date = datetime.strptime('2019-01-01', '%Y-%m-%d')
end_date = datetime.strptime('2019-01-03', '%Y-%m-%d')

# 循环生成每一天的日期格式
current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime('%Y-%m-%d')
    raw_data = weather_crawler(date_str)
    results.append(raw_data)
    date_last = date_str
    current_date += timedelta(days=1)
    print("已采集完" + date_str + "的数据。")
    time.sleep(3)
#合并表格并写入文件
weather_data = pd.concat(results)
print('合并成功')
try:
    with pd.HDFStore(file_path, mode='a') as store:
        store.put('weather_data', weather_data)
except:
    with pd.HDFStore(file_path, mode='w') as store:
        store.put('weather_data', weather_data)
print('已写入文件中')