from selenium import webdriver
from selenium.common import exceptions  # selenium异常模块

browser_driver = webdriver.Firefox(
    executable_path=r"/usr/lib/firefox/geckodriver",  # 这里必须要是绝对路径
    # windows是.exe文件 xxx/xxx/geckodriver.exe, xxx/xxx/firefox.exe
    # linux直接是xxx/xxx/geckodriver, xxx/xxx/firefox
    firefox_binary=r"/usr/lib/firefox/firefox",
    # options=options
)
url = r'https://www.baidu.com/'
browser_driver.get(url)
