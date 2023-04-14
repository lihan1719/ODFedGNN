import pandas as pd

# bike_data = pd.read_hdf('data/NYC_bike/raw_bike_data/NYC_2019.h5',
#                         key='bike_data')
weather_data = pd.read_hdf('data/NYC_bike/raw_bike_data/NYC_2019.h5',
                           key='weather_data')
# bike_data.to_csv('data/NYC_bike/raw_bike_data/NYC_Bike.csv', index=False)
weather_data.to_csv('data/NYC_bike/raw_bike_data/NYC_Weather.csv', index=False)
