import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

import transbigdata as tbd
import time
from haversine import haversine, Unit


def create_mapping(grid_rec):
    index_mapping = {}
    reverse_mapping = {}
    index = 0
    for _, row in grid_rec.iterrows():
        i, j = row['LONCOL'], row['LATCOL']
        index_mapping[(i, j)] = index
        reverse_mapping[index] = (i, j)
        index += 1
    return index_mapping, reverse_mapping


# 生成OD矩阵与地理邻接矩阵
def process_hourly_data(hourly_data=None,
                        grid_rec=None,
                        adj=False,
                        grid_size=None):
    index_mapping, reverse_mapping = create_mapping(grid_rec)
    grid_count = len(grid_rec)
    od_matrix = np.zeros((grid_count, grid_count))
    adjacency_matrix = np.zeros((grid_count, grid_count))
    if adj == True:
        grid_rec['centroid'] = grid_rec.centroid
        for _, row_i in grid_rec.iterrows():
            for _, row_j in grid_rec.iterrows():
                start_grid = (row_i['LONCOL'], row_i['LATCOL'])
                end_grid = (row_j['LONCOL'], row_j['LATCOL'])
                start_index = index_mapping[start_grid]
                end_index = index_mapping[end_grid]
                if haversine(row_i['centroid'].coords[0],
                             row_j['centroid'].coords[0],
                             unit=Unit.METERS) < grid_size * 2:
                    adjacency_matrix[start_index, end_index] = haversine(
                        row_i['centroid'].coords[0],
                        row_j['centroid'].coords[0],
                        unit=Unit.METERS)
                else:
                    adjacency_matrix[start_index, end_index] = 0
        return adjacency_matrix
    else:
        for _, row in hourly_data.iterrows():
            start_grid = (row['SLONCOL'], row['SLATCOL'])
            end_grid = (row['ELONCOL'], row['ELATCOL'])
            start_index = index_mapping[start_grid]
            end_index = index_mapping[end_grid]
            od_matrix[start_index, end_index] = row['count']

    return od_matrix


def od_matrix_to_geopandas(od_matrix, reverse_mapping, oddata):
    # 获取非零元素的索引
    start_indices, end_indices = np.nonzero(od_matrix)
    od_data = []

    for start_index, end_index in zip(start_indices, end_indices):
        start_grid = reverse_mapping[start_index]
        end_grid = reverse_mapping[end_index]
        count = od_matrix[start_index, end_index]

        # 使用反向映射生成MultiLineString几何对象
        start_row = oddata.loc[(oddata['SLONCOL'] == start_grid[0])
                               & (oddata['SLATCOL'] == start_grid[1])]
        end_row = oddata.loc[(oddata['ELONCOL'] == end_grid[0])
                             & (oddata['ELATCOL'] == end_grid[1])]
        try:
            start_point = (start_row['SHBLON'].values[0],
                           start_row['SHBLAT'].values[0])
            end_point = (end_row['EHBLON'].values[0],
                         end_row['EHBLAT'].values[0])
        except:
            continue
        # geometry = MultiLineString([LineString([start_point, end_point])])
        geometry = tbd.tolinewitharrow(start_point[0], start_point[1],
                                       end_point[0], end_point[1])
        od_data.append([
            start_grid[0], start_grid[1], end_grid[0], end_grid[1], count,
            start_point[0], start_point[1], end_point[0], end_point[1],
            geometry
        ])

    # 创建一个新的geopandas DataFrame并添加相应的列
    columns = [
        'SLONCOL', 'SLATCOL', 'ELONCOL', 'ELATCOL', 'count', 'SHBLON',
        'SHBLAT', 'EHBLON', 'EHBLAT', 'geometry'
    ]
    od_gdf = gpd.GeoDataFrame(od_data, columns=columns)

    return od_gdf


# t_s = '2019-07-15 09:00:00'
# t_e = '2019-07-15 10:00:00'

# Bike_file = 'data/NYC_bike/raw_bike_data/NYC_Bike.csv'
# bike_data = pd.read_csv(Bike_file)
# bike_data['stime'] = pd.to_datetime(bike_data['stime'])
# bike_data.set_index('stime', inplace=True)
# bike_data = bike_data[t_s:t_e]
# bike_data.reset_index(inplace=True)
# region_file = 'data/NYC_bike/raw_bike_data/Manhattan.json'
# geobound = gpd.read_file(region_file)
# bike_data = rm_out(bike_data, geobound)
# _, grid_rec, params = gird_gen(grid_size=300, region_file=region_file)

# start_time = time.time()
# oddata = tbd.odagg_grid(bike_data, params, arrow=True)

# od, index_mapping, reverse_mapping = process_hourly_data(oddata, grid_rec)
# geodata = od_matrix_to_geopandas(od, reverse_mapping, oddata)
# # geodata = od_to_geodata(od, reverse_mapping, grid_rec)

# end_time = time.time()

# print("代码段执行时间：", end_time - start_time, "秒")
