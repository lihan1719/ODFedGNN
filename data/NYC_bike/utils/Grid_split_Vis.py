import json
import folium
import transbigdata as tbd
import geopandas as gpd
from folium.plugins import HeatMap


# 生成grid地理信息
def gird_gen(grid_size=1000, region_file=None):
    """_summary_

    Args:
        grid_size (int, optional): _description_. Defaults to 1000.
        region_file (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    theta = 30
    # 可视化栅格
    Manhattan_area = gpd.read_file(region_file)
    params_init = tbd.area_to_params(Manhattan_area, accuracy=grid_size)
    params_init.update({
        'theta': theta,
    })
    grid_rec, params = tbd.area_to_grid(Manhattan_area, params=params_init)
    geojson_data = json.loads(grid_rec.to_json())
    return geojson_data, grid_rec, params


# 绘制热力图
def create_heatmap(data, map_obj):
    heatmap = HeatMap(data[['slat', 'slon', 'demand']],
                      radius=10,
                      min_opacity=0.5,
                      max_val=1.0,
                      gradient={
                          0.2: 'blue',
                          0.45: 'lime',
                          0.6: 'yellow',
                          0.8: 'red'
                      },
                      overlay=True)
    heatmap.add_to(map_obj)
    return map_obj


# 添加grid图层
def add_grid_layer(geojson_data, map_obj):
    grid_layer = folium.GeoJson(geojson_data,
                                style_function=lambda feature: {
                                    'fillColor': 'transparent',
                                    'color': 'black',
                                    'weight': 1,
                                    'opacity': 0.7
                                })
    grid_layer.add_to(map_obj)
    return map_obj
