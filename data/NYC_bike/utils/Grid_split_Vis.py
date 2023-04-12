import json
import folium
import transbigdata as tbd
import geopandas as gpd

#栅格大小与角度
grid_size = 1000
theta = 30


def gird_vis(grid_size=1000, region_file=None):
    # 可视化栅格
    Manhattan_area = gpd.read_file(region_file)
    params_init = tbd.area_to_params(Manhattan_area, accuracy=grid_size)
    params_init.update({
        'theta': theta,
    })
    grid_rec, params = tbd.area_to_grid(Manhattan_area, params=params_init)
    geojson_data = json.loads(grid_rec.to_json())
    m = folium.Map(location=[40.763966, -73.97841], zoom_start=10)
    folium.GeoJson(geojson_data).add_to(m)
    return m, params


if __name__ == 'main':
    region_name = 'Manhattan_v2'
    region_file = 'raw_bike_data/' + region_name + '.json'
    grid_size = 1000
    grid_m = gird_vis()
    grid_m.save('data/NYC_bike/output/grid_{}_{}.html'.format(
        grid_size, region_name))
