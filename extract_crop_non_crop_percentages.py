
#pip install geemap
import ee
import geemap
import os
ee.Initialize()

ETpoints = ee.FeatureCollection('users/lliu6/PestControl_allSites')
dataset = ee.Image('USGS/NLCD/NLCD2001')
landcover = ee.Image(dataset.select('landcover'))

# source:https://github.com/google/earthengine-api/blob/master/python/examples/py/FeatureCollection/buffer.py
#.map()是对bart_stations中各元素循环应用()中的方法。
#lambda是一个匿名函数，是python语法的一种
#f是一个列表，后面为对f中每一个元素应用buffer(2000）方法
Arid_points_buffer = ETpoints.map(lambda f: f.buffer(2000))

# out_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
# nlcd_stats = os.path.join(out_dir, 'nlcd_stats.csv')

# if not os.path.exists(out_dir):
#     os.makedirs(out_dir)
nlcd_stats = "C:\\USDA-pestcontrol\\Landsat7\\nlcd_stats_test.csv"
# statistics_type can be either 'SUM' or 'PERCENTAGE'
# denominator can be used to convert square meters to other areal units, such as square kilimeters
geemap.zonal_statistics_by_group(landcover, Arid_points_buffer, nlcd_stats, statistics_type='PERCENTAGE', denominator=1000000, decimal_places=2)
