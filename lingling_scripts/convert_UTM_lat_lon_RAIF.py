#from pyproj import Proj, transform
#https://gis.stackexchange.com/questions/78838/converting-projected-coordinates-to-lat-lon-using-python
import pandas as pd
import numpy as np
import pyproj

print(pyproj.__version__)
print(pyproj.proj_version_str)

file = 'C:/USDA-pestcontrol/RAIF.coordinates.ETRS89H30.daniel.paredes.csv'
test1 = pd.read_csv(file)
xutm = test1.xutm
yutm = test1.yutm


#proj = pyproj.Transformer.from_crs(3857, 4326, always_xy=True)
proj = pyproj.Transformer.from_crs(25830, 4326, always_xy=True)

x1,y1 = xutm,yutm
x2, y2 = proj.transform(x1, y1)
#x2,y2 = proj.transform(inProj,outProj,x1,y1)

test1['x2'] = x2
test1['y2'] = y2

test1.to_csv('C:/USDA-pestcontrol/RAIF_coordinates_ETRS89H30_daniel_paredes_lat_lon.csv')
