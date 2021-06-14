import pandas as pd
import numpy as np
#import utm


file = 'C:/USDA-pestcontrol/RAIF_coordinates_ETRS89H30_daniel_paredes_lat_lon.csv'
test1 = pd.read_csv(file)

#连接三个字段
test1['new_id'] = test1['x2'].round(4).astype(str) + '-' + test1['y2'].round(4).astype(str) + '-' + test1['year'].astype(str)
test1_uni = test1.drop_duplicates('new_id')

print(test1_uni)

test1_uni.to_csv('C:/USDA-pestcontrol/RAIF_coordinates_ETRS89H30_daniel_paredes_lat_lon_drop_duplicates.csv')
