import pandas as pd
import numpy as np
#import utm

file = 'C:/USDA-pestcontrol/compare_Landsat_SR_EVI2/Landsat7_EVI_SR_current_year_buffer_2000.csv'
test1 = pd.read_csv(file)
EVI_SR = test1[['EVI_max','EVI_mean','EVI_min', 'EVI_stdDev']]

file = 'C:/USDA-pestcontrol/compare_Landsat_SR_EVI2/Landsat7_EVI_current_year_buffer_2000.csv'
test1 = pd.read_csv(file)
EVI = test1[['EVI_max','EVI_mean','EVI_min', 'EVI_stdDev']]

# To find the correlation among the
# columns of df1 and df2 along the column axis
print(EVI_SR.corrwith(EVI, axis = 0))


