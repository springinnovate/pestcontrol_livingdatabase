import pandas as pd
import numpy as np

sitesdata = pd.read_excel('C:/USDA-pestcontrol/BioControlDatabase_NoExclusions.xlsx', sheet_name="sitedata")
landusedata = pd.read_excel('C:/USDA-pestcontrol/BioControlDatabase_NoExclusions.xlsx', sheet_name="landusedata")

#将两个字段连接
sitesdata['study_id-site'] = (sitesdata['Study_ID']).astype(str) + '-' + (sitesdata['Site']).astype(str)
#将字段内容转为小写
sitesdata['study_id-site'] = sitesdata['study_id-site'].str.lower()

#将两个字段连接
landusedata['study_id-site'] = (landusedata['Study_ID']).astype(str) + '-' + (landusedata['Site']).astype(str)
#将字段内容转为小写
landusedata['study_id-site'] = landusedata['study_id-site'].str.lower()

landusedata_uni = landusedata.drop_duplicates('study_id-site')

result = pd.merge(sitesdata, landusedata_uni,on='study_id-site')

with pd.ExcelWriter('C:/USDA-pestcontrol/sitedata_landusedata_uni_merge.xlsx') as writer:
    result.to_excel(writer)
