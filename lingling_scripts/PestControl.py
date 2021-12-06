#!/usr/bin/env python
# coding: utf-8

# In[100]:


import pandas as pd
import numpy as np

sitesdata = pd.read_excel('C:/USDA-pestcontrol/BioControlDatabase.xlsx', sheet_name="sitedata")
landusedata = pd.read_excel('C:/USDA-pestcontrol/BioControlDatabase.xlsx', sheet_name="landusedata")
#ids = landusedata['Study_ID-Site']
#ids
#type(ids)

# # a series of Study_ID-Site IDs that are duplicated (the duplicates grouped together)
# #call duplicated() by default it marks all duplicates except for the first occurrence
# #you use the keep=False parameter it will show all the duplicates
# duplicates = ids[ids.duplicated(keep=False)].sort_values()
# #duplicates

# # a true/false series that marks whether or not each row's Study_ID-Site is in the list of duplicates
# row_id_is_duplicated = landusedata['Study_ID-Site'].isin(duplicates)
# #row_id_is_duplicated

# # get each row with a duplicated Study_ID-Site
# duplicate_rows = landusedata[row_id_is_duplicated]
# #duplicate_rows

# #duplicate_rows.sort_values(by=['Study_ID-Site']).to_csv('C:/USDA-pestcontrol/BioControlDatabase_landusedata_duplicates.xlsx')

unique = landusedata.drop_duplicates()

unique['Study_ID-Site'][unique['Study_ID-Site'].duplicated()]

#x = landusedata[landusedata['Study_ID-Site'] == 'mace01-RCC']

unique.drop(1539)  # this is the processed landuse data after removing duplicates

#result = pd.merge(sitesdata, landusedata,on="Study_ID-Site")
result = pd.merge(sitesdata, unique,on="Study_ID-Site")


with pd.ExcelWriter('C:/USDA-pestcontrol/Sitedata_landusedata_merge.xlsx') as writer:
    result.to_excel(writer)

with pd.ExcelWriter('C:/USDA-pestcontrol/landusedata_unique.xlsx') as writer:
    unique.to_excel(writer)


# # In[79]:


# mergedata = pd.read_excel('C:/USDA-pestcontrol/Sitedata_landusedata_merge2.xlsx', sheet_name="Sheet1")
# mergedata


# # In[80]:


# x1 = mergedata[mergedata['Study_ID-Site'] == 'bohn01-1-Mer']
# x1


# # In[81]:


# x2 =x1 = sitesdata[sitesdata['Study_ID-Site'] == 'bohn01-1-Mer']
# x2


# # In[82]:


# dfs = sitesdata['Study_ID-Site'].sort_values()
# dfs


# # In[83]:


# dfu = mergedata['Study_ID-Site'].sort_values()
# dfu


# # In[84]:


# when_false = dfs is dfu
# when_false


# # In[89]:


# Com = cmp(dfs,dfu)


# # In[90]:


# type(dfs)


# # In[91]:


# landusedata_ids = landusedata['Study_ID-Site'].unique()  # a list of all the Study_ID-Sites that exist in landusedata


# # In[94]:


# sitesdata[~sitesdata['Study_ID-Site'].isin(landusedata_ids)]


# # In[96]:


# x3 = sitesdata[sitesdata['Study_ID-Site'] == 'scha01-gI']
# x3


# # In[98]:


# x4 = unique[unique['Study_ID-Site'] == 'scha01-gI']
# x4


# # In[99]:


# x5 = landusedata[landusedata['Study_ID-Site'] == 'scha01-gI']
# x5


# # In[ ]:




