#!/usr/bin/env python
# encoding: utf-8
import os,numpy,csv;
stationEVI_filepath=r'D:\test_Landsat7_EVI_1999_2020_on_pixel.csv';

stationEVI_data=numpy.loadtxt(stationEVI_filepath,dtype=str,delimiter=',');
station_names=list(stationEVI_data[0]);
evi_index=[index for index in range(len(station_names)) if 'EVI' in station_names[index]];
evi_names=[station_names[i] for i in evi_index];
year_index=[index for index in range(len(station_names)) if 'Year' in station_names[index]];
property_Sindex=[index for index in range(len(station_names)) if 'BIOME' in station_names[index]];
property_index=range(property_Sindex[0],year_index[0]+1)
#
year_data=numpy.array(stationEVI_data[1:,year_index]).astype(numpy.float);
allYear_EVIdata=stationEVI_data[1:,evi_index];
matrix_flag=0;
for iyear in range(1999,2021):
    year_Stationindex=numpy.where(year_data==iyear)[0];
    if year_Stationindex.size==0: continue;
    yearEVI_index=[index for index in range(len(evi_names)) if str(iyear) in evi_names[index]];
    yearEVI_data=allYear_EVIdata[:,yearEVI_index];
    yearPro_data=stationEVI_data[1:,property_index];
    yearEVIPro_data=numpy.hstack((yearEVI_data,yearPro_data));
    if matrix_flag==0:
        stationEVIPro_infos=yearEVIPro_data;
    else:
        stationEVIPro_infos=numpy.vstack((stationEVIPro_infos,yearEVIPro_data));
# output the extract information
# stationEVIPro_infos=stationEVIPro_infos.astype(numpy.str)
# numpy.savetxt(r'D:\extractEVI.csv',stationEVI_data,delimiter=',')
stationEVIPro_infos=stationEVIPro_infos.tolist();
open_flag = open(r'D:\extractEVI.csv', 'w',newline='');
csv_flag = csv.writer(open_flag);
csv_flag.writerows(stationEVIPro_infos);
open_flag.close();