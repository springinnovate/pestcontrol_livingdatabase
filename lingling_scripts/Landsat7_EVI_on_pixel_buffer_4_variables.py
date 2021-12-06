import ee
import geemap

Map = geemap.Map()

ETpoints_0 = ee.FeatureCollection("users/lliu6/PestControl_allSites"),
geometry = ee.Geometry.MultiPoint(),
Landsat7_EVI = ee.ImageCollection("LANDSAT/LE07/C01/T1_8DAY_EVI"),
ETpoints = ee.FeatureCollection("users/lliu6/BioControlDatabase_RAIF_all_sites_years")

# Refernce:
# https:#thegeoict.com/blog/2019/08/05/filter-a-feature-collection-by-attribute-in-google-earth-engine/
# https:#developers.google.com/earth-engine/guides/feature_collection_filtering

filter = ee.Filter.inList('Year', ['1997'])
filteredpoints = ETpoints.filter(filter)
print('filteredpoints', filteredpoints.first())
print('ETpoints', ETpoints.first())
Map.addLayer(filteredpoints, {'color': 'green'}, 'filteredArea')

print('number of sites Count after filter:', filteredpoints.size())

GDriveOutputImgFolder = 'GEEOutputs/Landsat7_EVI'
MODIS_LAI_FPAR = Landsat7_EVI

new_list = ee.List([])

for i in range(1999, 2021, 1):
    # for i in range(1999, 2001, 1):
    start_date = i + '-01-01'
    end_date = i + '-12-31'

    Fpar_500m1 = MODIS_LAI_FPAR \
        .filter(ee.Filter.date(start_date, end_date)) \
        .select('EVI')

    print('Fpar_500m1', Fpar_500m1)

    Fpar_500m_mean = Fpar_500m1.reduce(ee.Reducer.mean()).select(['EVI_mean'], ['EVI_mean_' + i])
    # print(Fpar_500m_mean)

    Fpar_500m_std = Fpar_500m1.reduce(ee.Reducer.stdDev()).select(['EVI_stdDev'], ['EVI_stdDev_' + i])
    # print(EVI_std)

    Fpar_500m_min = (Fpar_500m1.reduce(ee.Reducer.minMax())).select(['EVI_min'], ['EVI_min_' + i])
    Fpar_500m_max = (Fpar_500m1.reduce(ee.Reducer.minMax())).select(['EVI_max'], ['EVI_max_' + i])

    # (Fpar_500m_min)
    # (Fpar_500m_max)

    Fpar_500m_year = Fpar_500m_mean.addBands(Fpar_500m_std).addBands(Fpar_500m_min).addBands(Fpar_500m_max)
    # ('Fpar_500m_year_test',Fpar_500m_year)
    new_list = _list.add(ee.Image(Fpar_500m_year))
# print('new_list',new_list)

MODIS_Fpar_year = ee.ImageCollection.fromImages(new_list)

print(MODIS_Fpar_year)
img = MODIS_Fpar_year.toBands()
print('img', img)

Arid_points = ETpoints
Map.addLayer(Arid_points, {}, 'Arid_points')

# on pixel
npp_time_series = img.reduceRegions({
    'collection': Arid_points,
    'reducer': ee.Reducer.mean(),
    'scale': 500})
print(npp_time_series.first())

# Export.table(npp_time_series, 'Fpar_500m_2000_2020_pixel', {fileFormat: "csv"})
Export.table.toDrive({
    'collection': npp_time_series,
    'description': 'Landsat7_EVI_1999_2020_on_pixel',
    'fileFormat': 'CSV',
    'folder': GDriveOutputImgFolder
})


# buffered by a given distance
def bufferPoly(feature):
    return feature.buffer(m)


for m in range(500, 2001, m=m + 500):
    # # create buffer
    # m = 1500; #500 1000 1500 2000
    print('m', m)

    Arid_points_buffer = Arid_points.map(bufferPoly)

    # Map.addLayer(Arid_points_buffer, {}, 'Arid_points_buffer')

    # buffer 500, 1000,1500,2000
    npp_time_series = img.reduceRegions({
        'collection': Arid_points_buffer,
        'reducer': ee.Reducer.mean(),
        'scale': 500})
    print(npp_time_series.first())

    Export.table.toDrive({
        'collection': npp_time_series,
        'description': 'Landsat7_EVI_1999_2020_' + m,
        'fileFormat': 'CSV',
        'folder': GDriveOutputImgFolder
    })
Map
