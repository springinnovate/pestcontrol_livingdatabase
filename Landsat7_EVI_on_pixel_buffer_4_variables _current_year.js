var ETpoints_0 = ee.FeatureCollection("users/lliu6/PestControl_allSites"),
    geometry = /* color: #d63000 */ee.Geometry.MultiPoint(),
    Landsat7_EVI = ee.ImageCollection("LANDSAT/LE07/C01/T1_8DAY_EVI"),
    ETpoints = ee.FeatureCollection("users/lliu6/BioControlDatabase_RAIF_all_sites_years"),
    Landsat5_EVI = ee.ImageCollection("LANDSAT/LT5_L1T_8DAY_EVI");
    
//Refernce:
//https://thegeoict.com/blog/2019/08/05/filter-a-feature-collection-by-attribute-in-google-earth-engine/
//https://developers.google.com/earth-engine/guides/feature_collection_filtering


var GDriveOutputImgFolder = 'GEEOUTPUTS/Landsat7_EVI'; 

//***********************************************ON PIXEL***************************
for (var i = 1997; i < 2019; i++){
  
var year = i.toString();

//FIND POINTS IN EACH YEAR
  
var filter = ee.Filter.inList('Year', [year]);
var filteredpoints= ETpoints.filter(filter);
//print('filteredpoints',filteredpoints.first());
//print('ETpoints',ETpoints.first());
//Map.addLayer(filteredpoints, {color: 'green'}, 'filteredArea');

print('number of sites Count after filter:', filteredpoints.size());

  var start_date = i + '-01-01';
  var end_date = i + '-12-31';

//https://www.w3schools.com/js/js_if_else.asp 
  if (i < 1999) {
  var MODIS_LAI_FPAR =  Landsat5_EVI;//  block of code to be executed if the condition is true
} else {
   var MODIS_LAI_FPAR =  Landsat7_EVI;
}
  print ('input',MODIS_LAI_FPAR)
  
  var Fpar_500m1 = MODIS_LAI_FPAR
                      .filter(ee.Filter.date(start_date, end_date))
                      .select('EVI');
 
  print('Fpar_500m1',Fpar_500m1) 
  
  
  var Fpar_500m_mean = Fpar_500m1.reduce(ee.Reducer.mean())//.select(['EVI_mean'],['EVI_mean_' + i]);
  //print(Fpar_500m_mean);
  
  var Fpar_500m_std = Fpar_500m1.reduce(ee.Reducer.stdDev())//.select(['EVI_stdDev'],['EVI_stdDev_' + i]);
  //print(EVI_std);
    
  var  Fpar_500m_min = (Fpar_500m1.reduce(ee.Reducer.minMax())).select(['EVI_min'])//,['EVI_min_' + i]);
  var  Fpar_500m_max = (Fpar_500m1.reduce(ee.Reducer.minMax())).select(['EVI_max'])//,['EVI_max_' + i]);
  

  var img = Fpar_500m_mean.addBands(Fpar_500m_std).addBands(Fpar_500m_min).addBands(Fpar_500m_max);
  
var Arid_points = filteredpoints;
Map.addLayer(Arid_points, {}, 'Arid_points');

// on pixel
var npp_time_series = img.reduceRegions({
  collection: Arid_points,
  reducer: ee.Reducer.mean(),
  scale: 30});
print(npp_time_series);

//https://developers.google.com/earth-engine/apidocs/ee-featurecollection-merge
 if (i == 1997) {
  var npp_merge = npp_time_series ;//  block of code to be executed if the condition is true
}else{ 
  var npp_merge = npp_merge.merge(npp_time_series)
}
print(npp_merge);
} // end for YEAR
Export.table.toDrive({
  collection:npp_merge,
  description: 'Landsat7_EVI_current_year_on_pixel',
  fileFormat:'CSV',
  folder: GDriveOutputImgFolder
}) 

//**************************************************************//

//buffered by a given distance
var bufferPoly = function(feature) {
  return feature.buffer(m); 
};



for (var m = 500; m < 2001; m=m+500){
  print('m',m);
for (var i = 1997; i < 2019; i++){
//for (var i = 1997; i < 2000; i++){
  
var year = i.toString();

//FIND POINTS IN EACH YEAR
  
var filter = ee.Filter.inList('Year', [year]);
var filteredpoints= ETpoints.filter(filter);
//print('filteredpoints',filteredpoints.first());
//print('ETpoints',ETpoints.first());
//Map.addLayer(filteredpoints, {color: 'green'}, 'filteredArea');

print('number of sites Count after filter:', filteredpoints.size());

  var start_date = i + '-01-01';
  var end_date = i + '-12-31';
 
//https://www.w3schools.com/js/js_if_else.asp 
  if (i < 1999) {
  var MODIS_LAI_FPAR =  Landsat5_EVI;//  block of code to be executed if the condition is true
} else {
   var MODIS_LAI_FPAR =  Landsat7_EVI;
}
  print ('input',MODIS_LAI_FPAR)
  
  var Fpar_500m1 = MODIS_LAI_FPAR
                      .filter(ee.Filter.date(start_date, end_date))
                      .select('EVI');
 
  print('Fpar_500m1',Fpar_500m1) 
  
  
  var Fpar_500m_mean = Fpar_500m1.reduce(ee.Reducer.mean())//.select(['EVI_mean'],['EVI_mean_' + i]);
  //print(Fpar_500m_mean);
  
  var Fpar_500m_std = Fpar_500m1.reduce(ee.Reducer.stdDev())//.select(['EVI_stdDev'],['EVI_stdDev_' + i]);
  //print(EVI_std);
    
  var  Fpar_500m_min = (Fpar_500m1.reduce(ee.Reducer.minMax())).select(['EVI_min'])//,['EVI_min_' + i]);
  var  Fpar_500m_max = (Fpar_500m1.reduce(ee.Reducer.minMax())).select(['EVI_max'])//,['EVI_max_' + i]);
  

  var img = Fpar_500m_mean.addBands(Fpar_500m_std).addBands(Fpar_500m_min).addBands(Fpar_500m_max);
  
var Arid_points = filteredpoints;
Map.addLayer(Arid_points, {}, 'Arid_points');
var Arid_points_buffer = Arid_points.map(bufferPoly);
//Map.addLayer(Arid_points_buffer, {}, 'Arid_points_buffer');

// buffer 500, 1000,1500,2000
var npp_time_series = img.reduceRegions({
  collection: Arid_points_buffer,
  reducer: ee.Reducer.mean(),
  scale: 30});
print(npp_time_series.first());

if (i == 1997) {
  var npp_merge1 = npp_time_series ;//  block of code to be executed if the condition is true
}else{ 
  var npp_merge1 = npp_merge1.merge(npp_time_series)
}
print(npp_merge1);

} // end for YEAR

Export.table.toDrive({
  collection:npp_merge1,
  description: 'Landsat7_EVI_current_year_buffer_'+ m,
  fileFormat:'CSV',
  folder: GDriveOutputImgFolder
}) 

}// end for bUFFER



