var ETpoints_0 = ee.FeatureCollection("users/lliu6/PestControl_allSites"),
    geometry = /* color: #d63000 */ee.Geometry.MultiPoint(),
    Landsat7_EVI = ee.ImageCollection("LANDSAT/LE07/C01/T1_8DAY_EVI"),
    ETpoints = ee.FeatureCollection("users/lliu6/BioControlDatabase_RAIF_all_sites_years"),
    Landsat5_EVI = ee.ImageCollection("LANDSAT/LT5_L1T_8DAY_EVI"),
    ETpoints_CONUS = ee.FeatureCollection("users/lliu6/BioControlDatabase_RAIF_CONUS");


// there is not data in 2018
var Landsat7_NDVI = ee.ImageCollection("LANDSAT/LE7_L1T_8DAY_NDVI");
var Landsat5_NDVI = ee.ImageCollection("LANDSAT/LT5_L1T_8DAY_NDVI");

var Landsat7_EVI =  Landsat7_NDVI;
var Landsat5_EVI = Landsat5_NDVI;

//Refernce:
//https://thegeoict.com/blog/2019/08/05/filter-a-feature-collection-by-attribute-in-google-earth-engine/
//https://developers.google.com/earth-engine/guides/feature_collection_filtering

var GDriveOutputImgFolder = 'GEEOUTPUTS';  
Arid_points = ETpoints_CONUS;
//buffered by a given distance 
var bufferPoly = function(feature) {
  return feature.buffer(m); 
};


var start_year = 0;

for (var m = 500; m < 501; m=m+500){
//  for (var m = 500; m < 2001; m=m+500){
  print('m',m);

//time period in CONUS from 2003 to 2014
for (var i = 2003; i < 2015; i++){
//  for (var i = 1997; i < 2019; i++){
//for (var i = 1997; i < 2000; i++){

// build mask using nearest distance 
if (i <= 2002){
 var mask_i =  2001;
} else if (i <= 2004){
  var mask_i = 2004;
}  else if (i <= 2006){
  var mask_i = 2006;
}  else if (i <= 2009){
  var mask_i = 2008;
}  else if (i <= 2012){
  var mask_i = 2011;
}  else if (i <= 2014){
  var mask_i = 2013;
} else{
 var  mask_i = 2016;
}
 
var lc_file = "USGS/NLCD/NLCD"+mask_i;  
print(lc_file);
var dataset1 = ee.Image(lc_file);
var landcover1 = ee.Image(dataset1.select('landcover'));
print(landcover1);

var mask_crop = landcover1.eq(82)
https://gis.stackexchange.com/questions/371489/applying-less-than-and-greater-than-threshold-in-image-segmentation-in-google-ea
var mask_noncrop = (landcover1.lt(82)).or(landcover1.gt(82))
//Map.addLayer(mask_crop, {color: 'green'}, 'mask_crop');
//Map.addLayer(mask_noncrop, {color: 'red'}, 'mask_noncrop');


var year = i.toString();

//FIND POINTS IN EACH YEAR
var filter = ee.Filter.inList('Year', [year]);
var filteredpoints= ETpoints_CONUS.filter(filter);
print(typeof(filteredpoints));
print('filteredpoints',filteredpoints.first());
//print('ETpoints',ETpoints.first());
//Map.addLayer(filteredpoints, {color: 'green'}, 'filteredArea');

//print('number of sites Count after filter:', i, filteredpoints.size());

https://stackoverflow.com/questions/43948008/convert-what-looks-like-a-number-but-isnt-to-an-integer-google-earth-engine
var num_points = filteredpoints.size().getInfo();
print('number of sites:', i, num_points);
print(num_points+'$');
print(typeof(num_points));

print(parseInt(num_points.toString())>0, num_points);
if (num_points > 0) {
  print('start');
  
  start_year = start_year+1;// record the first year with points in the CONUS
  
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
                      .select('NDVI');
 
  print('Fpar_500m1',Fpar_500m1) 
  
  
  var Fpar_500m_mean = Fpar_500m1.reduce(ee.Reducer.mean())//.select(['EVI_mean'],['EVI_mean_' + i]);
  //print(Fpar_500m_mean);
  
  var Fpar_500m_std = Fpar_500m1.reduce(ee.Reducer.stdDev())//.select(['EVI_stdDev'],['EVI_stdDev_' + i]);
  //print(EVI_std);
    
  var  Fpar_500m_min = (Fpar_500m1.reduce(ee.Reducer.minMax())).select(['NDVI_min'])//,['EVI_min_' + i]);
  var  Fpar_500m_max = (Fpar_500m1.reduce(ee.Reducer.minMax())).select(['NDVI_max'])//,['EVI_max_' + i]);
  
//********************************crop pixels**************************
// apply crop_mask
var img = (Fpar_500m_mean.addBands(Fpar_500m_std).addBands(Fpar_500m_min).addBands(Fpar_500m_max)).updateMask(mask_crop);
Map.addLayer(img, {}, 'image_crop');  
var Arid_points = filteredpoints;
Map.addLayer(Arid_points, {}, 'Arid_points');
print('crop',m)
var Arid_points_buffer = Arid_points.map(bufferPoly);
//Map.addLayer(Arid_points_buffer, {}, 'Arid_points_buffer');

// buffer 500, 1000,1500,2000
var npp_time_series = img.reduceRegions({
  collection: Arid_points_buffer,
  reducer: ee.Reducer.mean(),
  scale: 30});
print(npp_time_series.first());

print('start_year',start_year)
if (start_year == 1) {
  var npp_merge1 = npp_time_series ;//  block of code to be executed if the condition is true
}else{ 
  var npp_merge1 = npp_merge1.merge(npp_time_series)
}
print(npp_merge1);


//***********************non crop pixels******************************
// apply noncrop pixels
var img = (Fpar_500m_mean.addBands(Fpar_500m_std).addBands(Fpar_500m_min).addBands(Fpar_500m_max)).updateMask(mask_noncrop);
  
var Arid_points = filteredpoints;
Map.addLayer(Arid_points, {}, 'Arid_points');
print('non-crop',m)
var Arid_points_buffer = Arid_points.map(bufferPoly);
//Map.addLayer(Arid_points_buffer, {}, 'Arid_points_buffer');

// buffer 500, 1000,1500,2000
var npp_time_series = img.reduceRegions({
  collection: Arid_points_buffer,
  reducer: ee.Reducer.mean(),
  scale: 30});
print(npp_time_series.first());

print('start_year',start_year);
if (start_year == 1) {
  var npp_merge_non_crop = npp_time_series ;//  block of code to be executed if the condition is true
}else{ 
  var npp_merge_non_crop = npp_merge_non_crop.merge(npp_time_series)
}
print(npp_merge_non_crop.first());

}// end if num_points
} // end for YEAR

Export.table.toDrive({
  collection:npp_merge1,
  description: 'Landsat7_NDVI_crop_current_year_buffer_'+ m,
  fileFormat:'CSV',
  folder: GDriveOutputImgFolder
}) 

Export.table.toDrive({
  collection:npp_merge_non_crop,
  description: 'Landsat7_NDVI_noncrop_current_year_buffer_'+ m,
  fileFormat:'CSV',
  folder: GDriveOutputImgFolder
}) 

}// end for bUFFER



