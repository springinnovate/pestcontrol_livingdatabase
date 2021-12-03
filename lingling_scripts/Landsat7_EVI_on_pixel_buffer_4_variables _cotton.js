
//Refernce:
//https://thegeoict.com/blog/2019/08/05/filter-a-feature-collection-by-attribute-in-google-earth-engine/
//https://developers.google.com/earth-engine/guides/feature_collection_filtering


var GDriveOutputImgFolder = 'GEEOUTPUTS/Landsat7_EVI'; 

//***********************************************ON PIXEL***************************

  var start_date = '1997-01-01';
  var end_date = '2008-12-31';
  
  var Fpar_500m1 = L5.merge(L7)
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
  
var Arid_points = ETpoints;
Map.addLayer(Arid_points, {}, 'Arid_points');

// on pixel
var npp_time_series = img.reduceRegions({
  collection: Arid_points,
  reducer: ee.Reducer.mean(),
  scale: 30});
print(npp_time_series);


Export.table.toDrive({
  collection:npp_time_series,
  description: 'Landsat7_EVI_cotton_current_year_on_pixel',
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


  var start_date = '1997-01-01';
  var end_date = '2008-12-31';

  
  var Fpar_500m1 = L5.merge(L7)
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
  
var Arid_points = ETpoints;
Map.addLayer(Arid_points, {}, 'Arid_points');
var Arid_points_buffer = Arid_points.map(bufferPoly);
//Map.addLayer(Arid_points_buffer, {}, 'Arid_points_buffer');

// buffer 500, 1000,1500,2000
var npp_time_series = img.reduceRegions({
  collection: Arid_points_buffer,
  reducer: ee.Reducer.mean(),
  scale: 30});
print(npp_time_series.first());

Export.table.toDrive({
  collection:npp_time_series,
  description: 'Landsat7_EVI_cotton_current_year_buffer_'+ m,
  fileFormat:'CSV',
  folder: GDriveOutputImgFolder
}) 

}// end for bUFFER



