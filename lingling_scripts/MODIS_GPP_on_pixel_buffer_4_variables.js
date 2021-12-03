var MODIS_GPP = ee.ImageCollection("MODIS/006/MOD17A2H"),
    ETpoints = ee.FeatureCollection("users/lliu6/PestControl_allSites");



var GDriveOutputImgFolder = 'GEEOutputs/GPP';

var new_list = ee.List([]);
 
for (var i = 2000; i < 2021; i++){
//for (var i = 2000; i < 2001; i++){
  var start_date = i + '-01-01';
  var end_date = i + '-12-31';
  
  var GPP1 = MODIS_GPP
                      .filter(ee.Filter.date(start_date, end_date))
                      .select('Gpp');
 
                      
                      
  // using map function to do .multiply(0.0001)
  // var GPP1 = MODIS_GPP
  //                     .filter(ee.Filter.date(start_date, end_date))
  //                     .select('Gpp')
  //                     .map(function(img) {
  //                       return  img.multiply(0.0001);
  //                     });
                      
    //print(GPP1);
    
    var gpp_mean = GPP1.reduce(ee.Reducer.mean()).select(['Gpp_mean'],['Gpp_mean_' + i]);
    //print(gpp_mean);
  
    var gpp_std = GPP1.reduce(ee.Reducer.stdDev()).select(['Gpp_stdDev'],['Gpp_stdDev_' + i]);
    //print(gpp_std);
    
    var  gpp_min = (GPP1.reduce(ee.Reducer.minMax())).select(['Gpp_min'],['Gpp_min_' + i]);
    var  gpp_max = (GPP1.reduce(ee.Reducer.minMax())).select(['Gpp_max'],['Gpp_max_' + i]);
   // var gpp_minMax = GPP1.reduce(ee.Reducer.minMax());
    
  // print(gpp_min);
  // print(gpp_max);
    
  //  var gpp_year = gpp_mean.addBands(gpp_std).addBands(gpp_min).addBands(gpp_max);
   // print('gpp_year',gpp_year);
    
  var gpp_year = gpp_mean.addBands(gpp_std).addBands(gpp_min).addBands(gpp_max).multiply(0.0001);
    print('gpp_year_test',gpp_year);
    
    var new_list = new_list.add(ee.Image(gpp_year));
    print('new_list',new_list);
}  

var MODIS_GPP_year = ee.ImageCollection.fromImages(new_list);

print(MODIS_GPP_year);

var img = MODIS_GPP_year.toBands();
print('img',img);


var Arid_points = ETpoints;

// on pixel
var npp_time_series = img.reduceRegions({
  collection: Arid_points,
  reducer: ee.Reducer.mean(),
  scale: 500});
print(npp_time_series.first());

//Export.table(npp_time_series, 'gpp_2000_2020_pixel', {fileFormat: "csv"});
Export.table.toDrive({
  collection:npp_time_series,
  description: 'gpp_2000_2020_on_pixel',
  fileFormat:'CSV',
  folder: GDriveOutputImgFolder
}) 

// create buffer
var m = 1500; //500 1000 1500 2000;

//buffered by a given distance
var bufferPoly = function(feature) {
  return feature.buffer(m); 
};

var Arid_points_buffer = Arid_points.map(bufferPoly);

Map.addLayer(Arid_points, {}, 'Arid_points');
Map.addLayer(Arid_points_buffer, {}, 'Arid_points_buffer');


// buffer 500, 1000,1500,2000
var npp_time_series = img.reduceRegions({
  collection: Arid_points_buffer,
  reducer: ee.Reducer.mean(),
  scale: 500});
print(npp_time_series.first());

Export.table.toDrive({
  collection:npp_time_series,
  description: 'gpp_2000_2020_' + m,
  fileFormat:'CSV',
  folder: GDriveOutputImgFolder
}) 



