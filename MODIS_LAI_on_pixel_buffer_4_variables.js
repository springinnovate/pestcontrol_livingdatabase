var ETpoints = ee.FeatureCollection("users/lliu6/PestControl_allSites"),
    MODIS_LAI_FPAR = ee.ImageCollection("MODIS/006/MOD15A2H");



var GDriveOutputImgFolder = 'GEEOutput/LAI';

var new_list = ee.List([]);

for (var i = 2000; i < 2021; i++){
//for (var i = 2000; i < 2001; i++){
  var start_date = i + '-01-01';
  var end_date = i + '-12-31';

  var Lai_500m1 = MODIS_LAI_FPAR
                      .filter(ee.Filter.date(start_date, end_date))
                      .select('Lai_500m');

  //print('Lai_500m1',Lai_500m1)


  var Lai_500m_mean = Lai_500m1.reduce(ee.Reducer.mean()).select(['Lai_500m_mean'],['Lai_500m_mean_' + i]);
  //print(Lai_500m_mean);

  var Lai_500m_std = Lai_500m1.reduce(ee.Reducer.stdDev()).select(['Lai_500m_stdDev'],['Lai_500m_stdDev_' + i]);
  //print(Lai_500m_std);

     var  Lai_500m_min = (Lai_500m1.reduce(ee.Reducer.minMax())).select(['Lai_500m_min'],['Lai_500m_min_' + i]);
     var  Lai_500m_max = (Lai_500m1.reduce(ee.Reducer.minMax())).select(['Lai_500m_max'],['Lai_500m_max_' + i]);


  //(Lai_500m_min);
  //(Lai_500m_max);

    var Lai_500m_year = (Lai_500m_mean.addBands(Lai_500m_std).addBands(Lai_500m_min).addBands(Lai_500m_max)).multiply(0.1);
   //print('Lai_500m_year_test',Lai_500m_year);

  // var Lai_500m_year = (Lai_500m_mean.addBands(Lai_500m_std).addBands(Lai_500m_min).addBands(Lai_500m_max));
  // print('Lai_500m_year',Lai_500m_year);
  var new_list = new_list.add(ee.Image(Lai_500m_year));
  //print('new_list',new_list);

}

 var MODIS_Lai_year = ee.ImageCollection.fromImages(new_list);

 print(MODIS_Lai_year);

 var img = MODIS_Lai_year.toBands();
 print('img',img);


var Arid_points = ETpoints;

// on pixel
var npp_time_series = img.reduceRegions({
  collection: Arid_points,
  reducer: ee.Reducer.mean(),
  scale: 500});
print(npp_time_series.first());

//Export.table(npp_time_series, 'Lai_500m_2000_2020_pixel', {fileFormat: "csv"});
Export.table.toDrive({
  collection:npp_time_series,
  description: 'Lai_2000_2020_on_pixel',
  fileFormat:'CSV',
  folder: GDriveOutputImgFolder
})


for (var m = 500; m < 2001; m=m+500){
// // create buffer
// var m = 1500; //500 1000 1500 2000;
print('m',m);

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
  description: 'Lai_2000_2020_' + m,
  fileFormat:'CSV',
  folder: GDriveOutputImgFolder
})

}

