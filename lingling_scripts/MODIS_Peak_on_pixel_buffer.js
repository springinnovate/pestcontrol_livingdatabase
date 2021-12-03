var ETpoints = ee.FeatureCollection("users/lliu6/PestControl_allSites"),
    MODIS_Phen = ee.ImageCollection("MODIS/006/MCD12Q2"),
    geometry = /* color: #d63000 */ee.Geometry.MultiPoint();


var GDriveOutputImgFolder = 'GEEOutputs/MODIS_phenology';

var DATE_1 = ee.Date('1970-01-01');
var new_list = ee.List([]);

for (var i = 2001; i < 2019; i++){

var DATE_2 = ee.Date(i+'-01-01');
var diff_1 = DATE_2.difference(DATE_1, 'days');
//print('diff_1',diff_1
//print('i',i)

var img1 = MODIS_Phen
                      .select('Peak_1')
                      .filterDate(i+'-01-01',i+'-12-31')
                      ;

var img_year = (img1.toBands()).subtract(diff_1);
//print(img_year);

var new_list = new_list.add(ee.Image(img_year));

}

// print('new_list',new_list);

var MODIS_Phen_year = ee.ImageCollection.fromImages(new_list);

print(MODIS_Phen_year);

var img = MODIS_Phen_year.toBands();
print('img',img);



//on pixel
var Arid_points = ETpoints;

var greenup_1__time_series = img.reduceRegions({
  collection: Arid_points,
  reducer: ee.Reducer.mean(),
  scale: 500});
print(greenup_1__time_series.first());

Export.table.toDrive({
  collection:greenup_1__time_series,
  description: 'peak_1_2001_2018_on_pixel',
  fileFormat:'CSV',
  folder: GDriveOutputImgFolder
})


for (var m = 500; m < 2001; m=m+500){

// // buffer
// var m = 500; //500 1000 1500 2000;
print('m',m);

//buffered by a given distance
var bufferPoly = function(feature) {
  return feature.buffer(m);
};

var Arid_points_buffer = Arid_points.map(bufferPoly);

Map.addLayer(Arid_points, {}, 'Arid_points');
Map.addLayer(Arid_points_buffer, {}, 'Arid_points_buffer');


// buffer// buffer 500, 1000,1500,2000
var greenup_1__time_series = img.reduceRegions({
  collection: Arid_points_buffer,
  reducer: ee.Reducer.mean(),
  scale: 500});
print(greenup_1__time_series.first());

Export.table.toDrive({
  collection:greenup_1__time_series,
  description: 'peak_1_2001_2018_' + m,
  fileFormat:'CSV',
  folder: GDriveOutputImgFolder
})

}


// //**********************************************************************
// // buffer
// var m = 1000; //500 1000 1500 2000;

// //buffered by a given distance
// var bufferPoly = function(feature) {
//   return feature.buffer(m);
// };

// var Arid_points_buffer = Arid_points.map(bufferPoly);

// Map.addLayer(Arid_points, {}, 'Arid_points');
// Map.addLayer(Arid_points_buffer, {}, 'Arid_points_buffer');


// // buffer// buffer 500, 1000,1500,2000
// var greenup_1__time_series = img.reduceRegions({
//   collection: Arid_points_buffer,
//   reducer: ee.Reducer.mean(),
//   scale: 500});
// print(greenup_1__time_series.first());

// Export.table.toDrive({
//   collection:greenup_1__time_series,
//   description: 'greenup_1_2001_2018_' + m,
//   fileFormat:'CSV',
//   folder: GDriveOutputImgFolder
// });

// //***************************************************************************
// // buffer
// var m = 1500; //500 1000 1500 2000;

// //buffered by a given distance
// var bufferPoly = function(feature) {
//   return feature.buffer(m);
// };

// var Arid_points_buffer = Arid_points.map(bufferPoly);

// Map.addLayer(Arid_points, {}, 'Arid_points');
// Map.addLayer(Arid_points_buffer, {}, 'Arid_points_buffer');


// // buffer// buffer 500, 1000,1500,2000
// var greenup_1__time_series = img.reduceRegions({
//   collection: Arid_points_buffer,
//   reducer: ee.Reducer.mean(),
//   scale: 500});
// print(greenup_1__time_series.first());

// Export.table.toDrive({
//   collection:greenup_1__time_series,
//   description: 'greenup_1_2001_2018_' + m,
//   fileFormat:'CSV',
//   folder: GDriveOutputImgFolder
// })

// //*******************************************************************************

// // buffer
// var m = 2000; //500 1000 1500 2000;

// //buffered by a given distance
// var bufferPoly = function(feature) {
//   return feature.buffer(m);
// };

// var Arid_points_buffer = Arid_points.map(bufferPoly);

// Map.addLayer(Arid_points, {}, 'Arid_points');
// Map.addLayer(Arid_points_buffer, {}, 'Arid_points_buffer');


// // buffer// buffer 500, 1000,1500,2000
// var greenup_1__time_series = img.reduceRegions({
//   collection: Arid_points_buffer,
//   reducer: ee.Reducer.mean(),
//   scale: 500});
// print(greenup_1__time_series.first());

// Export.table.toDrive({
//   collection:greenup_1__time_series,
//   description: 'greenup_1_2001_2018_' + m,
//   fileFormat:'CSV',
//   folder: GDriveOutputImgFolder
// })
