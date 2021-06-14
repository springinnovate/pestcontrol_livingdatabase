var ETpoints = ee.FeatureCollection("users/lliu6/PestControl_allSites"),
    MODIS_Phenology = ee.ImageCollection("MODIS/006/MCD12Q2");


var GDriveOutputImgFolder = 'GEEOutputs/MODIS_phenology';

var img = ee.Image('MODIS/006/MOD17A3HGF/2001_01_01').select(['Npp'],['Npp_' + 2001]);

//Adding years as bands to one image
for (var i = 2002; i < 2020; i++){
var img = img.addBands(ee.Image('MODIS/006/MOD17A3HGF/' + i + '_01_01')
                              .select(['Npp'],['Npp_' + i])
//                              .map(function(img) {
//                        return  img.multiply(0.0001);
//                       })
                       );
}
var img1 = img.multiply(0.0001);
img = img1;
print(img);

// on pixel
var Arid_points = ETpoints;

var npp_time_series = img.reduceRegions({
  collection: Arid_points,
  reducer: ee.Reducer.mean(),
  scale: 500});
print(npp_time_series.first());

//Export.table(npp_time_series, 'npp_time_series_on_pixel', {fileFormat: "csv"});
Export.table.toDrive({
  collection:npp_time_series,
  description: 'npp_2000_2019_on_pixel',
  fileFormat:'CSV',
  folder: GDriveOutputImgFolder
})


// buffer
var m = 500; //500 1000 1500 2000;

//buffered by a given distance
var bufferPoly = function(feature) {
  return feature.buffer(m);
};

var Arid_points_buffer = Arid_points.map(bufferPoly);

Map.addLayer(Arid_points, {}, 'Arid_points');
Map.addLayer(Arid_points_buffer, {}, 'Arid_points_buffer');


// buffer// buffer 500, 1000,1500,2000
var npp_time_series = img.reduceRegions({
  collection: Arid_points_buffer,
  reducer: ee.Reducer.mean(),
  scale: 500});
print(npp_time_series.first());

//Export.table(npp_time_series, 'npp_time_series_' + m, {fileFormat: "csv"});
Export.table.toDrive({
  collection:npp_time_series,
  description: 'npp_2000_2019_' + m,
  fileFormat:'CSV',
  folder: GDriveOutputImgFolder
})

//**********************************************************************
// buffer
var m = 1000; //500 1000 1500 2000;

//buffered by a given distance
var bufferPoly = function(feature) {
  return feature.buffer(m);
};

var Arid_points_buffer = Arid_points.map(bufferPoly);

Map.addLayer(Arid_points, {}, 'Arid_points');
Map.addLayer(Arid_points_buffer, {}, 'Arid_points_buffer');


// buffer// buffer 500, 1000,1500,2000
var npp_time_series = img.reduceRegions({
  collection: Arid_points_buffer,
  reducer: ee.Reducer.mean(),
  scale: 500});
print(npp_time_series.first());

//Export.table(npp_time_series, 'npp_time_series_' + m, {fileFormat: "csv"});
Export.table.toDrive({
  collection:npp_time_series,
  description: 'npp_2000_2019_' + m,
  fileFormat:'CSV',
  folder: GDriveOutputImgFolder
})

//***************************************************************************
// buffer
var m = 1500; //500 1000 1500 2000;

//buffered by a given distance
var bufferPoly = function(feature) {
  return feature.buffer(m);
};

var Arid_points_buffer = Arid_points.map(bufferPoly);

Map.addLayer(Arid_points, {}, 'Arid_points');
Map.addLayer(Arid_points_buffer, {}, 'Arid_points_buffer');


// buffer// buffer 500, 1000,1500,2000
var npp_time_series = img.reduceRegions({
  collection: Arid_points_buffer,
  reducer: ee.Reducer.mean(),
  scale: 500});
print(npp_time_series.first());

//Export.table(npp_time_series, 'npp_time_series_' + m, {fileFormat: "csv"});
Export.table.toDrive({
  collection:npp_time_series,
  description: 'npp_2000_2019_' + m,
  fileFormat:'CSV',
  folder: GDriveOutputImgFolder
})

//*******************************************************************************

// buffer
var m = 2000; //500 1000 1500 2000;

//buffered by a given distance
var bufferPoly = function(feature) {
  return feature.buffer(m);
};

var Arid_points_buffer = Arid_points.map(bufferPoly);

Map.addLayer(Arid_points, {}, 'Arid_points');
Map.addLayer(Arid_points_buffer, {}, 'Arid_points_buffer');


// buffer// buffer 500, 1000,1500,2000
var npp_time_series = img.reduceRegions({
  collection: Arid_points_buffer,
  reducer: ee.Reducer.mean(),
  scale: 500});
print(npp_time_series.first());

//Export.table(npp_time_series, 'npp_time_series_' + m, {fileFormat: "csv"});
Export.table.toDrive({
  collection:npp_time_series,
  description: 'npp_2000_2019_' + m,
  fileFormat:'CSV',
  folder: GDriveOutputImgFolder
})
