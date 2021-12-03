var MODIS_GPP = ee.ImageCollection("MODIS/006/MOD17A2H"),
    ETpoints = ee.FeatureCollection("users/lliu6/PestControl_allSites");


// imput  select; year;scale
//output name in export product name and years
//var img = ee.Image('MODIS/006/MOD17A3HGF/2001_01_01').select(['Npp'],['Npp_' + 2001]);

//Adding years as bands to one image
//for (var i = 2000; i < 2021; i++){
//var img = img.addBands(ee.Image('MODIS/006/MOD17A3HGF/' + i + '_01_01').select(['Npp'],['Npp_' + i]))
//}
//print(img);

//Adding years as bands to one image
var img1 =             MODIS_GPP
                      .select('Gpp')
                      .filterDate('2000-01-01','2021-01-01')
                      .map(function(image){
                        return image.multiply(0.0001);
                      });


var img = img1.toBands();
print(img);

var Arid_points = ETpoints;

// on pixel
var npp_time_series = img.reduceRegions({
  collection: Arid_points,
  reducer: ee.Reducer.mean(),
  scale: 500});
print(npp_time_series.first());

Export.table(npp_time_series, 'gpp_2000_2021_pixel', {fileFormat: "csv"});

// create buffer
var m = 500; //500 1000 1500 2000;

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

Export.table(npp_time_series, 'gpp_2000_2021_' + m, {fileFormat: "csv"});


