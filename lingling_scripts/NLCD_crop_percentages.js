var NLCD = ee.ImageCollection("USGS/NLCD");


var GDriveOutputImgFolder = 'GEEOUTPUTS/Landsat7_EVI';

var ETpoints = ee.FeatureCollection("users/lliu6/BioControlDatabase_RAIF_all_sites_years");

var Arid_points = ETpoints;


// var landcover = NLCD
//                       .select('NLCD'+i+'_landcover')
//                       // .select('landcover')
//                       .filterDate(i+'-01-01',i+'-12-31')
//                       ;
var dataset = ee.Image('USGS/NLCD/NLCD2001');
var landcover1 = ee.Image(dataset.select('landcover'));
print(landcover1);


//buffered by a given distance
var m = 500;

var bufferPoly = function(feature) {
  return feature.buffer(m);
};

var Arid_points_buffer = Arid_points.map(bufferPoly);

// Reduce the region. The region parameter is the Feature geometry.
var totalpixels = landcover1.reduceRegions({
  collection: Arid_points_buffer,
  reducer: ee.Reducer.count(),
  scale: 30,
//  maxPixels: 1e9
});



// generate image only includes crop value 82
//https://gis.stackexchange.com/questions/342777/count-the-number-of-pixels-according-to-their-value-using-for-loop
var crop = landcover1.updateMask(landcover1.eq(82));
// Reduce the region. The region parameter is the Feature geometry.
var croppixels = crop.reduceRegions({
  collection: Arid_points_buffer,
  reducer: ee.Reducer.count(),
  scale: 30,
//  maxPixels: 1e9
});


Export.table.toDrive({
  collection: croppixels ,
  description: 'test_NLCD2001_crop_pixels',
  fileFormat:'CSV',
  folder: GDriveOutputImgFolder
})

Export.table.toDrive({
  collection: totalpixels ,
  description: 'test_NLCD2001_total_pixels',
  fileFormat:'CSV',
  folder: GDriveOutputImgFolder
})


//https://gis.stackexchange.com/questions/328920/creating-one-table-out-of-multiple-feature-collections-and-two-properties


// var npp_time_series = img.reduceRegions({
//   collection: Arid_points_buffer,
//   reducer: ee.Reducer.mean(),
//   scale: 500});






// var GDriveOutputImgFolder = 'GEEOutputs/NLCD_crop';

// var new_list = ee.List([]);



// var dataset = ee.Image('USGS/NLCD/NLCD2004')
// var landcover2 = ee.Image(dataset.select('landcover'))
// var dataset = ee.Image('USGS/NLCD/NLCD2006')
// var landcover3 = ee.Image(dataset.select('landcover'))
// var dataset = ee.Image('USGS/NLCD/NLCD2008')
// var landcover4 = ee.Image(dataset.select('landcover'))
// var dataset = ee.Image('USGS/NLCD/NLCD2011')
// var landcover5 = ee.Image(dataset.select('landcover'))
// var dataset = ee.Image('USGS/NLCD/NLCD2013')
// var landcover6 = ee.Image(dataset.select('landcover'))
// var dataset = ee.Image('USGS/NLCD/NLCD2016')
// var landcover7 = ee.Image(dataset.select('landcover'))

// for (var i = 2000; i < 2021; i++){
//   print('NLCD'+i+'_landcover')



// var img_year = img1.toBands();
// print(img_year);

// var new_list = new_list.add(ee.Image(img_year));

// }


// var MODIS_Phen_year = ee.ImageCollection.fromImages(new_list);
// print(MODIS_Phen_year);
// var img = MODIS_Phen_year.toBands();
// print('img',img);


// var Arid_points = ETpoints;



// for (var m = 500; m < 2001; m=m+500){
// // // create buffer
// // var m = 1500; //500 1000 1500 2000;
// print('m',m);

// //buffered by a given distance
// var bufferPoly = function(feature) {
//   return feature.buffer(m);
// };

// var Arid_points_buffer = Arid_points.map(bufferPoly);

// Map.addLayer(Arid_points, {}, 'Arid_points');
// Map.addLayer(Arid_points_buffer, {}, 'Arid_points_buffer');


// // buffer 500, 1000,1500,2000
// var npp_time_series = img.reduceRegions({
//   collection: Arid_points_buffer,
//   reducer: ee.Reducer.mean(),
//   scale: 500});
// print(npp_time_series.first());

// Export.table.toDrive({
//   collection:npp_time_series,
//   description: 'Landsat7_EVI_1999_2020_' + m,
//   fileFormat:'CSV',
//   folder: GDriveOutputImgFolder
// })

// }

