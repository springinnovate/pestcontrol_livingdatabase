var ETpoints_0 = ee.FeatureCollection("users/lliu6/PestControl_allSites"),
    geometry = /* color: #d63000 */ee.Geometry.MultiPoint(),
    ETpoints = ee.FeatureCollection("users/lliu6/BioControlDatabase_RAIF_all_sites_years"),
    L8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"),
    L7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2"),
    L5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2"),
    world = ee.FeatureCollection("FAO/GAUL/2015/level0");


//L5 and L7 are very similiar, so we callirated L8 to L7 to keep all three sensors consistent
// https://www.sciencedirect.com/science/article/pii/S0034425715302455
// var L8_band_cal = function(img){
//   var b4 = img.select('SR_B4'); 
//   var b5 = img.select('SR_B5');
  
//   var b4_cal = (b4.divide(10000.0).multiply(0.9372).add(0.0123)).multiply(10000);
//   var b5_cal = (b5.divide(10000.0).multiply(0.8339).add(0.0448)).multiply(10000);
//   return img.addBands(b4_cal.rename('b3_cal'))
//         .addBands(b5_cal.rename('b4_cal'))
//         .select(['b3_cal', 'b4_cal', 'pixel_qa'],['B3','B4', 'pixel_qa']);
// };

// Expressions
// var f_evi = '2.5 * ((nir - red) / (nir + 2.4 * red + 1))'; // EVI2 formula (two-band version)
//   var f_evi = '2.5 * ((B4 - B3) / (B4 + 2.4 * B3 + 1))'; // EVI2 formula (two-band version)
var f_evi =  '2.5 * ((NIR-RED) / (NIR + 6 * RED - 7.5* BLUE +1))'

//NDVI
//var f_evi =  '(NIR-RED) / (NIR + RED)'

// VegIndex calculator. Calculate the EVI index (two-band versiob)
function calcIndex(image){
  var evi = image.expression(
      f_evi,
        {
            RED: image.select('SR_B3').multiply(0.0000275),    // 620-670nm, RED
            NIR: image.select('SR_B4').multiply(0.0000275),    // 841-876nm, 
            BLUE: image.select('SR_B1').multiply(0.0000275)
        //   B3:image.select('B3').multiply(0.0001),    // 620-670nm, RED
        //   B4:image.select('B4').multiply(0.0001)    // 841-876nm, NIR
         });
    // Rename that band to something appropriate
    //var dimage = ee.Date(ee.Number(image.get('system:time_start'))).format();
    //return evi.select([0], [what]).set({'datef': dimage,'system:time_start': ee.Number(image.get('system:time_start'))});
   //return image.addBands((evi.rename('EVI')).multiply(10000).int16());
   return image.addBands((evi.rename('EVI')).multiply(10000).int16());

}

// Function to mask clouds in Landsat imagery.
var maskClouds = function(image){
  // bit positions: find by raising 2 to the bit flag code 
  var cloudBit = Math.pow(2, 3); //32
  var shadowBit = Math.pow(2, 4); // 8
  var snowBit = Math.pow(2, 5); //16
  var fillBit = Math.pow(2,0); // 1
  // extract pixel quality band
  var qa = image.select('QA_PIXEL');    
  // create and apply mask
  var mask = qa.bitwiseAnd(cloudBit).eq(0).and(  // no clouds
              qa.bitwiseAnd(shadowBit).eq(0)).and( // no cloud shadows
              qa.bitwiseAnd(snowBit).eq(0)).and(   // no snow
              qa.bitwiseAnd(fillBit).eq(0))   ; // no fill
  
  // display orginal, mask and images_updated_with_mask
  //Map.addLayer(image, {bands: ['B5', 'B4', 'B3'], min: 0, max: 3000}, 'image');
  //Map.addLayer(mask, {}, 'mask');
  //Map.addLayer(image.updateMask(mask), {bands: ['B5', 'B4', 'B3'], min: 0, max: 3000}, 'image.updateMask(mask)');
  return image.updateMask(mask);   
};

// Function to mask excess EVI2 values defined as > 10000 and < 0
var maskExcess = function(image) {
    var hi = image.lte(10000);
    var lo = image.gte(0);
    var masked = image.mask(hi.and(lo));
    return image.mask(masked);
  };
  
  
  
//Refernce:
//https://thegeoict.com/blog/2019/08/05/filter-a-feature-collection-by-attribute-in-google-earth-engine/
//https://developers.google.com/earth-engine/guides/feature_collection_filtering


var GDriveOutputImgFolder = 'GEEOUTPUTS/Landsat7_EVI';  
 
//***********************************************ON PIXEL***************************
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
  var landsat_evi = L5.merge(L7)
                      //.merge(L8.map(L8_band_cal))
                      //L7
                      .filter(ee.Filter.date(start_date, end_date))
                      //.filterBounds(country)
                      //.filterBounds(Spain_sites)
                      .map(maskClouds)
                      //.map(getNDVI)
                      .map(calcIndex)
                      .map(maskExcess)
                      .select('EVI');
                      
  
  var Fpar_500m1 = landsat_evi;
 
  print('Fpar_500m1',Fpar_500m1.first()) 
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
  description: 'Landsat7_EVI_SR_current_year_on_pixel',
  fileFormat:'CSV',
  folder: GDriveOutputImgFolder
})  

//**************************************************************//

//buffered by a given distance
var bufferPoly = function(feature) {
  return feature.buffer(m); 
};



for (var m = 500; m < 2001; m=m+500){
//for (var m = 500; m < 1001; m=m+500){
  print('m',m);
for (var i = 1997; i < 2019; i++){
//for (var i = 1997; i < 2001; i++){
  
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
  
  var landsat_evi = L5.merge(L7)
                      //.merge(L8.map(L8_band_cal))
                      //L7
                      .filter(ee.Filter.date(start_date, end_date))
                      //.filterBounds(country)
                      //.filterBounds(Spain_sites)
                      .map(maskClouds)
                      //.map(getNDVI)
                      .map(calcIndex)
                      .map(maskExcess)
                      .select('EVI');
                      
  
  var Fpar_500m1 = landsat_evi;
 
  print('Fpar_500m1',Fpar_500m1.first()) 
  
  
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
  description: 'Landsat7_EVI_SR_current_year_buffer_'+ m,
  fileFormat:'CSV',
  folder: GDriveOutputImgFolder
}) 

}// end for bUFFER
