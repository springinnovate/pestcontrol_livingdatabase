var DEFAULTYEAR = '2005';
function generateDatasets(endYear) {
  var yearlyMax = function(year, collection, bandName) {
    var startDate = ee.Date.fromYMD(year, 1, 1);
    var endDate = startDate.advance(1, 'year');
    var yearCollection = collection.filterDate(startDate, endDate);
    return yearCollection.max().select(bandName).set('year', year);
  };

  var yearlyMode = function(year, collection, bandName) {
    /// I hacked this in on 11/12/2024 because Becky & Colleen were on the
    // phone and needed it rightaway
    var startDate = ee.Date.fromYMD(year, 1, 1);
    var endDate = startDate.advance(1, 'year');
    var yearCollection = collection.filterDate(startDate, endDate);
    return yearCollection.mode().select(bandName).set('year', year);
  };

  var local_datasets = {
    '(*clear*)': '',
  };

  var band_names = [
    'dewpoint_2m_temperature',
    'maximum_2m_air_temperature',
    'mean_2m_air_temperature',
    'minimum_2m_air_temperature',
    'total_precipitation',
    'dewpoint_2m_temperature',
    'maximum_2m_air_temperature',
    'mean_2m_air_temperature',
    'minimum_2m_air_temperature',
    'total_precipitation',
    'dewpoint_2m_temperature',
    'maximum_2m_air_temperature',
    'mean_2m_air_temperature',
    'minimum_2m_air_temperature',
    'total_precipitation',
  ];

  var image_names = [
    'dewpoint annual maximum (ERA5/MONTHLY)',
    'max temp annual maximum (ERA5/MONTHLY)',
    'mean temp annual maximum (ERA5/MONTHLY)',
    'min temp annual maximum (ERA5/MONTHLY)',
    'precip annual maximum (ERA5/MONTHLY)',
    'dewpoint annual mean (ERA5/MONTHLY)',
    'max temp annual mean (ERA5/MONTHLY)',
    'mean temp annual mean (ERA5/MONTHLY)',
    'min temp annual mean (ERA5/MONTHLY)',
    'precip annual mean (ERA5/MONTHLY)',
    'dewpoint annual minimum (ERA5/MONTHLY)',
    'max temp annual minimum (ERA5/MONTHLY)',
    'mean temp annual minimum (ERA5/MONTHLY)',
    'min temp annual minimum (ERA5/MONTHLY)',
    'precip annual minimum (ERA5/MONTHLY)',
  ];

  for (var i = 0; i < band_names.length; i++) {
    var image = ee.ImageCollection.fromImages(ee.List.sequence(endYear - 2, endYear).map(function(year) {
      return yearlyMax(
        year,
        ee.ImageCollection('ECMWF/ERA5/MONTHLY').filter(ee.Filter.calendarRange(endYear - 2, endYear, 'year')),
        band_names[i]);
    })).mean().rename('B0');
    local_datasets[image_names[i]] = image;
  }

  var band_names_modis = [
    'EVI_Amplitude_1',
    'EVI_Area_1',
    'Greenup_1',
    'Peak_1',
    'Dormancy_1',
  ];

  var image_names_modis = [
    'EVI Amplitude (Max greenness) (MODIS/MCD12Q2)',
    'EVI Area (Total productivity) (MODIS/MCD12Q2)',
    'Greenup Day of Year (MODIS/MCD12Q2)',
    'Peak Day of Year (MODIS/MCD12Q2)',
    'Dormancy Day of Year (MODIS/MCD12Q2)',
  ];

  for (var j = 0; j < band_names_modis.length; j++) {
    var image_modis = ee.ImageCollection('MODIS/061/MCD12Q2').select(band_names_modis[j])
      .filter(ee.Filter.calendarRange(endYear, endYear, 'year'))
      .map(function(image) {
        return image.focal_mean({
          radius: 1000,
          units: 'meters'
        }).set('system:time_start', image.get('system:time_start'));
      }).mean().rename('B0');
    local_datasets[image_names_modis[j]] = image_modis;
  }

  local_datasets['Global SRTM Topographic Diversity'] = (
    ee.Image('CSP/ERGo/1_0/Global/SRTM_topoDiversity').select('constant').rename('B0'));

  local_datasets['GFSAD1000: Cropland Extent 1km Multi-Study Crop Mask'] = (
    ee.Image('USGS/GFSAD1000_V1').select('landcover').rename('B0'));


  // I hacked this in on 11/12/2024 because Becky & Colleen were on the
  // phone and needed it rightaway
  var localYear = null;
  if (endYear < 2016) {
    localYear = 2016;
  } else if (endYear > 2023) {
    localYear = 2023;
  } else {
    localYear = endYear
  }

  var croplabel = 'Dynamic World V1 -- Crop probability (only defined between 2016-2023)';
  local_datasets[croplabel] = yearlyMax(
    localYear,
    ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1'),
    'crops').rename('B0');

  var classificationlabel = 'Dynamic World V1 -- Most likely label in that year (only defined between 2016-2023)';
  local_datasets[classificationlabel] = yearlyMode(
    localYear,
    ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1'),
    'label').rename('B0');

  return local_datasets;
}

var discreteBands = {
  'Dynamic World V1 -- Most likely label in that year (only defined between 2016-2023)': true,
  'GFSAD1000: Cropland Extent 1km Multi-Study Crop Mask': true
};
var discretePalettes = {
  'Dynamic World V1 -- Most likely label in that year (only defined between 2016-2023)': {
    palette: ['#aec3d4', '#152106', '#225129', '#369b47', '#30eb5b', '#387242',
              '#6a2325', '#c3aa69', '#b76031'],
    labels: ['Water', 'Trees', 'Grass', 'Flooded vegetation', 'Crops',
             'Shrub & scrub', 'Built area', 'Bare ground', 'Snow & ice'],
    values: [0,1,2,3,4,5,6,7,8]
  },
  'GFSAD1000: Cropland Extent 1km Multi-Study Crop Mask': {
    palette: ['#ffffff', '#ffff64', '#aaf0f0', '#dcf064', '#c4c464',
              '#006400'],
    labels: ['Water', 'Cropland', 'Cropland (irrigated)', 'Cropland (rainfed)', 'Mixed Cropland', 'Non-cropland'],
    values: [0,1,2,3,4,5]
  }
};

var legend_styles = {
  'black_to_red': ['000000', '005aff', '43c8c8', 'fff700', 'ff0000'],
  'blue_to_green': ['440154', '414287', '218e8d', '5ac864', 'fde725'],
  'cividis': ['00204d', '414d6b', '7c7b78', 'b9ac70', 'ffea46'],
  'viridis': ['440154', '355e8d', '20928c', '70cf57', 'fde725'],
  'blues': ['f7fbff', 'c6dbef', '6baed6', '2171b5', '08306b'],
  'reds': ['fff5f0', 'fcbba1', 'fb6a4a', 'cb181d', '67000d'],
  'turbo': ['321543', '2eb4f2', 'affa37', 'f66c19', '7a0403'],
};
var default_legend_style = 'blue_to_green';


function applyPalette(datasetName, active_context) {
  if (discreteBands[datasetName]) {
    var discreteScheme = discretePalettes[datasetName];
    active_context.visParams = {
      min: Math.min.apply(null, discreteScheme.values),
      max: Math.max.apply(null, discreteScheme.values),
      palette: discreteScheme.palette
    };
    active_context.legend_panel.clear();
    active_context.legend_panel.add(ui.Label({value: datasetName, style: {fontWeight: 'bold'}}));
    discreteScheme.labels.forEach(function(label, idx) {
      active_context.legend_panel.add(ui.Label({
        value: label,
        style: {backgroundColor: discreteScheme.palette[idx], padding: '4px'}
      }));
    });
  } else {
    active_context.visParams.palette = legend_styles[default_legend_style];
    active_context.build_legend_panel();
  }
  active_context.updateVisParams();
}

function changeColorScheme(key, active_context) {
  active_context.visParams.palette = legend_styles[key];
  active_context.build_legend_panel();
  active_context.updateVisParams();
}

var linkedMap = ui.Map();
var linker = ui.Map.Linker([ui.root.widgets().get(0), linkedMap]);

var splitPanel = ui.SplitPanel({
  firstPanel: linker.get(0),
  secondPanel: linker.get(1),
  orientation: 'horizontal',
  wipe: true,
  style: {stretch: 'both'}
});
ui.root.widgets().reset([splitPanel]);

var panel_list = [];
[[Map, 'left'], [linkedMap, 'right']].forEach(function(mapside, index) {
    var active_context = {
      'datasets': {},
      'last_layer': null,
      'raster': null,
      'point_val': null,
      'last_point_layer': null,
      'map': mapside[0],
      'legend_panel': null,
      'visParams': null,
      'datasetName': null,
    };

    active_context.map.style().set('cursor', 'crosshair');
    active_context.visParams = {
      min: 0.0,
      max: 100.0,
      palette: legend_styles[default_legend_style],
    };

    var panel = ui.Panel({
      layout: ui.Panel.Layout.flow('vertical'),
      style: {
        'position': "middle-"+mapside[1],
        'backgroundColor': 'rgba(255, 255, 255, 0.4)'
      }
    });

    var default_control_text = mapside[1]+' controls';
    var controls_label = ui.Label({
      value: default_control_text,
      style: {
        backgroundColor: 'rgba(0, 0, 0, 0)',
      }
    });
    var select = ui.Select({
      items: Object.keys(active_context.datasets),
      onChange: function(datasetName, self) {
          self.setDisabled(true);
          active_context.map.setZoom(10);
          var original_value = self.getValue();
          self.setPlaceholder('loading (this may mean the app is crashed, reload the page if you think so)...');
          self.setValue(null, false);
          if (active_context.last_layer !== null) {
            active_context.map.remove(active_context.last_layer);
            min_val.setDisabled(true);
            max_val.setDisabled(true);
          }
          if (active_context.datasets[datasetName] == '') {
            self.setValue(original_value, false);
            self.setDisabled(false);
            return
          }
          active_context.raster = active_context.datasets[datasetName];

          console.log(datasetName);
          active_context.datasetName = datasetName
          if (datasetName in discretePalettes) {
            console.log('discrete');
            applyPalette(datasetName, active_context);
            min_val.setValue("n/a", false);
            max_val.setValue("n/a", false);
            min_val.setDisabled(true);
            max_val.setDisabled(true);
            active_context.last_layer = active_context.map.addLayer(
                active_context.raster, active_context.visParams);
          } else {
            active_context.visParams.palette = legend_styles[default_legend_style];
            var mean_reducer = ee.Reducer.percentile([10, 90], ['p10', 'p90']);
            var meanDictionary = active_context.raster.reduceRegion({
              reducer: mean_reducer,
              geometry: active_context.map.getBounds(true),
              bestEffort: true,
            });

            ee.data.computeValue(meanDictionary, function (val) {
              if (val['B0_p10'] != val['B0_p90']) {
                active_context.visParams = {
                  min: val['B0_p10'],
                  max: val['B0_p90'],
                  palette: active_context.visParams.palette,
                };
              } else {
                active_context.visParams = {
                  min: 0,
                  max: val['B0_p90'],
                  palette: active_context.visParams.palette,
                };
              }
              active_context.last_layer = active_context.map.addLayer(
                active_context.raster, active_context.visParams);
              min_val.setValue(active_context.visParams.min, false);
              max_val.setValue(active_context.visParams.max, false);
              min_val.setDisabled(false);
              max_val.setDisabled(false);

            });
            build_legend_panel();
          }
          self.setValue(original_value, false);
          self.setDisabled(false);
      }
    });

    var selectSet = false;
    var yearChangeFunc = function(value) {
      var endYear = parseInt(value, 10);
      active_context.datasets = generateDatasets(endYear);
      var currentKey = select.getValue();
      if (!selectSet) {
        select.items().reset(Object.keys(active_context.datasets));
        selectSet = true;
      }
      var previousValue = select.getValue();
      if (previousValue) {
        select.setValue('(*clear*)');
        select.setValue(previousValue);
      }
    };

    var active_year = ui.Textbox({
      value: DEFAULTYEAR.toString(),
      style: {width: '200px'},
      onChange: yearChangeFunc
    });

    // Manually trigger the onChange event with the default value
    yearChangeFunc(DEFAULTYEAR.toString());

    var min_val = ui.Textbox(
      0, 0, function (value) {
        active_context.visParams.min = +(value);
        updateVisParams();
      });
    min_val.setDisabled(true);

    var max_val = ui.Textbox(
      100, 100, function (value) {
        active_context.visParams.max = +(value);
        updateVisParams();
      });
    max_val.setDisabled(true);

    active_context.point_val = ui.Textbox('nothing clicked');
    function updateVisParams() {
      if (active_context.last_layer !== null) {
        active_context.last_layer.setVisParams(active_context.visParams);
      }
    }
    active_context.updateVisParams = updateVisParams;
    select.setPlaceholder('Choose a dataset...');
    var range_button = ui.Button(
      'Detect Range', function (self) {
        self.setDisabled(true);
        var base_label = self.getLabel();
        self.setLabel('Detecting...');
        var mean_reducer = ee.Reducer.percentile([10, 90], ['p10', 'p90']);
        var meanDictionary = active_context.raster.reduceRegion({
          reducer: mean_reducer,
          geometry: active_context.map.getBounds(true),
          bestEffort: true,
        });
        ee.data.computeValue(meanDictionary, function (val) {
          min_val.setValue(val['B0_p10'], false);
          max_val.setValue(val['B0_p90'], true);
          self.setLabel(base_label)
          self.setDisabled(false);
        });
      });

    panel.add(ui.Label({
        value: 'Current Year',
        style:{'backgroundColor': 'rgba(0, 0, 0, 0)'}
      }));
    panel.add(active_year);
    panel.add(controls_label);
    panel.add(select);
    panel.add(ui.Label({
        value: 'min',
        style:{'backgroundColor': 'rgba(0, 0, 0, 0)'}
      }));
    panel.add(min_val);
    panel.add(ui.Label({
        value: 'max',
        style:{'backgroundColor': 'rgba(0, 0, 0, 0)'}
      }));
    panel.add(max_val);
    panel.add(range_button);
    panel.add(ui.Label({
      value: 'picked point',
      style: {'backgroundColor': 'rgba(0, 0, 0, 0)'}
    }));
    panel.add(active_context.point_val);
    panel_list.push([panel, min_val, max_val, active_context]);
    active_context.map.add(panel);

    function build_legend_panel() {
      var makeRow = function(color, name) {
        var colorBox = ui.Label({
          style: {
            backgroundColor: '#' + color,
            padding: '4px 25px 4px 25px',
            margin: '0 0 0px 0',
            position: 'bottom-center',
          }
        });
        var description = ui.Label({
          value: name,
          style: {
            margin: '0 0 0px 0px',
            position: 'top-center',
            fontSize: '10px',
            padding: 0,
            border: 0,
            textAlign: 'center',
            backgroundColor: 'rgba(0, 0, 0, 0)',
          }
        });

        return ui.Panel({
          widgets: [colorBox, description],
          layout: ui.Panel.Layout.Flow('vertical'),
          style: {
            backgroundColor: 'rgba(0, 0, 0, 0)',
          }
        });
      };

      var names = ['Low', '', '', '', 'High'];
      if (active_context.legend_panel !== null) {
        active_context.legend_panel.clear();
      } else {
        active_context.legend_panel = ui.Panel({
          layout: ui.Panel.Layout.Flow('horizontal'),
          style: {
            position: 'top-center',
            padding: '0px',
            backgroundColor: 'rgba(255, 255, 255, 0.4)'
          }
        });
        active_context.legend_select = ui.Select({
          items: Object.keys(legend_styles),
          placeholder: default_legend_style,
          onChange: function(key, self) {
            changeColorScheme(key, active_context);
        }});
        active_context.map.add(active_context.legend_panel);
      }
      active_context.legend_panel.add(active_context.legend_select);
      for (var i = 0; i<5; i++) {
        var row = makeRow(active_context.visParams.palette[i], names[i]);
        active_context.legend_panel.add(row);
      }
    }

    active_context.map.setControlVisibility(false);
    active_context.map.setControlVisibility({"mapTypeControl": true});
    build_legend_panel();
    active_context.build_legend_panel = build_legend_panel;
});

var clone_to_right = ui.Button(
  'Use this range in both windows', function () {
      panel_list[1][1].setValue(panel_list[0][1].getValue(), false)
      panel_list[1][2].setValue(panel_list[0][2].getValue(), true)
});
var clone_to_left = ui.Button(
  'Use this range in both windows', function () {
      panel_list[0][1].setValue(panel_list[1][1].getValue(), false)
      panel_list[0][2].setValue(panel_list[1][2].getValue(), true)
});

panel_list.forEach(function (panel_array) {
  var map = panel_array[3].map;
  map.onClick(function (obj) {
    var point = ee.Geometry.Point([obj.lon, obj.lat]);

    [panel_list[0][3], panel_list[1][3]].forEach(function (active_context) {
      if (active_context.last_layer !== null) {

        active_context.point_val.setValue('sampling...')
        var point_sample = active_context.raster.sampleRegions({
          collection: ee.FeatureCollection(point),
          geometries: true,
          scale: 10,
        });
        ee.data.computeValue(point_sample, function (val) {
          if (val.features.length > 0) {
            var feature = val.features[0];
            var properties = feature.properties;
            var firstPropertyKey = Object.keys(properties)[0];
            var firstPropertyValue = properties[firstPropertyKey];

            if (active_context.datasetName in discretePalettes) {
              var paletteInfo = discretePalettes[active_context.datasetName];
              var index = paletteInfo.values.indexOf(firstPropertyValue);
              if (index !== -1) {
                active_context.point_val.setValue(paletteInfo.labels[index]);
              } else {
                active_context.point_val.setValue('Unknown');
              }
            } else {
              active_context.point_val.setValue(firstPropertyValue.toString());
            }

            if (active_context.last_point_layer !== null) {
              active_context.map.remove(active_context.last_point_layer);
            }

            active_context.last_point_layer = active_context.map.addLayer(
              point, {'color': '#FF00FF'});
          } else {
            active_context.point_val.setValue('nodata');
          }
        });
      }
    })
  }
});

panel_list[0][0].add(clone_to_right);
panel_list[1][0].add(clone_to_left);
