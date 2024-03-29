var test_point = ee.Geometry.Point([-120.286646, 36.512154]);
var blob = test_point.buffer(8000);
var MODIS_YEAR = 2002;
var datasets = {
    '(*clear*)': '',
    'Greenup_1-modis-natural_MODIS_YEAR': ['projects/ecoshard-202922/assets/allbands', 'image', 'Greenup_1-modis-natural', MODIS_YEAR],
    'Greenup_1-modis-cultivated_MODIS_YEAR': ['projects/ecoshard-202922/assets/allbands', 'image', 'Greenup_1-modis-cultivated', MODIS_YEAR],
    'nat-pixel-prop_MODIS_YEAR': ['projects/ecoshard-202922/assets/allbands', 'image', 'natural-pixel-prop', MODIS_YEAR],
    'cult-pixel-prop_MODIS_YEAR': ['projects/ecoshard-202922/assets/allbands', 'image', 'cultivated-pixel-prop', MODIS_YEAR],
    'MODIS_Greenup_1_MODIS_YEAR': ['MODIS/006/MCD12Q2', 'imagecollection', 'Greenup_1', MODIS_YEAR],
    'MODIS_MidGreenup_1_MODIS_YEAR': ['MODIS/006/MCD12Q2', 'imagecollection', 'MidGreenup_1', MODIS_YEAR],
    'MODIS_Peak_1_MODIS_YEAR': ['MODIS/006/MCD12Q2', 'imagecollection', 'Peak_1', MODIS_YEAR],
    'MODIS_Maturity_1_MODIS_YEAR': ['MODIS/006/MCD12Q2', 'imagecollection', 'Maturity_1', MODIS_YEAR],
    'MODIS_MidGreendown_1_MODIS_YEAR': ['MODIS/006/MCD12Q2', 'imagecollection', 'MidGreendown_1', MODIS_YEAR],
    'MODIS_Senescence_1_MODIS_YEAR': ['MODIS/006/MCD12Q2', 'imagecollection', 'Senescence_1', MODIS_YEAR],
    'MODIS_Dormancy_1_MODIS_YEAR': ['MODIS/006/MCD12Q2', 'imagecollection', 'Dormancy_1', MODIS_YEAR],
    'MODIS_EVI_Minimum_1_MODIS_YEAR': ['MODIS/006/MCD12Q2', 'imagecollection', 'EVI_Minimum_1', MODIS_YEAR],
    'MODIS_EVI_Amplitude_1_MODIS_YEAR': ['MODIS/006/MCD12Q2', 'imagecollection', 'EVI_Amplitude_1', MODIS_YEAR],
    'MODIS_EVI_Area_1_MODIS_YEAR': ['MODIS/006/MCD12Q2', 'imagecollection', 'EVI_Area_1', MODIS_YEAR],
    'MODIS_QA_Overall_1_MODIS_YEAR': ['MODIS/006/MCD12Q2', 'imagecollection', 'QA_Overall_1', MODIS_YEAR],
    'MODIS/006/MCD12Q1/2001': ['MODIS/006/MCD12Q1', 'imagecollection', 'LC_Type2', 2001],
    'MODIS/006/MCD12Q1/2002': ['MODIS/006/MCD12Q1', 'imagecollection', 'LC_Type2', 2002],
    'MODIS/006/MCD12Q1/2003': ['MODIS/006/MCD12Q1', 'imagecollection', 'LC_Type2', 2003],
    'MODIS/006/MCD12Q1/2004': ['MODIS/006/MCD12Q1', 'imagecollection', 'LC_Type2', 2004],
    'MODIS/006/MCD12Q1/2005': ['MODIS/006/MCD12Q1', 'imagecollection', 'LC_Type2', 2005],
    'MODIS/006/MCD12Q1/2006': ['MODIS/006/MCD12Q1', 'imagecollection', 'LC_Type2', 2006],
    'MODIS/006/MCD12Q1/2007': ['MODIS/006/MCD12Q1', 'imagecollection', 'LC_Type2', 2007],
    'MODIS/006/MCD12Q1/2008': ['MODIS/006/MCD12Q1', 'imagecollection', 'LC_Type2', 2008],
    'MODIS/006/MCD12Q1/2009': ['MODIS/006/MCD12Q1', 'imagecollection', 'LC_Type2', 2009],
    'MODIS/006/MCD12Q1/2010': ['MODIS/006/MCD12Q1', 'imagecollection', 'LC_Type2', 2010],
    'MODIS/006/MCD12Q1/2011': ['MODIS/006/MCD12Q1', 'imagecollection', 'LC_Type2', 2011],
    'MODIS/006/MCD12Q1/2012': ['MODIS/006/MCD12Q1', 'imagecollection', 'LC_Type2', 2012],
    'MODIS/006/MCD12Q1/2013': ['MODIS/006/MCD12Q1', 'imagecollection', 'LC_Type2', 2013],
    'MODIS/006/MCD12Q1/2014': ['MODIS/006/MCD12Q1', 'imagecollection', 'LC_Type2', 2014],
    'MODIS/006/MCD12Q1/2015': ['MODIS/006/MCD12Q1', 'imagecollection', 'LC_Type2', 2015],
    'MODIS/006/MCD12Q1/2016': ['MODIS/006/MCD12Q1', 'imagecollection', 'LC_Type2', 2016],
    'USDA/NASS/CDL/1997': ['USDA/NASS/CDL', 'imagecollection', 'cropland', 1997],
    'USDA/NASS/CDL/1998': ['USDA/NASS/CDL', 'imagecollection', 'cropland', 1998],
    'USDA/NASS/CDL/1999': ['USDA/NASS/CDL', 'imagecollection', 'cropland', 1999],
    'USDA/NASS/CDL/2000': ['USDA/NASS/CDL', 'imagecollection', 'cropland', 2000],
    'USDA/NASS/CDL/2001': ['USDA/NASS/CDL', 'imagecollection', 'cropland', 2001],
    'USDA/NASS/CDL/2002': ['USDA/NASS/CDL', 'imagecollection', 'cropland', 2002],
    'USDA/NASS/CDL/2003': ['USDA/NASS/CDL', 'imagecollection', 'cropland', 2003],
    'USDA/NASS/CDL/2004': ['USDA/NASS/CDL', 'imagecollection', 'cropland', 2004],
    'USDA/NASS/CDL/2005': ['USDA/NASS/CDL', 'imagecollection', 'cropland', 2005],
    'USDA/NASS/CDL/2006': ['USDA/NASS/CDL', 'imagecollection', 'cropland', 2006],
    'USDA/NASS/CDL/2007': ['USDA/NASS/CDL', 'imagecollection', 'cropland', 2007],
    'USDA/NASS/CDL/2008': ['USDA/NASS/CDL', 'imagecollection', 'cropland', 2008],
    'USDA/NASS/CDL/2009': ['USDA/NASS/CDL', 'imagecollection', 'cropland', 2009],
    'USDA/NASS/CDL/2010': ['USDA/NASS/CDL', 'imagecollection', 'cropland', 2010],
    'USDA/NASS/CDL/2011': ['USDA/NASS/CDL', 'imagecollection', 'cropland', 2011],
    'USDA/NASS/CDL/2012': ['USDA/NASS/CDL', 'imagecollection', 'cropland', 2012],
    'USDA/NASS/CDL/2013': ['USDA/NASS/CDL', 'imagecollection', 'cropland', 2013],
    'USDA/NASS/CDL/2014': ['USDA/NASS/CDL', 'imagecollection', 'cropland', 2014],
    'USDA/NASS/CDL/2015': ['USDA/NASS/CDL', 'imagecollection', 'cropland', 2015],
    'USDA/NASS/CDL/2016': ['USDA/NASS/CDL', 'imagecollection', 'cropland', 2016],
    'USGS/NLCD_RELEASES/2016_REL/1992': ['USGS/NLCD_RELEASES/2016_REL', 'imagecollection', 'landcover', 1992],
    'USGS/NLCD_RELEASES/2016_REL/2001': ['USGS/NLCD_RELEASES/2016_REL', 'imagecollection', 'landcover', 2001],
    'USGS/NLCD_RELEASES/2016_REL/2004': ['USGS/NLCD_RELEASES/2016_REL', 'imagecollection', 'landcover', 2004],
    'USGS/NLCD_RELEASES/2016_REL/2006': ['USGS/NLCD_RELEASES/2016_REL', 'imagecollection', 'landcover', 2006],
    'USGS/NLCD_RELEASES/2016_REL/2008': ['USGS/NLCD_RELEASES/2016_REL', 'imagecollection', 'landcover', 2008],
    'USGS/NLCD_RELEASES/2016_REL/2011': ['USGS/NLCD_RELEASES/2016_REL', 'imagecollection', 'landcover', 2011],
    'USGS/NLCD_RELEASES/2016_REL/2013': ['USGS/NLCD_RELEASES/2016_REL', 'imagecollection', 'landcover', 2013],
    'USGS/NLCD_RELEASES/2016_REL/2016': ['USGS/NLCD_RELEASES/2016_REL', 'imagecollection', 'landcover', 2016],

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

function changeColorScheme(key, active_context) {
  active_context.visParams.palette = legend_styles[key];
  active_context.build_legend_panel();
  active_context.updateVisParams();
}

var linkedMap = ui.Map();
Map.setCenter(0, 0, 2);
Map.setCenter(test_point.getInfo().coordinates[0], test_point.getInfo().coordinates[1], 12);
var linker = ui.Map.Linker([ui.root.widgets().get(0), linkedMap]);
// Create a SplitPanel which holds the linked maps side-by-side.
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
      'last_layer': null,
      'raster': null,
      'point_val': null,
      'last_point_layer': null,
      'map': mapside[0],
      'legend_panel': null,
      'visParams': null,
    };

    active_context.map.style().set('cursor', 'crosshair');
    active_context.map.addLayer(test_point, {'color': '#FF0000'});
    active_context.map.addLayer(blob, {'color': '#FF0000'});
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
      items: Object.keys(datasets),
      onChange: function(key, self) {
          self.setDisabled(true);
          var original_value = self.getValue();
          self.setPlaceholder('loading ...');
          self.setValue(null, false);
          if (active_context.last_layer !== null) {
            active_context.map.remove(active_context.last_layer);
            min_val.setDisabled(true);
            max_val.setDisabled(true);
          }
          if (datasets[key] == '') {
            self.setValue(original_value, false);
            self.setDisabled(false);
            return;
          }
          var year = null;
          if (datasets[key][1] == 'imagecollection') {
            active_context.raster = ee.ImageCollection(
              datasets[key][0]);
            if (datasets[key].length == 4) { // has a date too
                year = datasets[key][3];
                active_context.raster = active_context.raster.filter(
                      ee.Filter.date(
                          year+'-01-01', year+'-12-31'));
            }
            active_context.raster = active_context.raster.select(datasets[key][2]).first().rename('B0');
          } else { //image
            active_context.raster = ee.Image(
              datasets[key][0]);
            if (datasets[key].length == 4) { // has a date too
                year = datasets[key][3];
                active_context.raster = active_context.raster.filter(
                      ee.Filter.date(
                          year+'-01-01', year+'-12-31'));
            }
            active_context.raster = active_context.raster.select(datasets[key][2]).rename('B0');
          }

          var mean_reducer = ee.Reducer.percentile([10, 90], ['p10', 'p90']);
          var meanDictionary = active_context.raster.reduceRegion({
            reducer: mean_reducer,
            geometry: active_context.map.getBounds(true),
            bestEffort: true,
          });

          ee.data.computeValue(meanDictionary, function (val) {
            active_context.visParams = {
              min: val['B0_p10'],
              max: val['B0_p90'],
              palette: active_context.visParams.palette,
            };
            active_context.last_layer = active_context.map.addLayer(
              active_context.raster, active_context.visParams);
            min_val.setValue(active_context.visParams.min, false);
            max_val.setValue(active_context.visParams.max, false);
            min_val.setDisabled(false);
            max_val.setDisabled(false);
            self.setValue(original_value, false);
            self.setDisabled(false);
          });
      }
    });

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
      // Add color and and names
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

//panel_list.push([panel, min_val, max_val, map, active_context]);
panel_list.forEach(function (panel_array) {
  var map = panel_array[3].map;
  map.onClick(function (obj) {
    var point = ee.Geometry.Point([obj.lon, obj.lat]);
    [panel_list[0][3], panel_list[1][3]].forEach(function (active_context) {
      if (active_context.last_layer !== null) {
        active_context.point_val.setValue('sampling...')
        var point_sample = active_context.raster.sampleRegions({
          collection: point,
          //scale: 10,
          //geometries: true
        });
        ee.data.computeValue(point_sample, function (val) {
          if (val.features.length > 0) {
            active_context.point_val.setValue(val.features[0].properties.B0.toString());
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
    });
  });
});

panel_list[0][0].add(clone_to_right);
panel_list[1][0].add(clone_to_left);
