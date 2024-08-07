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

var valid_years = $VALID_YEARS$;
var dataset_name = '$DATASET_NAME$';
var dataset_id = '$DATASET_ID$';
var band_id = '$BAND_ID$';
var image_only = $IMAGE_ONLY$;

function changeColorScheme(key, active_context) {
  active_context.visParams.palette = legend_styles[key];
  active_context.build_legend_panel();
  active_context.updateVisParams();
}

var panel_list = [];
var active_context = {
  'last_layer': null,
  'raster': null,
  'point_val': null,
  'last_point_layer': null,
  'map': Map,
  'legend_panel': null,
  'visParams': null,
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
    'position': "middle-left",
    'backgroundColor': 'rgba(255, 255, 255, 0.4)'
  }
});

var default_control_text = dataset_name;
var controls_label = ui.Label({
  value: default_control_text,
  style: {
    backgroundColor: 'rgba(0, 0, 0, 0)',
  }
});
var select = ui.Select({
  items: valid_years,
  onChange: function(year, self) {
      self.setDisabled(true);
      var original_value = self.getValue();
      self.setPlaceholder('loading ...');
      self.setValue(null, false);
      if (active_context.last_layer !== null) {
        active_context.map.remove(active_context.last_layer);
        min_val.setDisabled(true);
        max_val.setDisabled(true);
      }
      if (year == '') {
        self.setValue(original_value, false);
        self.setDisabled(false);
        return;
      }

      // fix this active_context.raster = ee.Image.loadGeoTIFF(valid_years[key]);
      var local_dataset_id = dataset_id.replace('{year}', year);
      if (image_only) {
        active_context.raster = ee.Image(local_dataset_id);
      } else {
          active_context.raster = ee.ImageCollection(local_dataset_id)
              .filter(ee.Filter.date(year+'-01-01', year+'-12-31'))
              .first();
      }
      active_context.raster = active_context.raster.select(band_id).rename('B0');
      Map.centerObject(active_context.raster);

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
select.setPlaceholder('Pick a valid year...');
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
      self.setLabel(base_label);
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

//panel_list.push([panel, min_val, max_val, map, active_context]);
panel_list.forEach(function (panel_array) {
  var map = panel_array[3].map;
  map.onClick(function (obj) {
    var point = ee.Geometry.Point([obj.lon, obj.lat]);
    [panel_list[0][3]].forEach(function (active_context) {
      if (active_context.last_layer !== null) {
        active_context.point_val.setValue('sampling...');
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
