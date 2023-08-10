import React, { useState, useEffect } from 'react';
import axios from 'axios';
import FileSaver from 'file-saver';
import './App.css';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { MapContainer, TileLayer, CircleMarker, useMap, GeoJSON  } from 'react-leaflet';
import GeoRasterLayer from 'georaster-layer-for-leaflet';
import parse_georaster from 'georaster';
import chroma from 'chroma-js';
import JSZip from 'jszip';
import Papa from 'papaparse';

function formatList(list) {
  if (!Array.isArray(list)) {
    return list.toString();
  }

  return list.map(item => {
    if (Array.isArray(item)) {
      return `(${item.join(', ')})`;
    }
    return item;
  }).join(', ');
}

function CheckBoxWithDetails({ keyValue, bbsOverlap, dataset }) {
  const [detailsVisible, setDetailsVisible] = useState(false);

  const handleCheckboxChange = (e) => {
    setDetailsVisible(e.target.checked);
  };

  return (
    <React.Fragment key={keyValue}>
      <input
        type="checkbox"
        id={keyValue}
        name={keyValue}
        value={keyValue}
        disabled={!bbsOverlap}
        checked={bbsOverlap ? undefined : false}
        onChange={handleCheckboxChange}
      />
      <label htmlFor={keyValue}> {keyValue} {renderBoundingBox(dataset.bounds)} </label>
         <a href={dataset.viewer} className="spaced-link">[viewer]</a>
         <a href={dataset.documentation} className="spaced-link">[documentation]</a>
      {detailsVisible && (
        <div>
            <label>Natural:</label>
              <input type="text" name={keyValue + "_natural"} defaultValue={formatList(dataset.mask_types.natural)} />
              <br />
              <label>Cultivated:</label>
              <input type="text" name={keyValue + "_cultivated"} defaultValue={formatList(dataset.mask_types.cultivated)} />
        </div>
      )}
      <br />
    </React.Fragment>
  );
}

function SampleDataFetcher() {
  const [url, setUrl] = useState();
  const [buttonText, setButtonText] = useState("Select a file")
  const files = [
    {
      url: 'https://storage.googleapis.com/ecoshard-root/pestalytics_sample_data/cotton_site_info_2002_small.csv',
      name: '4 point file of cotton farms in CA in 1 year (cotton_site_info_2002_small.csv)'
    },
    {
      url: 'https://storage.googleapis.com/ecoshard-root/pestalytics_sample_data/cotton_site_info_2002-2008.csv',
      name: 'Multipoint/year file of cotton farms in CA (cotton_site_info_2002-2008.csv)'
    },
    {
      url: 'https://storage.googleapis.com/ecoshard-root/pestalytics_sample_data/fixed_scrubbed_plotdata.cotton.csv',
      name: 'Complex and data-dirty multipoint file of samples in Spain (fixed_scrubbed_plotdata.cotton.csv)'
    }];

  const handleDownload = () => {
    if (url) {
      window.open(url, '_blank');
      setButtonText("Downloaded!");
      setUrl(null);
    }
  };

  const handleChange = (e) => {
    const selectedFile = files.find(file => file.url === e.target.value);
    if (selectedFile) {
      setUrl(selectedFile.url);
      setButtonText("Download " + selectedFile.name);
    } else {
      setUrl(null);
    }
  };

  return (
    <div>
      Sample Data:
      <select onChange={handleChange}>
        <option value="">Select a file...</option>
        {files.map((file, index) => (
          <option key={index} value={file.url}>{file.name}</option>
        ))}
      </select><br/>
      <button
        onClick={handleDownload}
        disabled={!url}
      >
        {buttonText}
      </button>
    </div>
  );
}

function MapComponent({ mapCenter, markers, geoJsonStrList, rasterIds, rasterToRender }) {
  let opacity = 1;
  let color = ['#ffffff', '#ff0000'];
  let bounds = [];
  if (geoJsonStrList.length > 0) {
    geoJsonStrList.forEach((geoJsonStr) => {
      const geoJsonLayer = L.geoJson(geoJsonStr); // L is the leaflet instance
      bounds.push(geoJsonLayer.getBounds());
    });
  } else if (markers.length > 0) {
    let boundObj = calculateBoundingBoxAndPoints(markers, 0, 1)[0];
    bounds.push([
      [boundObj.minLat-1,
       boundObj.minLong-1],
      [boundObj.maxLat+1,
       boundObj.maxLong+1]
      ]);
  }

  if (bounds.length === 0) {
    bounds = [[-90, -180], [90, 180]];
  }
  return (
    <div>
      <MapContainer
        className="markercluster-map"
        zoom={4}
        maxZoom={18}>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
        />
        <MapControl bounds={bounds} />
        {rasterToRender &&
          <RasterLayers
            raster_id={rasterToRender.raster_id}
            raster={rasterToRender.raster}
            opacity={opacity}
            color={color} />}
        <BufferRegions key={Date.now()} geoJsonStrList={geoJsonStrList} opacity={opacity} />
        <LocationMarkers markers={markers} />
      </MapContainer>
    </div>
  );
}

const BufferRegions = ({ geoJsonStrList, opacity }) => {
  return (
    <>
      {geoJsonStrList.map((geojson_str, idx) => (
        <GeoJSON
          key={idx}
          data={geojson_str}
          style={(feature) => ({
            color: feature?.properties?.color || '#000000',
            opacity: 0.1,
            weight: 1
          })}
        />
      ))}
    </>
  );
};

function LocationMarkers({markers}) {
  return (
    <>
      {markers.map((point, index) => (
        <CircleMarker
          key={index}
          center={point}
          color='red'
          fillColor='red'
          radius={3}
          fillOpacity={1}
        />
      ))}
    </>
  );
}

function MapControl({bounds}) {
  const map = useMap();
  useEffect(() => {
    map.fitBounds(bounds);
  }, [bounds, map]);
  return null;
}

function InfoPanel({info}) {
  if (info !== null) {
    return (
      <div className="Info-panel">
      {info}
      </div>);
  } else {
    return (<div className="Info-panel"></div>);
  };
}

function renderBoundingBox(boundingBox) {
  return (
    <React.Fragment>
      Latitudes: {(boundingBox.minLat).toFixed(1)} to {(boundingBox.maxLat).toFixed(1)}, Longitudes: {(boundingBox.minLong).toFixed(1)} to {(boundingBox.maxLong).toFixed(1)}
    </React.Fragment>);
};

function AvailableDatsets({datasets, boundingBox}) {
  return (
    <React.Fragment>
      {String(datasets) === datasets ? datasets :
        Object.keys(datasets).map(keyValue => {
          let bbsOverlap = doBoundingBoxesOverlap(datasets[keyValue].bounds, boundingBox);
          return (
            <CheckBoxWithDetails
                keyValue={keyValue}
                bbsOverlap={bbsOverlap}
                dataset={datasets[keyValue]}
            />
          );
      })
    }
    </React.Fragment>
  );
};

function doBoundingBoxesOverlap(bbox1, bbox2) {
  try {
    return (
      bbox1.minLong <= bbox2.maxLong && bbox1.maxLong >= bbox2.minLong &&
      bbox1.minLat <= bbox2.maxLat && bbox1.maxLat >= bbox2.minLat);
  } catch (error) {
    return false;
  };
}

function CSVParser({
    setLatField, setLongField, setYearField, setHeaders, setTableData}) {
  const [file, setFile] = useState();

  const handleChange = (event) => {
    setFile(event.target.files[0]);
  };

  useEffect(() => {
    if (file) {
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: function(results) {
          const filteredData = results.data.filter(row => {
            return Object.values(row).some(
              value => value && value.trim() !== '');
          });
          setHeaders(results.meta.fields);
          setLatField(findMatch(
            results.meta.fields,
            ['lat', 'latitude', 'x']));
          setLongField(findMatch(
            results.meta.fields,
            ['lon', 'long', 'longitude', 'y']));
          setYearField(findMatch(results.meta.fields, ['.*year.*']));
          setTableData(filteredData);
        }
      });
    }
  }, [file, setHeaders, setLatField, setLongField, setYearField, setTableData]);

  const findMatch = (headers, matches) => {
    const regexMatches = matches.map(
      match => new RegExp(match, 'i')); // 'i' for case-insensitive
    for (let header of headers) {
      if (regexMatches.some(regex => regex.test(header))) {
        return header;
      }
    }
    return headers[0]; // Default to the first header if no match is found
  };

  return (
    <div>
    <label>
      {!file && (
        <>
          Upload Point Site data CSV, must include fields for latitude, longitude, and year of sample:
          <br />
        </>
      )}
      <input type="file" name="file" onChange={handleChange} accepts=".csv" />
    </label>
    </div>
  );
}

function RasterLayers({ raster_id, raster, opacity, color }) {
  const currentRasterLayer = React.useRef(null);
  const map = useMap();
  // Use an effect to create and add the layers when the rasterLayers, opacity or color changes
  useEffect(() => {
     if (currentRasterLayer.current) {
        map.removeLayer(currentRasterLayer.current);
      }

    const layer = new GeoRasterLayer({
      georaster: raster,
      opacity: opacity,
      pixelValuesToColorFn: pixelValues => chroma.scale(color)(pixelValues[0] / 255).hex(),
    });
    layer.addTo(map);
    currentRasterLayer.current = layer;
  }, [raster, map, opacity, color]);
};

function DownloadManager({csvData, csvFilename, rasterIds, taskId, setRasterToRender}) {
  const [selectedRasterId, setSelectedRasterId] = useState();
  const [downloadButtonText, setDownloadButtonText] = useState();
  const [rasterBlob, setRasterBlob] = useState();
  const handleSelection = (id) => {
    setSelectedRasterId(id);
    setRasterBlob(null);
    setDownloadButtonText("Download Raster " + id);
  };

  function saveRaster(rasterBlob, raster_id) {
    FileSaver.saveAs(rasterBlob, raster_id+'.tif');
  };

  function downloadRaster(task_id, raster_id) {
    axios.post("/api/download_raster/"+task_id+"/"+raster_id).then(res => {
      var data = res.data;
      var local_taskId = data.task_id;
      setRasterToRender(null);

      var pollTask = setInterval(function() {
        axios.get("/api/task/" + local_taskId).then(res => {
          var data = res.data;
          if (data.state === "SUCCESS" || data.state === "FAILURE") {
            clearInterval(pollTask);
            if (data.state === "SUCCESS") {
              (async () => {
                const url = data.result;
                const response = await axios.get(
                  url, { responseType: 'arraybuffer' });
                const jszip = new JSZip();
                const zip = await jszip.loadAsync(response.data);
                const file = Object.values(zip.files)[0];
                const arrayBuffer = await file.async('arraybuffer');
                const blob = new Blob([arrayBuffer], {type: 'image/tiff'}); // Replace 'image/tiff' with the actual MIME type of your raster file
                setRasterBlob(blob);
                const raster = await parse_georaster(arrayBuffer);
                setRasterToRender({raster_id: raster_id, raster: raster});
                setDownloadButtonText('Click to save '+raster_id+ ' to disk');
              })();
            } else {
              setDownloadButtonText("Error: " + data.status);
            }
          } else {
            setDownloadButtonText(
              "Downloading... running for " +
              parseFloat(data.time_running).toFixed(1) + "s");
          }
        }).catch(err => {
          if (err.response && err.response.data.data) {
            setDownloadButtonText(err.response.data.error);
          } else {
            setDownloadButtonText(err.message);
          }
        });
      }, 500);
    });
  };

  function handleCsvDownload() {
      FileSaver.saveAs(csvData, csvFilename);
  };
  if (!csvData) {
    return;
  }
  return (
    <React.Fragment>
      <p>Table complete, click button to download {csvFilename}</p>
      <br/>
      <button onClick={handleCsvDownload}>Download Table</button>
    <select value={selectedRasterId} onChange={(e) => handleSelection(e.target.value)}>
      <option value={null}>--Select a raster to download--</option>
      {rasterIds.map((raster_id) => (
        <option key={raster_id} value={raster_id}>
          {raster_id}
        </option>
      ))}
    </select>
    {selectedRasterId &&
      <button onClick={
        rasterBlob ?
          () => saveRaster(rasterBlob, selectedRasterId) :
          () => downloadRaster(taskId, selectedRasterId)
      }>{downloadButtonText}</button>
    }
    </React.Fragment>
  );
}

function calculateBoundingBoxAndPoints(rawData, latField, longField) {
  let bbox = {
      minLat: 90,
      maxLat: -90,
      minLong: 360,
      maxLong: -180,
  };
  var latLngPointArray = [];
  for (let point of rawData) {
    let lat = parseFloat(point[latField]);
    let long = parseFloat(point[longField]);
    if (isNaN(lat) || isNaN(long)) {
      continue;
    }
    if (lat < -90 || lat > 90 || long < -180 || long > 180) {
      continue;
    }
    latLngPointArray.push([lat, long]);
    if (lat < bbox.minLat) {
      bbox.minLat = lat;
    } else if (lat > bbox.maxLat) {
      bbox.maxLat = lat;
    }

    if (long < bbox.minLong) {
      bbox.minLong = long;
    } else if (long > bbox.maxLong) {
      bbox.maxLong = long;
    }
  }
  return [bbox, latLngPointArray];
}

function TableSubmitForm({
    availableDatasets,
    setDataInfo,
    setMapCenter,
    setMarkers,
    setGeoJsonStrList,
    setRasterIds,
    setCsvData,
    setCsvFilename,
    setTaskId,
    setFileUploaded
  }) {
  const [formActive, setFormActive] = useState(true);
  const [submitButtonText, setSubmitButtonText] = useState("Click to process");
  const [yearField, setYearField] = useState(null);
  const [longField, setLongField] = useState(null);
  const [latField, setLatField] = useState(null);
  const [headers, setHeaders] = useState([]);
  const [tableData, setTableData] = useState(null);
  const [boundingBox, setBoundingBox] = useState(null);

  useEffect(() => {
    if (longField === null || latField === null || tableData === null) {
      return;
    }
    let bbox = null;
    let latLngPointArray = null;
    [bbox, latLngPointArray] = calculateBoundingBoxAndPoints(tableData, latField, longField);
    if (latLngPointArray.length > 0) {
      setBoundingBox(bbox);
      setMarkers(latLngPointArray);
      setGeoJsonStrList([]);
    }
    setFileUploaded(true);
  }, [longField, latField, tableData, setMarkers, setFileUploaded, setGeoJsonStrList]);

  function processCompletedData(data, time_running) {
    setRasterIds(data.band_ids);
    setMapCenter(data.center);
    setMarkers(data.points);
    //setGeoJsonStrList(data.points);
    setGeoJsonStrList(data.geojson_str_list);
    const csvData = new Blob(
      [data.csv_blob_result], { type: 'text/csv;charset=utf-8;' });
    setCsvData(csvData);
    setCsvFilename(data.csv_filename);
    setDataInfo("Complete in " + time_running + "s! Click to download CSV table.");
    setSubmitButtonText("Click to process");
    setFormActive(true);
  };

  function handleSubmit(event) {
    event.preventDefault();
    setDataInfo("processing, please wait")
    setFormActive(false);
    setSubmitButtonText("processing, please wait");
    // Read the form data
    const form = event.target;
    const formData = new FormData(form);
    var datasets_to_process = [];
    for (var key in availableDatasets) {
      var value = availableDatasets[key];
      if (value !== null) {
        if (form[key].checked) {
          var naturalValue = "["+form[key + "_natural"].value+"]"; // assuming you've named the input fields accordingly
          var cultivatedValue = "["+form[key + "_cultivated"].value+"]";
          datasets_to_process.push({
            key: key,
            natural: naturalValue,
            cultivated: cultivatedValue
          });
        }
      }
    };
    formData.append('datasets_to_process', JSON.stringify(datasets_to_process));
    for (const [name,value] of formData) {
      console.log(name, ":", value)
    }

    // Request made to the backend api
    // Send formData object
    axios.post("/api/uploadfile", formData).then(res => {
      var data = res.data;
      var taskId = data.task_id;
      setTaskId(taskId);

      var pollTask = setInterval(function() {
        axios.get("/api/task/" + taskId).then(res => {
          var data = res.data;
          if (data.state === "SUCCESS" || data.state === "FAILURE") {
            clearInterval(pollTask);
            if (data.state === "SUCCESS") {
              // Handle successful completion here
              processCompletedData(
                data.result,
                parseFloat(data.time_running).toFixed(1));
              console.log("Task completed successfully");
            } else {
              setDataInfo("Error: " + data.status);
              setSubmitButtonText("Click to process");
              setFormActive(true);
            }
          } else {
            setDataInfo(
              "Task running for " +
              parseFloat(data.time_running).toFixed(1) + "s");
          }
        }).catch(err => {
          if (err.response && err.response.data.data) {
            setDataInfo(err.response.data.error);
          } else {
            setDataInfo(err.message);
          }
          setSubmitButtonText("error, try again");
          setFormActive(true);
        });
      }, 500);
    });
  };

  return (
    <form method="post" onSubmit={handleSubmit}>
      <hr/>
      <CSVParser
        setLatField={setLatField}
        setLongField={setLongField}
        setYearField={setYearField}
        setHeaders={setHeaders}
        setTableData={setTableData}
      />
      <div>
      {headers.length > 0 ? (
        <>
          <h3>Verify the following are the correct lat/long/year fields in your CSV:</h3>
          <label>Longitude field:
            <select value={longField} name="long_field" onChange={e => setLongField(e.target.value)}>
              {headers.map((header, i) => (
                <option key={i} value={header}>{header}</option>
              ))}
            </select>
          </label>
          <label>Latitude field:
            <select value={latField} name="lat_field" onChange={e => setLatField(e.target.value)}>
              {headers.map((header, i) => (
                <option key={i} value={header}>{header}</option>
              ))}
            </select>
          </label>
          <label>Year field:
            <select value={yearField} name="year_field" onChange={e => setYearField(e.target.value)}>
              {headers.map((header, i) => (
                <option key={i} value={header}>{header}</option>
              ))}
            </select>
          </label>
        <br/>
        <hr/>
        </>
        ) : ""
      }
       {headers.length ? (
        <>
          <h3>(optional) Choose Landcover Datasets to mask the <a href="https://developers.google.cn/earth-engine/datasets/catalog/MODIS_061_MCD12Q2#bands">MODIS phenological variables</a> by "cultivated" vs "natural":</h3>
            <AvailableDatsets datasets={availableDatasets} boundingBox={boundingBox}/>
            <p>
            <label>search buffer_size around sample points(m):
              <input type="number" name="buffer_size" defaultValue="5000"/>
            </label>
            </p>
            <br/>
            <button type="submit" disabled={!formActive}>{submitButtonText}</button><br/>
          <hr/>
        </>
      ) : ""
      }
      </div>
    </form>
  );
}

function App() {
  const [serverStatus, setServerStatus] = useState();
  const [availableDatasets, setAvailableDatasets] = useState({'server error, please reload': null});
  const [dataInfo, setDataInfo] = useState(null);
  const [mapCenter, setMapCenter] = useState(null);
  const [markers, setMarkers] = useState([]);
  const [geoJsonStrList, setGeoJsonStrList] = useState([]);
  const [rasterIds, setRasterIds] = useState([]); // New state for raster URLs
  const [csvData, setCsvData] = useState(null);
  const [csvFilename, setCsvFilename] = useState(null);
  const [taskId, setTaskId] = useState(null);
  const [rasterToRender, setRasterToRender] = useState();
  const [fileUploaded, setFileUploaded] = useState();

  useEffect(() => {
    fetch('/api/time').then(
      res => res.json()).then(
      data => {
        setServerStatus('');
      }).catch(
        setServerStatus('ERROR: backend server down, refresh this page and try again'));
  }, []);


  useEffect(() => {
    fetch('/api/available_datasets').then(res => res.json()).then(data => {
      setAvailableDatasets(data);
    });

  }, []);

  return (
    <div className="App">
      <div>
        <h1>Pestalytics - Phenological Sampling App</h1>
        <p>
          {serverStatus && serverStatus}
        </p>
        { !fileUploaded && <SampleDataFetcher /> }
      </div>
      <TableSubmitForm
        availableDatasets={availableDatasets}
        setDataInfo={setDataInfo}
        setMapCenter={setMapCenter}
        setMarkers={setMarkers}
        setGeoJsonStrList={setGeoJsonStrList}
        setRasterIds={setRasterIds}
        setCsvData={setCsvData}
        setCsvFilename={setCsvFilename}
        setTaskId={setTaskId}
        setFileUploaded={setFileUploaded}
        />
      <InfoPanel info={dataInfo}/>
      <br/>
      <DownloadManager
        csvData={csvData}
        csvFilename={csvFilename}
        rasterIds={rasterIds}
        taskId={taskId}
        setRasterToRender={setRasterToRender}
        />
      <MapComponent
        mapCenter={mapCenter}
        markers={markers}
        geoJsonStrList={geoJsonStrList}
        rasterIds={rasterIds}
        rasterToRender={rasterToRender}
        />
    </div>
  );
}

export default App;
