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


function MapComponent({ mapCenter, markers, geoJsonStrList, rasterIds, rasterToRender }) {
  const [availableRasterIdList, setAvailableRasterIdList] = useState([]);
  const [opacity, setOpacity] = useState(1);
  const [color, setColor] = useState(['#ffffff', '#ff0000']); // white to red gradient
  const [selectedRaster, setSelectedRaster] = useState(null);

/*  useEffect(() => {
    setAvailableRasterIdList(rasterIds);
  }, [rasterIds]);*/

  // Calculate the bounds here
  let bounds = [];
  geoJsonStrList.forEach((geoJsonStr) => {
    const geoJsonLayer = L.geoJson(geoJsonStr); // L is the leaflet instance
    bounds.push(geoJsonLayer.getBounds());
  });

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
          center={[point[1], point[2]]}
          color='red'
          fillColor='red'
          radius={3}
          fillOpacity={1}
          key={point[0]}
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
    return (<div className="Info-panel"><p>Load a csv.</p></div>);
  };
}

function renderBoundingBox(boundingBox) {
  return (
    <React.Fragment>
      Latitudes: {boundingBox.minLat} to {boundingBox.maxLat}, Longitudes: {boundingBox.minLong} to {boundingBox.maxLong}
    </React.Fragment>);
};

function AvailableDatsets({datasets, boundingBox}) {
  return (
    <React.Fragment>
      {String(datasets) === datasets ? datasets :
        Object.keys(datasets).map(key => {
          let bbsOverlap = doBoundingBoxesOverlap(datasets[key].bounds, boundingBox);
          return (
            <React.Fragment key={key}>
            <label
              htmlFor={key}
              className={!bbsOverlap ? "disabled" : ""}
            >
              <input
                type="checkbox"
                id={key}
                name={key}
                value={key}
                disabled={!bbsOverlap}
              />
              {key} {renderBoundingBox(datasets[key].bounds)}
            </label>
            <br/>
          </React.Fragment>
      )})}
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
          setHeaders(results.meta.fields);
          setLatField(findMatch(
            results.meta.fields,
            ['lat', 'latitude', 'x']));
          setLongField(findMatch(
            results.meta.fields,
            ['lon', 'long', 'longitude', 'y']));
          setYearField(findMatch(results.meta.fields, ['.*year.*']));
          setTableData(results.data);
          // TODO: at this point we could parse through the points to find
          // TODO: the bounding box and make sure it intersects with each
          // TODO: dataset bounding box?
        }
      });
    }
  }, [file, setHeaders, setLatField, setLongField, setYearField]);

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
    <label>Select sample CSV:
      <input type="file" name="file" onChange={handleChange} accepts=".csv">
      </input>
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
  const [selectedRaster, setSelectedRaster] = useState();
  const [downloadButtonText, setDownloadButtonText] = useState();
  const handleSelection = (id) => {
    setSelectedRaster(id);
    setDownloadButtonText("Download Raster " + id);
  };

  function downloadRaster(task_id, raster_id) {
    axios.post("/download_raster/"+task_id+"/"+raster_id).then(res => {
      var data = res.data;
      var local_taskId = data.task_id;

      var pollTask = setInterval(function() {
        axios.get("/task/" + local_taskId).then(res => {
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
                const raster = await parse_georaster(arrayBuffer);
                setRasterToRender({raster_id: raster_id, raster: raster});
              })();
            } else {
              console.error("Error: " + data.status);
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
    <select value={selectedRaster} onChange={(e) => handleSelection(e.target.value)}>
      <option value={null}>--Select a raster to download--</option>
      {rasterIds.map((raster_id) => (
        <option key={raster_id} value={raster_id}>
          {raster_id}
        </option>
      ))}
    </select>
    {selectedRaster &&
      <button onClick={() => downloadRaster(taskId, selectedRaster)}>{downloadButtonText}</button>
    }

    {/*{availableRasterIdList.map(([raster_id, raster, url]) =>
      <div>
        <label key={raster_id}>
          <input
            type="radio"
            value={raster_id}
            checked={selectedRaster === raster_id}
            onChange={() => {
              setSelectedRaster(raster_id);
              setRasterToRender([[raster_id, raster]]);
            }}
          />
          <a href={url}>{raster_id}</a>
        </label>
      </div>
    )}*/}
    </React.Fragment>

  );
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
    setTaskId
  }) {
  const [formActive, setFormActive] = useState(true);
  const [submitButtonText, setSubmitButtonText] = useState("Submit form");
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
    let bbox = {
      minLat: parseFloat(tableData[0][latField]),
      maxLat: parseFloat(tableData[0][latField]),
      minLong: parseFloat(tableData[0][longField]),
      maxLong: parseFloat(tableData[0][longField]),
    };

    // Iterate over the tableData updating the bounding box.
    for (let point of tableData) {
      let lat = parseFloat(point[latField]);
      let long = parseFloat(point[longField]);

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
    setBoundingBox(bbox);
  }, [longField, latField, tableData]);
  function processCompletedData(data, time_running) {
    /*const urls = [];
    for (const raster_id in data.band_and_bounds_by_id) {
      if (data.band_and_bounds_by_id.hasOwnProperty(raster_id)) {
        urls.push([raster_id, data.band_and_bounds_by_id[raster_id]]);
      }
    }*/
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
    setSubmitButtonText("Submit form");
    //setFormActive(true);
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
          datasets_to_process.push(key);
        }
      }
    };
    formData.append('datasets_to_process', datasets_to_process);
    for (const [name,value] of formData) {
      console.log(name, ":", value)
    }

    // Request made to the backend api
    // Send formData object
    axios.post("/uploadfile", formData).then(res => {
      var data = res.data;
      var taskId = data.task_id;
      setTaskId(taskId);

      var pollTask = setInterval(function() {
        axios.get("/task/" + taskId).then(res => {
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
              console.error("Error: " + data.status);
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
            <select value={latField} name="lat_field" onChange={e => setLatField(e.target.value)}>
              {headers.map((header, i) => (
                <option key={i} value={header}>{header}</option>
              ))}
            </select>
          </label>
          <label>Latitude field:
            <select value={longField} name="long_field" onChange={e => setLongField(e.target.value)}>
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
        </>) : (
        <p>load a csv...</p>
        )
      }
      </div>
      <br/>
      <hr/>
       {headers.length ? (
        <>
          <p>Choose Datasets:</p>
            <AvailableDatsets datasets={availableDatasets} boundingBox={boundingBox}/>
            <hr/>
            <p>Edit any other desired fields:</p>
            <label>buffer_size (m):
              <input type="number" name="buffer_size" defaultValue="5000"/>
            </label><br/>
            <button type="submit" disabled={!formActive}>{submitButtonText}</button><br/>
          </>
      ) : (
        <p>Please select your CSV</p>
      )}
      <hr/>
    </form>
  );
}

function App() {
  const [currentTime, setCurrentTime] = useState(0);
  const [availableDatasets, setAvailableDatasets] = useState({'server error, please reload': null});
  const [serverUp, setServerUp] = useState(false);
  const [dataInfo, setDataInfo] = useState(null);
  const [mapCenter, setMapCenter] = useState(null);
  const [markers, setMarkers] = useState([]);
  const [geoJsonStrList, setGeoJsonStrList] = useState([]);
  const [rasterUrls, setRasterUrls] = useState([]); // New state for raster URLs
  const [rasterIds, setRasterIds] = useState([]); // New state for raster URLs
  const [csvData, setCsvData] = useState(null);
  const [csvFilename, setCsvFilename] = useState(null);
  const [taskId, setTaskId] = useState(null);
  const [rasterToRender, setRasterToRender] = useState();

  useEffect(() => {
    fetch('/time').then(
      res => res.json()).then(
      data => {
        setCurrentTime(data.time);
        setServerUp(true);
      }).catch(
        setCurrentTime('ERROR: backend server down, refresh this page and try again'))
  }, []);


  useEffect(() => {
    fetch('/available_datasets').then(res => res.json()).then(data => {
      setAvailableDatasets(data);
    });

  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Pest Control Pheno Variable App</h1>
        <p>
          {serverUp && currentTime}
        </p>
      </header>
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
        />
      <InfoPanel info={dataInfo}/>
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

console.log("{'minLat': "+'-90'+", 'maxLat': "+'90'+", 'minLong': "+'-90'+", 'maxLong': "+'90'+"}");

export default App;
