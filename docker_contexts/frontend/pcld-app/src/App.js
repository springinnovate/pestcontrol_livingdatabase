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

const Dropdown = ({ id, values }) => (
  <select id={id}>
    {values.map((value, index) => (
      <option key={index} value={value}>
        {value}
      </option>
    ))}
  </select>
);

const DropdownContainer = ({ seedData }) => {
  useEffect(() => {
    // Any additional effects can be handled here
  }, [seedData]);

  return (
    <div>
      {seedData.map((data, index) => (
        <Dropdown key={index} id={data.id} values={data.values} />
      ))}
    </div>
  );
};

const seedData = [
  {
    id: 'dropdown1',
    values: ['Option 1', 'Option 2', 'Option 3']
  },
  {
    id: 'dropdown2',
    values: ['Choice A', 'Choice B', 'Choice C']
  }
];

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
          attribution='&copy; <a target="_blank" rel="noopener noreferrer" href="http://osm.org/copyright">OpenStreetMap</a> contributors'
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
      (Latitudes: {(boundingBox.minLat).toFixed(1)} to {(boundingBox.maxLat).toFixed(1)}, Longitudes: {(boundingBox.minLong).toFixed(1)} to {(boundingBox.maxLong).toFixed(1)})
    </React.Fragment>);
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
        <h1>SESYNC Query App</h1>
        <p>
          {serverStatus && serverStatus}
        </p>
        { !fileUploaded && <SampleDataFetcher /> }
      </div>
      <div>
      <DropdownContainer seedData={seedData} />;
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
