/*import {
  MapContainer,
  TileLayer,
  useMap,
  Marker,
  Popup
} from 'https://cdn.esm.sh/react-leaflet'*/
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import FileSaver from 'file-saver';
import './App.css';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { MapContainer, TileLayer, useMap, Marker } from 'react-leaflet';
import GeoRasterLayer from 'georaster-layer-for-leaflet';
import parse_georaster from 'georaster';
import Slider from 'react-input-slider';
import chroma from 'chroma-js';
import JSZip from 'jszip';
import Papa from 'papaparse';
//import "bootstrap/dist/css/bootstrap.min.css";

function MapComponent({ mapCenter, markers, rasterUrls }) {
  const [availableRasterIdList, setAvailableRasterIdList] = useState([]);
  const [rastersToRender, setRastersToRender] = useState([]);
  const [opacity, setOpacity] = useState(1);
  const [color, setColor] = useState(['#ffffff', '#ff0000']); // white to red gradient
  const [selectedRaster, setSelectedRaster] = useState(null);

  useEffect(() => {
    const loadRasters = async () => {
      const id_raster_list = [];
      for (const [url_id, url] of rasterUrls) {
        const response = await axios.get(url, { responseType: 'arraybuffer' });
        const jszip = new JSZip();
        const zip = await jszip.loadAsync(response.data);
        const file = Object.values(zip.files)[0];
        const arrayBuffer = await file.async('arraybuffer');
        const georaster = await parse_georaster(arrayBuffer);
        id_raster_list.push([url_id, georaster, url]);
      }
      setAvailableRasterIdList(id_raster_list);
    };
    loadRasters();
  }, [rasterUrls]);

  return (
    <div>
      <div>
        {availableRasterIdList.map(([raster_id, raster, url]) =>
          <div>
            <label key={raster_id}>
              <input
                type="radio"
                value={raster_id}
                checked={selectedRaster === raster_id}
                onChange={() => {
                  setSelectedRaster(raster_id);
                  setRastersToRender([[raster_id, raster]]);
                }}
              />
              <a href={url}>{raster_id}</a>
            </label>
          </div>
        )}
      </div>
      <Slider
        axis="x"
        x={opacity * 100}
        onChange={({ x }) => setOpacity(x / 100)}
      />
      <button onClick={() => setColor(['#ffffff', '#ff0000'])}>Red</button>
      <button onClick={() => setColor(['#ffffff', '#0000ff'])}>Blue</button>
      <MapContainer
        className="markercluster-map"
        center={mapCenter}
        zoom={4}
        maxZoom={18}>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
        />
        <MapControl center={mapCenter} />
        <LocationMarkers markers={markers} />
        {rastersToRender.map(([raster_id, raster]) =>
          <RasterLayers raster_id={raster_id} raster={raster} opacity={opacity} color={color} />)}
      </MapContainer>
    </div>
  );
}


function LocationMarkers({markers}) {
  const covidIcon = L.icon({
      iconUrl: 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Redpoint.svg/768px-Redpoint.svg.png',

      iconSize:     [4, 4], // size of the icon
      iconAnchor:   [0, 0], // point of the icon which will correspond to marker's location
      popupAnchor:  [0, 0] // point from which the popup should open relative to the iconAnchor
  });
  return (
    <React.Fragment>
      {markers.map(coord => <Marker
        position={[coord[1], coord[2]]}
        icon={covidIcon}
        key={coord[0]}
        ></Marker>)}
    </React.Fragment>
    );
}

function MapControl({center}) {
  const map = useMap();
  if (center !== null) {
    map.setView(center, 8);
  }
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

function AvailableDatsets({datasets}) {
  return (
    <React.Fragment>
      {String(datasets) === datasets ? datasets :
        Object.keys(datasets).map(key => {
          return (
            <React.Fragment key={key}>
            <label>
            <input type="checkbox" id={key} name={key} value={key}/>
            {key}
            </label>
            <br/>
          </React.Fragment>
      )})}
    </React.Fragment>
  );
};

function CSVParser() {
  const [file, setFile] = useState();
  const [headers, setHeaders] = useState([]);

  const handleChange = (event) => {
    setFile(event.target.files[0]);
  };

  useEffect(() => {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: function(results) {
        setHeaders(results.meta.fields);
      }
    });
  }, [file]);

  return (
    <div>
      <input type="file" name="file" onChange={handleChange} accepts=".csv"/>
      <div>
        <h3>Headers:</h3>
        {headers.map((header, i) => (
          <p key={i}>{header}</p>
        ))}
      </div>
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

function App() {
  const [currentTime, setCurrentTime] = useState(0);
  const [availableDatasets, setAvailableDatasets] = useState({'server error, please reload': null});
  const [dataInfo, setDataInfo] = useState(null);
  const [mapCenter, setMapCenter] = useState(null);
  const [markers, setMarkers] = useState([]);
  const [formProcessing, setFormProcessing] = useState(false);
  const [submitButtonText, setSubmitButtonText] = useState("Submit form");
  const [serverUp, setServerUp] = useState(false);
  const [rasterUrls, setRasterUrls] = useState([]); // New state for raster URLs
  const [longField, setLongField] = useState(null);
  const [latField, setLatField] = useState(null);
  const [yearField, setYearField] = useState(null);
  const [bufferSize, setBufferSize] = useState(null);

  function TableSubmitForm({availableDatasets}) {
    function handleSubmit(event) {
      event.preventDefault();
      setDataInfo("processing, please wait")
      setFormProcessing(true);
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
      axios.post("/uploadfile", formData).then(
        async res => {
          var data = res.data;
          const urls = [];
          for (const raster_id in data.url_by_header_id) {
            if (data.url_by_header_id.hasOwnProperty(raster_id)) {
              urls.push([raster_id, data.url_by_header_id[raster_id]]);
            }
          }
          setRasterUrls(urls);
          setMapCenter(data.center);
          setMarkers(data.points);
          const csvData = new Blob(
            [data.csv_blob_result], { type: 'text/csv;charset=utf-8;' });
          FileSaver.saveAs(csvData, data.csv_filename);
          setDataInfo("Success! Result saved to download folder as: '" + data.csv_filename + "'");
          setFormProcessing(false);
          setSubmitButtonText("Submit form");
        }).catch(err => {
          setDataInfo("SERVER ERROR")
          setFormProcessing(true);
          setSubmitButtonText("SERVER ERROR");
          console.log(err.message);
        });
    };

    return (
      <form method="post" onSubmit={handleSubmit}>
        <hr/>
        <label>Select sample CSV:
          <CSVParser />
        </label><br/>
        <hr/>
         {longField && latField && yearField && bufferSize ? (
          <>
            <p>Choose Datasets:</p>
              <AvailableDatsets datasets={availableDatasets} />
              <hr/>
              <p>Edit any other desired fields:</p>
              <label>long_field:
                <input type="text" name="long_field" defaultValue="long"/>
              </label><br/>
              <label>lat_field:
                <input type="text" name="lat_field" defaultValue="lat"/>
              </label><br/>
              <label>year_field:
                <input type="text" name="year_field" defaultValue="crop_year"/>
              </label><br/>
              <label>buffer_size (m):
                <input type="number" name="buffer_size" defaultValue="30000"/>
              </label><br/>
              <button type="submit" disabled={formProcessing}>{submitButtonText}</button><br/>
              <button type="reset" disabled={formProcessing}>Reset form</button>
            </>
        ) : (
          <p>Please select your CSV</p>
        )}
        <hr/>
      </form>
    );
  }

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
      <TableSubmitForm availableDatasets={availableDatasets} />
      <InfoPanel info={dataInfo}/>
      <MapComponent mapCenter={mapCenter} markers={markers} rasterUrls={rasterUrls} />
    </div>
  );
}

export default App;
