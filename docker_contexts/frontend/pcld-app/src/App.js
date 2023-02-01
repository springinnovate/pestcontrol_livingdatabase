/*import {
  MapContainer,
  TileLayer,
  useMap,
  Marker,
  Popup
} from 'https://cdn.esm.sh/react-leaflet'*/
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { MapContainer, TileLayer, useMap, Marker } from 'react-leaflet';
//import "bootstrap/dist/css/bootstrap.min.css";

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
      {
        Object.entries(info)
        .map( ([key, value]) => <p>{key} -- {value}</p> )
      }
      </div>);
  } else {
    return (<div className="Info-panel"><p>Load a csv.</p></div>);
  };
}

function App() {
  const [currentTime, setCurrentTime] = useState(0);
  const [availableDatasets, setAvailableDatasets] = useState(0);
  const [dataInfo, setDataInfo] = useState(null);
  const [mapCenter, setMapCenter] = useState(null);
  const [markers, setMarkers] = useState([]);
  const [formProcessing, setFormProcessing] = useState(false);
  const [submitButtonText, setSubmitButtonText] = useState("Submit form");

  function TableSubmitForm() {
    function handleSubmit(event) {
      event.preventDefault();
      setFormProcessing(true);
      setSubmitButtonText("processing, please wait");
      // Read the form data
      const form = event.target;
      const formData = new FormData(form);

      // Request made to the backend api
      // Send formData object
      axios.post("/uploadfile", formData).then(
        res => {
          var data = res.data;
          setDataInfo(data.info);
          setMapCenter(data.center);
          setMarkers(data.points);
          setFormProcessing(false);
          setSubmitButtonText("Submit form");
        });
    };

    return (
      <form method="post" onSubmit={handleSubmit}>
        <hr/>
        <label>Select sample CSV:
          <input type="file" name="file"/>
        </label><br/>
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
        <hr/>
      </form>
    );
  }

  useEffect(() => {
    fetch('/time').then(
      res => res.json()).then(
      data => {
        setCurrentTime(data.time);
      })
  }, []);


  useEffect(() => {
    fetch('/available_datasets').then(res => res.json()).then(data => {
      var dataset_str = '';
      for (var key in data) {
        dataset_str += key + ' ';
      }
      setAvailableDatasets(dataset_str);
    });

  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Pest Control Database</h1>
        <p>
          {currentTime}
        </p>
        <p>
          Available datasets: {availableDatasets}
        </p>
      </header>
      <TableSubmitForm />
      <InfoPanel info={dataInfo}/>
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
    </MapContainer>
    </div>
  );
}

export default App;

/*import axios from 'axios';

import React,{Component} from 'react';

class App extends Component {

  state = {

  // Initially, no file is selected
  selectedFile: null
  };

  // On file select (from the pop up)
  onFileChange = event => {

  // Update the state
  this.setState({ selectedFile: event.target.files[0] });

  };

  // On file upload (click the upload button)
  onFileUpload = () => {

  // Create an object of formData
  const formData = new FormData();

  // Update the formData object
  formData.append(
    "myFile",
    this.state.selectedFile,
    this.state.selectedFile.name
  );

  // Details of the uploaded file
  console.log(this.state.selectedFile);

  // Request made to the backend api
  // Send formData object
  axios.post("api/uploadfile", formData);
  };

  // File content to be displayed after
  // file upload is complete
  fileData = () => {

  if (this.state.selectedFile) {

    return (
    <div>
      <h2>File Details:</h2>
      <p>File Name: {this.state.selectedFile.name}</p>

      <p>File Type: {this.state.selectedFile.type}</p>

      <p>
      Last Modified:{" "}
      {this.state.selectedFile.lastModifiedDate.toDateString()}
      </p>

    </div>
    );
  } else {
    return (
    <div>
      <br />
      <h4>Choose before Pressing the Upload button</h4>
    </div>
    );
  }
  };

  render() {

  return (
    <div>
      <h1>
      GeeksforGeeks
      </h1>
      <h3>
      File Upload using React!
      </h3>
      <div>
        <input type="file" onChange={this.onFileChange} />
        <button onClick={this.onFileUpload}>
        Upload!
        </button>
      </div>
    {this.fileData()}
    </div>
  );
  }
}

export default App;
*/