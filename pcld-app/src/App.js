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
import { MapContainer, TileLayer, useMap } from 'react-leaflet'
//import "bootstrap/dist/css/bootstrap.min.css";

function MapControl({center}) {
  const map = useMap();
  map.setView(center, 10);
  return null;
}

function App() {
  const [currentTime, setCurrentTime] = useState(0);
  const [mapCenter, setMapCenter] = useState([51.0, 19.0]);
  useEffect(() => {
    fetch('/time').then(res => res.json()).then(data => {
      setCurrentTime(data.time);
    });

  }, []);

  const onFileChange = event => {
    const formData = new FormData();
    formData.append(
      "file",
      event.target.files[0]
    );
    // Request made to the backend api
    // Send formData object
    axios.post("/uploadfile", formData).then(
      res => {
        var data = res.data;
        setMapCenter(data.center);
      });
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Pest Control Database</h1>
        <p>
          System Time {currentTime}.
        </p>
        <div>
          <p>Upload CSV file</p>
          <input type="file" onChange={onFileChange} />
        </div>
        <p>Center: ({mapCenter[0]},{mapCenter[1]})</p>
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
    </MapContainer>
      </header>
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