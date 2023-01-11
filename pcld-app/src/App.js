//import 'leaflet/dist/leaflet.css';
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
//import { MapContainer, TileLayer, useMap, Marker, Popup } from 'react-leaflet'

//import "bootstrap/dist/css/bootstrap.min.css";

function App() {
  const [currentTime, setCurrentTime] = useState(0);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileData, setFileData] = useState(null);

  /*useEffect(() => {
    setSelectedFile(event.target.files[0]);
  });
*/
  useEffect(() => {
    fetch('/time').then(res => res.json()).then(data => {
      setCurrentTime(data.time);
    });
  }, []);

  const onFileChange = event => {
    setSelectedFile(event.target.files[0]);
    const formData = new FormData();
    formData.append(
      "file",
      event.target.files[0]
    );
    //setFileData(selectedFile);

    // Request made to the backend api
    // Send formData object
    axios.post("/uploadfile", formData).then(
      res => {
        var data = res.data.data;
        setFileData(data);
      });
  };

  // On file upload (click the upload button)
  const onFileUpload = (state) => {
    // Details of the uploaded file
    const formData = new FormData();
    formData.append(
      "file",
      selectedFile
    );
    //setFileData(selectedFile);

    // Request made to the backend api
    // Send formData object
    axios.post("/uploadfile", formData).then(
      res => {
        var data = res.data.data;
        setFileData(data);
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
          <button onClick={onFileUpload}>
            Upload
          </button>
        </div>
        <p>{fileData}</p>
        <div id="map">
        {/*<MapContainer center={[51.505, -0.09]} zoom={13} scrollWheelZoom={false}>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <Marker position={[51.505, -0.09]}>
            <Popup>
              A pretty CSS3 popup. <br /> Easily customizable.
            </Popup>
          </Marker>
        </MapContainer>*/}
        </div>
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