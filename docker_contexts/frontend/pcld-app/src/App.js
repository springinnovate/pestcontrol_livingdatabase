import React, { useState, useEffect } from 'react';
import './App.css';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { MapContainer, TileLayer, CircleMarker, useMap, GeoJSON  } from 'react-leaflet';

const Dropdown = ({ id, values }) => (
  <select id={id}>
    {values.map((value, index) => (
      <option key={index} value={value}>
        {value}
      </option>
    ))}
  </select>
);

const FilterComponent = ({ seedData }) => {
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

function MapComponent({ mapCenter, markers }) {
  let opacity = 1;
  let bounds = [];
  if (markers.length > 0) {
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


function App() {
  const [serverStatus, setServerStatus] = useState();
  const [mapCenter, setMapCenter] = useState(null);
  const [markers, setMarkers] = useState([]);
  const [taskId, setTaskId] = useState(null);
  const [dropdownData, setDropdownData] = useState([]);

  useEffect(() => {
    fetch('/api/time').then(
      res => res.json()).then(
      data => {
        setServerStatus('');
      }).catch(
        setServerStatus('ERROR: backend server down, refresh this page and try again'));
  }, []);

  useEffect(() => {
    fetch('/api/searchable_fields')
      .then(response => response.json())
      .then(data => {
        // Assuming 'filterable_fields' is the key in the response
        const fields = data.filterable_fields;

        // Prepare dropdown data based on the fetched fields
        const seedData = fields.map((field, index) => ({
          id: field,
          values: ['todo']
        }));

        setDropdownData(seedData);
      })
      .catch(error => {
        console.error('Error fetching filterable fields:', error);
      });
  }, []);

  return (
    <div className="App">
      <div>
        <h1>SESYNC Query App</h1>
        <p>
          {serverStatus && serverStatus}
        </p>
      </div>
      <div>
        {dropdownData.map(dropdown => (
          <div key={dropdown.id}>
            <label htmlFor={dropdown.id}>{dropdown.id}</label>
            <select id={dropdown.id}>
              {dropdown.values.map((value, index) => (
                <option key={index} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </div>
        ))}
      </div>
      <div>
      <FilterComponent seedData={seedData} />;
      </div>
      <MapComponent
        mapCenter={mapCenter}
        markers={markers}
        />
    </div>
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


export default App;
