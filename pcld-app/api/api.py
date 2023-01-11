import time
from flask import Flask
from flask import request
from werkzeug.utils import secure_filename
import pandas as pd
from io import StringIO
from shapely import MultiPoint
app = Flask(__name__)


@app.route('/time')
def get_current_time():
    return {'time': time.time()}

@app.route('/uploadfile', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      print(f'request: {request.files}')
      raw_data = request.files['file'].read().decode('utf-8')
      df = pd.read_csv(StringIO(raw_data))
      point_list = [(row[1][0], row[1][1]) for row in df[['lat', 'long']].iterrows()]
      points = MultiPoint(point_list)
      f = {
          'center': [points.centroid.x, points.centroid.y],
          'data': request.files['file'].read().decode('utf-8'),
          'points': point_list
          }
      print(f)
      return f
      # path = secure_filename(f.filename)
      # print(path)
      # f.save(path)
      # return 'file uploaded successfully'
