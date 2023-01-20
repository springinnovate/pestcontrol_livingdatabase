import os
import sys
import time
from flask import Flask
from flask import request
from werkzeug.utils import secure_filename
import pandas as pd
from io import StringIO
from shapely.geometry import MultiPoint
import logging

app = Flask(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)


LOGGER = logging.getLogger()


def create_app(config=None):
    """Create the Geoserver STAC Flask app."""
    LOGGER.debug('starting up!')

    app = Flask(__name__)
    @app.route('/time')
    def get_current_time():
        return {'time': time.time()}

    @app.route('/uploadfile', methods=['GET', 'POST'])
    def upload_file():
        if request.method == 'POST':
            print(f'request: {request.files}')
            raw_data = request.files['file'].read().decode('utf-8')
            df = pd.read_csv(StringIO(raw_data))
            point_list = [
                (index, row[1][0], row[1][1]) for index, row in enumerate(df[['lat', 'long']].iterrows())]
            points = MultiPoint(point_list)
            fields = list(df.columns)
            LOGGER.debug(f'fields: {fields}')
            f = {
              'center': [points.centroid.x, points.centroid.y],
              'data': request.files['file'].read().decode('utf-8'),
              'points': point_list,
              'info': fields,
              }
            print(f)
            return f
            # path = secure_filename(f.filename)
            # print(path)
            # f.save(path)
            # return 'file uploaded successfully'
    def index():
        return "Hello World!"

    # wait for API calls
    #app = Flask(__name__, instance_relative_config=False)
    # app.wsgi_app = ReverseProxied(app.wsgi_app)
    # flask_cors.CORS(app)

    # ensure the instance folder exists
    os.makedirs(app.instance_path, exist_ok=True)
    return app


if __name__ == '__main__':
    app = create_app()
    app.run()