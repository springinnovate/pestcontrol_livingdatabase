import time
from flask import Flask
from flask import request
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/time')
def get_current_time():
    return {'time': time.time()}

@app.route('/uploadfile', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      print(f'request: {request.files}')
      f = {
          'data': request.files['file'].read().decode('utf-8')
          }
      print(f)
      return f
      # path = secure_filename(f.filename)
      # print(path)
      # f.save(path)
      # return 'file uploaded successfully'
