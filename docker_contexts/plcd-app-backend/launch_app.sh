#!/bin/bash

export FLASK_APP=database_app.py
export FLASK_ENV=development
authbind --deep flask run --host=0.0.0.0 --port=80
