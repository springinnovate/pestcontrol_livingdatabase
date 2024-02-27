#!/bin/bash

export FLASK_APP=database_app.py
export FLASK_ENV=development
authbind --deep flask run --port=80
