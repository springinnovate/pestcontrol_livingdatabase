#!/bin/bash

export FLASK_APP=database_app.py
export FLASK_ENV=development
flask run --port=80
