#!/bin/bash

python ./src/preprocess.py && python ./src/train.py
cd ./src/wine-microservice
gunicorn --bind 0.0.0.0:5000 frontend:app