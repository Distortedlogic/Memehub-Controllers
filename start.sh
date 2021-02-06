#!/bin/bash

pip install -e .
gunicorn -c "python:config.gunicorn" "controller:APP"