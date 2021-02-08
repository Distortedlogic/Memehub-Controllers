#!/bin/bash

pip install -e .
mh load base
gunicorn -c "python:config.gunicorn" "controller:APP"