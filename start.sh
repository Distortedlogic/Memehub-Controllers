#!/bin/bash
pip install -e .
mh load base
mh load stonks
gunicorn -c "python:config.gunicorn" "controller:APP"