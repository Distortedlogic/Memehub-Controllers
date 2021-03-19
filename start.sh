#!/bin/bash
pip install -e .
mh load base
mh load stonks
gunicorn -c "python:config.gunicorn" "src:APP"