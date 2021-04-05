#!/bin/bash
pip install -e .
mh load stonk-market
gunicorn -c "python:config.gunicorn" "src.flask_app:APP"