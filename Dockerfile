# syntax = docker/dockerfile:experimental
FROM python:3.7.6-buster
RUN --mount=type=cache,target=/var/cache/apt \
	--mount=type=cache,target=/var/lib/apt apt-get update \
	&& apt-get install -qq -y build-essential libpq-dev tesseract-ocr \
	sqlite3 libsqlite3-dev python3-setuptools --no-install-recommends \
	&& rm -rf /var/lib/apt/lists/*
WORKDIR $/app
ENV PYTHONUNBUFFERED=1 \
	TERM=xterm \
	TF_CPP_MIN_LOG_LEVEL=2 \
	FLASK_RUN_PORT=8000 \
	FLASK_APP=controller:APP \
	FLASK_ENV=development
USER root
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
COPY . .
RUN pip install --editable .
CMD gunicorn -c "python:config.gunicorn" "controller:APP" --reload --log-level=error