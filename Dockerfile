# syntax = docker/dockerfile:experimental
FROM python:3.7.6-buster
RUN --mount=type=cache,target=/var/cache/apt \
	--mount=type=cache,target=/var/lib/apt apt-get update \
	&& apt-get install -qq -y build-essential libpq-dev tesseract-ocr \
	sqlite3 libsqlite3-dev python3-setuptools --no-install-recommends
WORKDIR $/app
ENV PYTHONUNBUFFERED 1
USER root
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
COPY . .
RUN pip install --editable .
CMD gunicorn -c "python:config.gunicorn" "controller:APP"