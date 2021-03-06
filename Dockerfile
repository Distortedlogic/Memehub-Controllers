FROM python:3.8.6-buster
RUN --mount=type=cache,target=/var/cache/apt \
	--mount=type=cache,target=/var/lib/apt apt-get update \
	&& apt-get install -qq -y build-essential libpq-dev tesseract-ocr \
	sqlite3 libsqlite3-dev python3-setuptools tree \
	--no-install-recommends && rm -rf /var/lib/apt/lists/*
WORKDIR /app
USER root
RUN --mount=type=cache,target=/root/.cache/pip pip install torch===1.7.1+cpu \
	torchvision===0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
ENV PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	TERM=xterm \
	IS_DOCKER=1
ADD start.flask.sh /
RUN chmod +x /start.flask.sh