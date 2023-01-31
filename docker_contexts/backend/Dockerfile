# Container for STAC API manager
FROM therealspring/python-gdal:3.0.4

RUN apt-get update -qq && \
    apt-get install -y \
    emacs \
    libspatialindex-dev \
    python3-pip \
    postgresql-client \
    openssl \
    curl \
    git \
    && \
    rm -rf /var/lib/apt/lists/*

RUN pip install earthengine-api

SHELL ["/bin/bash", "-c"]

RUN mkdir -p /usr/local/pcld_api

WORKDIR /usr/local/pcld_api
COPY requirements.txt /usr/local/pcld_api/
RUN pip3 install -r requirements.txt

COPY . /usr/local/pcld_api/
EXPOSE 8888
