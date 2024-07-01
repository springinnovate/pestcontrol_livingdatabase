# Container for STAC API manager
FROM ubuntu:jammy

RUN apt-get update -qq && \
    apt-get install -y \
    emacs \
    libspatialindex-dev \
    python3-pip \
    postgresql-client \
    openssl \
    curl \
    git \
    gdal-bin \
    python3-gdal \
    && \
    rm -rf /var/lib/apt/lists/*

RUN pip install earthengine-api

SHELL ["/bin/bash", "-c"]

RUN mkdir -p /usr/local/pcld_api_config

WORKDIR /usr/local/pcld_api_config
COPY requirements.txt /usr/local/pcld_api_config/
RUN pip3 install -r requirements.txt

WORKDIR /usr/local/pcld_api
