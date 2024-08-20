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

RUN apt-get install apt-transport-https ca-certificates gnupg curl

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-cli -y


SHELL ["/bin/bash", "-c"]

RUN mkdir -p /usr/local/pcld_api_config

WORKDIR /usr/local/pcld_api_config
COPY requirements.txt /usr/local/pcld_api_config/
RUN pip3 install -r requirements.txt

ENV GOOGLE_APPLICATION_CREDENTIALS="/usr/local/pcld_api/secrets/service-account-key.json"

WORKDIR /usr/local/pcld_api
