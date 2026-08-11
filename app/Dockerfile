# Container for the PCLD API and Celery worker.
FROM ubuntu:jammy

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        curl \
        emacs \
        gdal-bin \
        git \
        gnupg \
        libspatialindex-dev \
        openssl \
        postgresql-client \
        python3-gdal \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update -qq && \
    apt-get install -y --no-install-recommends google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir earthengine-api

WORKDIR /usr/local/pcld_api_config
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /usr/local/pcld_api
COPY . ./
