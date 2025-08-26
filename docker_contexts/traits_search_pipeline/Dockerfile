# docker build -t py-sqlalchemy-openai:latest . && docker run --rm -it -v "$PWD":/app -w /app py-sqlalchemy-openai:latest

# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

CMD ["bash"]
