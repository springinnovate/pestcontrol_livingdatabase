# docker build -t py-sqlalchemy-openai:latest . && docker run --rm -it -v "$PWD":/app -w /app py-sqlalchemy-openai:latest

FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

CMD ["bash"]
