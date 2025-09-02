# docker build -t pcld_trait_search_env:latest . && docker run --rm -it -v "%CD%":/app -w /app pcld_trait_search_env:latest

FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

CMD ["bash"]
