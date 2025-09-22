# docker build -t pcld_trait_search_env:latest . && docker run --rm -it -v "%CD%":/app -w /app pcld_trait_search_env:latest

FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install common system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    psmisc \
    htop \
    less \
    vim-tiny \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
RUN python -m playwright install --with-deps chromium

CMD ["bash"]
