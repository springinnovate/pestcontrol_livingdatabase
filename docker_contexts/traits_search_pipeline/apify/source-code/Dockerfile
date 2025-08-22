FROM apify/actor-python:3.13
COPY requirements.txt ./

RUN echo "Python version:" \
 && python --version \
 && echo "Pip version:" \
 && pip --version \
 && echo "Installing dependencies:" \
 && pip install -r requirements.txt \
 && echo "All installed Python packages:" \
 && pip freeze

RUN playwright install --with-deps chromium

COPY . ./

RUN python3 -m compileall -q src/

CMD ["python3", "-m", "src"]
