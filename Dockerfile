FROM python:3.14-slim

RUN pip install --no-cache-dir \
    numpy==2.5.2 \
    pandas==3.0.5 \
    pyarrow==25.0.1

WORKDIR /data
