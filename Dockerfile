# syntax=docker/dockerfile:1.2
ARG DEBIAN_FRONTEND=noninteractive

ARG PYTHON_IMAGE="python:3.8-slim-bullseye"

FROM $PYTHON_IMAGE AS builder-image

RUN echo 'export PS1="\[\e[36m\]goldilox>\[\e[m\] "' >> /root/.bashrc

ENV PYTHONUNBUFFERED=TRUE PYTHONDONTWRITEBYTECODE=TRUE PYTHONUNBUFFERED=1  PYTHONIOENCODING=UTF-8

RUN apt-get update && apt update && apt install -y git gcc python3-pip

RUN pip install -U pip && pip3 install --no-cache-dir  pydantic fastapi click vaex gunicorn uvicorn cloudpickle numpy pandas

RUN pip3 install goldilox==0.0.2

ARG PIPELINE_FILE

ENV PIPELINE_PATH "/home/pipeline.pkl"

ENV REQUIRMENTS "/home/requirements.txt"

COPY $PIPELINE_FILE $PIPELINE_PATH

RUN glx freeze $PIPELINE_PATH REQUIRMENTS

RUN pip3 install -r REQUIRMENTS

RUN ln -s /home/pipeline.pkl pipeline

EXPOSE 8000

CMD ["glx", "serve", "/home/pipeline.pkl"]
