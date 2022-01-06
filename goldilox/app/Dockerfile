# syntax=docker/dockerfile:1.2
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_IMAGE="python:3.8-slim-bullseye"

FROM $PYTHON_IMAGE AS builder-image

ARG PIPELINE_PATH

RUN echo 'export PS1="\[\e[36m\]goldilox>\[\e[m\] "' >> /root/.bashrc

ENV PYTHONUNBUFFERED=TRUE PYTHONDONTWRITEBYTECODE=TRUE PYTHONUNBUFFERED=1  PYTHONIOENCODING=UTF-8

RUN apt-get update && apt update && apt install -y git gcc

ENV LOCAL_PIPELINE_PATH "/home/pipeline.pkl"

COPY $PIPELINE_PATH $LOCAL_PIPELINE_PATH

RUN pip install -U pip && pip3 install --no-cache-dir wheel && pip3 install --no-cache-dir goldilox==0.0.1a9  pydantic fastapi click vaex

RUN glx install $LOCAL_PIPELINE_PATH

EXPOSE 8000

CMD ["glx", "serve", "$LOCAL_PIPELINE_PATH"]
