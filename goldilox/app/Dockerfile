# syntax=docker/dockerfile:1.2
ARG DEBIAN_FRONTEND=noninteractive

ARG PYTHON_IMAGE="python:3.8-slim-bullseye"

FROM $PYTHON_IMAGE AS venv-image

RUN echo 'export PS1="\[\e[36m\]goldilox>\[\e[m\] "' >> /root/.bashrc

RUN apt-get update && apt update && apt install -y git gcc python3-pip nano

RUN apt-get install -y nginx && mkdir -p /var/log/nginx

# Get Rust for blake3
#RUN apt-get install curl && curl https://sh.rustup.rs -sSf | sh -s -- -y
#ENV PATH="/root/.cargo/bin:${PATH}"

ENV WORK_DIR "/opt/program"

RUN mkdir -p $WORK_DIR

WORKDIR $WORK_DIR

ARG GOLDILOX_VERSION='0.0.19'

RUN --mount=type=cache,target=/root/.cache pip3 install -U pip \
    &&  pip3 install goldilox==$GOLDILOX_VERSION

ARG PIPELINE_FILE

COPY $PIPELINE_FILE $WORK_DIR/pipeline.pkl

RUN glx export $WORK_DIR/pipeline.pkl $WORK_DIR --framework='gunicorn' --nginx

RUN --mount=type=cache,target=/root/.cache pip3 install -r $WORK_DIR/requirements.txt

RUN ln -s $WORK_DIR/pipeline.pkl pipeline

RUN glx hackvaex

ARG USE_NGINX

ENV NGINX_CONFIG=/opt/program/nginx.conf USE_NGINX=$USE_NGINX PATH=/program:${PATH} PYTHONUNBUFFERED=TRUE PYTHONDONTWRITEBYTECODE=TRUE PYTHONUNBUFFERED=1  PYTHONIOENCODING=UTF-8

EXPOSE 5000 8080

CMD glx serve pipeline $USE_NGINX

ARG PYTHON_IMAGE="conda-forge/mambaforge-pypy3"

FROM $PYTHON_IMAGE AS conda-image

USER root

RUN echo 'export PS1="\[\e[36m\]goldilox>\[\e[m\] "' >> /root/.bashrc

RUN apt-get update && apt update && apt install -y git gcc python3-pip

ENV WORK_DIR "/opt/program"

RUN mkdir -p $WORK_DIR

WORKDIR $WORK_DIR

ARG GOLDILOX_VERSION='0.0.19'

RUN --mount=type=cache,target=/root/.cache pip3 install -U pip \
    &&  pip3 install goldilox==$GOLDILOX_VERSION

ARG PIPELINE_FILE

ARG PYTHON_VERSION="3.8"

ENV REQUIRMENTS "/opt/program/environment.yml"

COPY $PIPELINE_FILE $WORK_DIR/pipeline.pkl

RUN ln -s $WORK_DIR/pipeline.pkl pipeline

RUN glx export pipeline $WORK_DIR --framework='gunicorn' --nginx

RUN  --mount=type=cache,mode=0755,target=/opt/conda/pkgs micromamba install -y -n base -f $REQUIRMENTS \
    && micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN --mount=type=cache,target=/root/.cache pip3 install goldilox==$GOLDILOX_VERSION

ARG USE_NGINX

ENV NGINX_CONFIG=/opt/program/nginx.conf USE_NGINX=$USE_NGINX PATH=/program:${PATH} PYTHONUNBUFFERED=TRUE PYTHONDONTWRITEBYTECODE=TRUE PYTHONUNBUFFERED=1  PYTHONIOENCODING=UTF-8

EXPOSE 5000 8080

CMD glx serve pipeline $USE_NGINX

