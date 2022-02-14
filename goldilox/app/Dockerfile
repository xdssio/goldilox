# syntax=docker/dockerfile:1.2
ARG DEBIAN_FRONTEND=noninteractive

ARG PYTHON_IMAGE="python:3.8-slim-bullseye"

FROM $PYTHON_IMAGE AS venv-image

RUN echo 'export PS1="\[\e[36m\]goldilox>\[\e[m\] "' >> /root/.bashrc

ENV PYTHONUNBUFFERED=TRUE PYTHONDONTWRITEBYTECODE=TRUE PYTHONUNBUFFERED=1  PYTHONIOENCODING=UTF-8

RUN apt-get update && apt update && apt install -y git gcc python3-pip

#RUN --mount=type=cache,mode=0755,target=/root/.cache pip install -U pip \
#    && pip3 install --no-cache-dir scikit-learn pydantic fastapi click vaex-core vaex-ml vaex-hdf5 gunicorn uvicorn cloudpickle numpy pandas pickle5
ARG GOLDILOX_VERSION='0.0.9'

RUN --mount=type=cache,mode=0755,target=/root/.cache pip install -U pip \
    && pip3 install scikit-learn pydantic fastapi click gunicorn uvicorn cloudpickle numpy pandas pickle5

RUN --mount=type=cache,mode=0755,target=/root/.cache pip3 install -U goldilox==$GOLDILOX_VERSION

RUN mkdir -p /home/app \
    && mkdir -p /home/app/models

WORKDIR /home/app

ARG PIPELINE_FILE

ARG REQUIRMENTS_FILE

ENV PIPELINE_PATH "/home/app/pipeline.pkl"

ENV REQUIRMENTS "/home/app/requirements.txt"

COPY $PIPELINE_FILE $PIPELINE_PATH

RUN glx freeze $PIPELINE_PATH $REQUIRMENTS

RUN --mount=type=cache,mode=0755,target=/root/.cache pip3 install -r $REQUIRMENTS

RUN ln -s /home/pipeline.pkl pipeline

EXPOSE 8000 5000

CMD ["glx", "serve", "/home/app/pipeline.pkl"]

# TODO
ARG PYTHON_IMAGE="continuumio/anaconda3"

FROM $PYTHON_IMAGE AS conda-image

RUN echo 'export PS1="\[\e[36m\]goldilox>\[\e[m\] "' >> /root/.bashrc

RUN pip3 install -U goldilox

ARG PYTHON_VERSION="3.8"

ARG PIPELINE_FILE

ENV PIPELINE_PATH "/home/pipeline.pkl"

ENV ENVIRONMENT "/home/environment.yml"

COPY $PIPELINE_FILE $PIPELINE_PATH

RUN glx environment $PIPELINE_PATH $ENVIRONMENT

RUN conda env create --name=conda_env python=$PYTHON_VERSION -f $ENVIRONMENT

# todo is needed?
SHELL ["conda", "run", "-n", "conda_env", "/bin/bash", "-c"]

RUN ln -s /home/pipeline.pkl pipeline

EXPOSE 8000 5000

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "conda_env", "glx", "serve", "/home/pipeline.pkl"]

