# syntax=docker/dockerfile:1.2
ARG DEBIAN_FRONTEND=noninteractive

ARG PYTHON_VERSION="3.8"

ARG PYTHON_IMAGE="python:$PYTHON_VERSION-slim-bullseye"

FROM $PYTHON_IMAGE AS venv-image

RUN echo 'export PS1="\[\e[36m\]goldilox>\[\e[m\] "' >> /root/.bashrc

ENV PYTHONUNBUFFERED=TRUE PYTHONDONTWRITEBYTECODE=TRUE PYTHONUNBUFFERED=1  PYTHONIOENCODING=UTF-8

RUN apt-get update && apt update && apt install -y git gcc python3-pip

RUN pip install -U pip && pip3 install --no-cache-dir  pydantic fastapi click vaex gunicorn uvicorn cloudpickle numpy pandas

RUN pip3 install -U goldilox

ARG PIPELINE_FILE

ENV PIPELINE_PATH "/home/pipeline.pkl"

ENV REQUIRMENTS "/home/requirements.txt"

COPY $PIPELINE_FILE $PIPELINE_PATH

RUN glx freeze $PIPELINE_PATH $REQUIRMENTS

RUN pip3 install -r $REQUIRMENTS

RUN ln -s /home/pipeline.pkl pipeline

EXPOSE 8000 5000

CMD ["glx", "serve", "/home/pipeline.pkl"]

# TODO
FROM continuumio/anaconda3 AS conda-image

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

