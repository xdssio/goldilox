# syntax=docker/dockerfile:1.2
ARG DEBIAN_FRONTEND=noninteractive

ARG PYTHON_IMAGE="python:3.8-slim-bullseye"

FROM $PYTHON_IMAGE AS venv-image

RUN echo 'export PS1="\[\e[36m\]goldilox>\[\e[m\] "' >> /root/.bashrc

ENV PYTHONUNBUFFERED=TRUE PYTHONDONTWRITEBYTECODE=TRUE PYTHONUNBUFFERED=1  PYTHONIOENCODING=UTF-8

RUN apt-get update && apt update && apt install -y git gcc python3-pip

ARG GOLDILOX_VERSION='0.0.15'

RUN --mount=type=cache,mode=0755,target=/root/.cache pip3 install goldilox==$GOLDILOX_VERSION

RUN mkdir -p /home/app

WORKDIR /home/app

ARG PIPELINE_FILE

ENV PIPELINE_PATH "/home/app/pipeline.pkl"

ENV REQUIRMENTS "/home/app/requirements.txt"

COPY $PIPELINE_FILE $PIPELINE_PATH

RUN glx freeze $PIPELINE_PATH --output $REQUIRMENTS

RUN --mount=type=cache,mode=0755,target=/root/.cache pip3 install -r $REQUIRMENTS

RUN ln -s /home/app/pipeline.pkl pipeline

EXPOSE 8000 5000

CMD ["glx", "serve", "pipeline"]

ARG PYTHON_IMAGE="conda-forge/mambaforge-pypy3"

FROM $PYTHON_IMAGE AS conda-image

RUN echo 'export PS1="\[\e[36m\]goldilox>\[\e[m\] "' >> /root/.bashrc

RUN apt-get update && apt update && apt install -y git gcc python3-pip

ARG GOLDILOX_VERSION='0.0.15'

RUN --mount=type=cache,mode=0755,target=/root/.cache pip3 install goldilox==$GOLDILOX_VERSION

RUN mkdir -p /home/app

WORKDIR /home/app

ARG PIPELINE_FILE

ARG PYTHON_VERSION="3.8"

ENV PIPELINE_PATH "/home/app/pipeline.pkl"

ENV REQUIRMENTS "/home/app/environment.yaml"

COPY $PIPELINE_FILE $PIPELINE_PATH

RUN glx freeze $PIPELINE_PATH --output $REQUIRMENTS

RUN --mount=type=cache,mode=0755,target=/opt/conda/pkgs conda env create --name=conda_env python=$PYTHON_VERSION -f $REQUIRMENTS

SHELL ["/bin/bash","-c"]

RUN conda init

RUN echo 'conda activate conda_env' >> ~/.bashrc

SHELL ["conda", "run", "-n", "conda_env", "/bin/bash", "-c"]

ARG GOLDILOX_VERSION='0.0.15'

RUN --mount=type=cache,mode=0755,target=/root/.cache pip3 install goldilox==$GOLDILOX_VERSION

RUN ln -s /home/app/pipeline.pkl pipeline

EXPOSE 8000 5000

CMD ["conda", "run", "--no-capture-output", "-n", "conda_env", "glx", "serve", "pipeline"]
