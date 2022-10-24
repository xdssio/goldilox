# syntax=docker/dockerfile:1.2
ARG DEBIAN_FRONTEND=noninteractive

ARG PYTHON_IMAGE="python:3.8-slim-bullseye"

FROM $PYTHON_IMAGE AS venv-image

RUN echo 'export PS1="\[\e[36m\]goldilox>\[\e[m\] "' >> /root/.bashrc

ENV PYTHONUNBUFFERED=TRUE PYTHONDONTWRITEBYTECODE=TRUE PYTHONUNBUFFERED=1  PYTHONIOENCODING=UTF-8

RUN apt-get update && apt update && apt install -y git gcc python3-pip nano

RUN apt-get install -y nginx && mkdir -p /var/log/nginx

#RUN apt-get install -y curl \
#    && curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf > rustup.sh \
#    && sh rustup.sh -y \
#    && bash -c "echo source $HOME/.cargo/env >> /etc/bash.bashrc"

ENV WORK_DIR "/opt/program"

RUN mkdir -p $WORK_DIR

WORKDIR $WORK_DIR

ARG GOLDILOX_VERSION='0.0.16'

# TODO REMOVE ################
ADD . $WORK_DIR/goldilox

RUN --mount=type=cache,target=/root/.cache pip3 install -U pip \
    &&  pip3 install -e $WORK_DIR/goldilox # TODO pip3 install goldilox==$GOLDILOX_VERSION

# TODO REMOVE UNTIL ################

ARG PIPELINE_FILE

COPY $PIPELINE_FILE $WORK_DIR/pipeline.pkl

RUN glx export $WORK_DIR/pipeline.pkl $WORK_DIR --framework='gunicorn' --nginx

RUN --mount=type=cache,target=/root/.cache pip3 install -r $WORK_DIR/requirements.txt

# Hack veax for faster startup
RUN mkdir -p $(python3 -c 'import site; print(site.getsitepackages()[0])') \
    && echo """def get_versions(): return {'vaex-core': '4.5.1', 'vaex-hdf5': '0.10.0', 'vaex-ml': '0.14.0'}"""  >> $(python3 -c 'import site; print(site.getsitepackages()[0])')/vaex/version.py

RUN ln -s $WORK_DIR/pipeline.pkl pipeline

ARG USE_NGINX

ENV NGINX_CONFIG=/opt/program/nginx.conf USE_NGINX=$USE_NGINX

EXPOSE 5000 8080

CMD glx serve pipeline $USE_NGINX

ARG PYTHON_IMAGE="conda-forge/mambaforge-pypy3"

FROM $PYTHON_IMAGE AS conda-image

RUN echo 'export PS1="\[\e[36m\]goldilox>\[\e[m\] "' >> /root/.bashrc

RUN apt-get update && apt update && apt install -y git gcc python3-pip

ARG GOLDILOX_VERSION='0.0.15'

RUN --mount=type=cache,mode=0755,target=/root/.cache pip3 install goldilox==$GOLDILOX_VERSION

RUN mkdir -p /opt/program

WORKDIR /opt/program

ARG PIPELINE_FILE

ARG PYTHON_VERSION="3.8"

ENV PIPELINE_PATH "/opt/program/pipeline.pkl"

ENV REQUIRMENTS "/opt/program/environment.yaml"

COPY $PIPELINE_FILE $PIPELINE_PATH

RUN glx freeze $PIPELINE_PATH --output $REQUIRMENTS

RUN --mount=type=cache,mode=0755,target=/opt/conda/pkgs conda env create --name=conda_env python=$PYTHON_VERSION -f $REQUIRMENTS

SHELL ["/bin/bash","-c"]

RUN conda init

RUN echo 'conda activate conda_env' >> ~/.bashrc

SHELL ["conda", "run", "-n", "conda_env", "/bin/bash", "-c"]

ARG GOLDILOX_VERSION='0.0.15'

RUN --mount=type=cache,mode=0755,target=/root/.cache pip3 install goldilox==$GOLDILOX_VERSION

RUN ln -s /opt/program/pipeline.pkl pipeline

ARG NGINX

ENV NGINX=$NGINX

EXPOSE 5000 8080

CMD conda run --no-capture-output -n conda_env glx serve pipeline $NGINX

