# syntax=docker/dockerfile:1.2
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_IMAGE="3.7.12-slim-buster"
ARG LOCAL_PIPELINE_PATH="home/pipeline.pkl"
ARG PIPELINE_PATH


FROM $PYTHON_IMAGE AS builder-image

RUN echo 'export PS1="\[\e[36m\]goldilox>\[\e[m\] "' >> /root/.bashrc

COPY $PIPELINE_PATH $LOCAL_PIPELINE_PATH

RUN pip3 install goldilox==0.0.1a5

RUN glx install $LOCAL_PIPELINE_PATH


ENV PYTHONUNBUFFERED=TRUE PYTHONDONTWRITEBYTECODE=TRUE PYTHONUNBUFFERED=1  PYTHONIOENCODING=UTF-8 TZ=Asia/Jerusalemm

EXPOSE 8000

CMD ["glx", "serve", "$LOCAL_PIPELINE_PATH"]
