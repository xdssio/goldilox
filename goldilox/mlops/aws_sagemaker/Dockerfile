# syntax=docker/dockerfile:1.2
ARG DEBIAN_FRONTEND=noninteractive

ARG PYTHON_IMAGE="python:3.8-slim-bullseye"

FROM $PYTHON_IMAGE AS venv-image

RUN echo 'export PS1="\[\e[36m\]goldilox>\[\e[m\] "' >> /root/.bashrc

RUN apt-get update && apt update && apt install -y git gcc python3-pip nano

ENV WORK_DIR "/opt/program"

RUN mkdir -p $WORK_DIR \
    && mkdir -p /opt/ml/code

WORKDIR $WORK_DIR

ARG GOLDILOX_VERSION='0.0.24'

RUN --mount=type=cache,target=/root/.cache pip3 install -U pip setuptools \
    && pip3 install goldilox==$GOLDILOX_VERSION \
    && pip3 install awslambdaric mangum 'cloudpathlib[s3]' sagemaker-training multi-model-server sagemaker-inference smdebug

ARG PIPELINE_FILE

ENV PIPELINE_PATH $WORK_DIR/pipeline.pkl

COPY $PIPELINE_FILE $PIPELINE_PATH


##### TODO remove #####
COPY . $WORK_DIR/goldilox

RUN pip3 install --no-deps -e $WORK_DIR/goldilox
### TODO remove #####

RUN glx export $PIPELINE_PATH $WORK_DIR --framework='sagemaker'

RUN --mount=type=cache,target=/root/.cache pip3 install -r $WORK_DIR/requirements.txt

RUN glx hackvaex

RUN ln -s $WORK_DIR/train.py /opt/ml/code/train.py && chmod 755 /opt/ml/code/train.py

ENV PYTHONPATH="${PYTHONPATH}:${WORK_DIR}" PATH=/opt/ml/code:${PATH} PIPELINE_PATH=$WORK_DIR/pipeline.pkl SAGEMAKER_PROGRAM=/opt/ml/code/train.py

ENV PYTHONUNBUFFERED=TRUE PYTHONDONTWRITEBYTECODE=TRUE PYTHONUNBUFFERED=1  PYTHONIOENCODING=UTF-8 OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

EXPOSE 5000 8080

