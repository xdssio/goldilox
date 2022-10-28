# syntax=docker/dockerfile:1.2
ARG DEBIAN_FRONTEND=noninteractive

ARG PYTHON_IMAGE="python:3.8-slim-bullseye"

FROM $PYTHON_IMAGE AS venv-image

RUN echo 'export PS1="\[\e[36m\]goldilox>\[\e[m\] "' >> /root/.bashrc

RUN apt-get update && apt update && apt install -y git gcc python3-pip nano

ENV WORK_DIR "/opt/program"

RUN mkdir -p $WORK_DIR

WORKDIR $WORK_DIR

ARG GOLDILOX_VERSION='0.0.22'

RUN --mount=type=cache,target=/root/.cache pip3 install -U pip setuptools \
    && pip3 install goldilox==$GOLDILOX_VERSION \
    && pip3 install awslambdaric mangum

ARG PIPELINE_FILE

COPY $PIPELINE_FILE $WORK_DIR/pipeline.pkl

RUN glx export $WORK_DIR/pipeline.pkl $WORK_DIR --framework='lambda'

RUN chmod 755 $WORK_DIR/entrypoint.sh \
    && chmod 755  $WORK_DIR/aws-lambda-rie

RUN --mount=type=cache,target=/root/.cache pip3 install -r $WORK_DIR/requirements.txt

RUN glx hackvaex

ENV PYTHONPATH="${PYTHONPATH}:${WORK_DIR}" PATH=/opt/program:${PATH} HOME='/tmp'

ENV PYTHONUNBUFFERED=TRUE PYTHONDONTWRITEBYTECODE=TRUE PYTHONUNBUFFERED=1  PYTHONIOENCODING=UTF-8 OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

EXPOSE 5000 8080

ENTRYPOINT [ "/opt/program/entrypoint.sh" ]

CMD ["handler.handler" ]

