#!/bin/bash
if [ -z "${AWS_LAMBDA_RUNTIME_API}" ]; then
    exec /opt/program/aws-lambda-rie /usr/local/bin/python -m awslambdaric $1
else
    exec /usr/local/bin/python -m awslambdaric $1
fi