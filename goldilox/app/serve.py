#!/usr/bin/env python

# This file implements the scoring service shell. You don't necessarily need to modify it for various
# algorithms. It starts nginx and gunicorn with the correct configurations and then simply waits until
# gunicorn exits.
#
# The flask server is specified to be the app object in wsgi.py
#
# We set the following parameters:
#
# Parameter                Environment Variable              Default Value
# ---------                --------------------              -------------
# number of workers        MODEL_SERVER_WORKERS              the number of CPU cores
# timeout                  MODEL_SERVER_TIMEOUT              60 seconds

import contextlib
import multiprocessing
import os
import signal
import subprocess
import sys

cpu_count = multiprocessing.cpu_count()

model_server_timeout = os.environ.get('MODEL_SERVER_TIMEOUT', 60)
model_server_workers = int(os.environ.get('MODEL_SERVER_WORKERS', cpu_count))

import os


def is_docker():
    path = '/proc/self/cgroup'
    return (
            os.path.exists('/.dockerenv') or
            os.path.isfile(path) and any('docker' in line for line in open(path))
    )


def sigterm_handler(nginx_pid, gunicorn_pid):
    with contextlib.suppress(OSError):
        os.kill(nginx_pid, signal.SIGQUIT)
    with contextlib.suppress(OSError):
        os.kill(gunicorn_pid, signal.SIGTERM)
    sys.exit(0)


def start_server(nginx=False, options: dict = None):
    print('Starting the inference server with {} workers.'.format(model_server_workers))
    nginx_config_path = 'nginx.conf'
    # link the log streams to stdout/err so they will be logged to the container logs
    if is_docker():
        subprocess.check_call(['ln', '-sf', '/dev/stdout', '/var/log/nginx/access.log'])
        subprocess.check_call(['ln', '-sf', '/dev/stderr', '/var/log/nginx/error.log'])
        nginx_config_path = '/opt/program/nginx.conf'

    nginx = subprocess.Popen(['nginx', '-c', nginx_config_path])
    gunicorn = subprocess.Popen(['gunicorn',
                                 '--timeout', str(model_server_timeout),
                                 '-k', 'sync',
                                 '-b', 'unix:/tmp/gunicorn.sock',
                                 '-w', str(model_server_workers),
                                 'wsgi:app'])
    pids = set([gunicorn.pid, nginx.pid])
    signal.signal(signal.SIGTERM, lambda a, b: sigterm_handler(nginx.pid, gunicorn.pid))

    # If either subprocess exits, so do we.
    while True:
        pid, _ = os.wait()
        if pid in pids:
            break

    sigterm_handler(nginx.pid, gunicorn.pid)
    print('Inference server exiting')


if __name__ == '__main__':
    start_server()
