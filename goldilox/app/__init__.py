import contextlib
import json
import logging
import multiprocessing
import os
import re
import signal
import subprocess
import sys
from typing import List, Union

from pydantic import create_model

import goldilox
from goldilox.utils import to_nulls, process_variables

PIPELINE = "pipeline"
RAW = "raw"
GUNICORN = "gunicorn"
UVICORN = "uvicorn"
PATH = "path"

logger = logging.getLogger(__name__)


def is_docker():
    path = '/proc/self/cgroup'
    return (
            os.path.exists('/.dockerenv') or
            os.path.isfile(path) and any('docker' in line for line in open(path))
    )


def parse_query(query):
    if not isinstance(query, list):
        query = [query]
    return [q.dict() for q in query if q]


def process_response(items):
    items = goldilox.Pipeline.to_records(items)
    if not isinstance(items, list):
        items = [items]
    for item in items:
        for key, value in item.items():
            item[key] = to_nulls(value)
    return items


def get_query_class(raw):
    return create_model(
        "Query",
        **{k: (type(v), None) for k, v in raw.items()},
        __config__=type("QueryConfig", (object,), {"schema_extra": {"example": raw}}),
    )


def get_app(path: str, root_path: str = ''):
    import io
    import pandas as pd
    from fastapi import FastAPI, HTTPException
    from starlette.requests import Request
    from goldilox.config import ALLOW_CORS, CORS_ORIGINS, ALLOW_HEADERS, ALLOW_METHODS, ALLOW_CREDENTIALS
    logger = logging.getLogger(__name__)
    app = FastAPI(root_path=root_path)
    if ALLOW_CORS:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=CORS_ORIGINS,
            allow_credentials=ALLOW_CREDENTIALS,
            allow_methods=ALLOW_METHODS,
            allow_headers=ALLOW_HEADERS,
        )
    PIPELINE = "pipeline"

    pipeline = goldilox.Pipeline.from_file(path)
    # A dynamic way to create a pydanic model based on the raw data
    raw = process_response(pipeline.raw)[0]
    Query = get_query_class(raw)

    def get_pipeline():
        return app.state._state.get(PIPELINE, pipeline)

    @app.post("/inference", response_model=List[dict])
    def inference(data: List[Query], columns: str = ""):
        logger.info("/inference")
        data = parse_query(data)
        if not data:
            raise HTTPException(status_code=400, detail="No data provided")
        try:
            columns = None if not columns else columns.split(",")
            ret = get_pipeline().inference(data, columns=columns)

        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=400, detail=str(
                f"Issue with inference, try runing your pipeline locally with 'pipeline.inference(data)' to see what is the problem:\n{e}"))

        return process_response(ret)

    @app.post("/invocations", response_model=Union[List[dict], str])
    async def invocations(request: Request, columns: str = ""):
        logger.info("/invocations")
        content_type = request.headers.get("content-type", None)
        charset = request.headers.get("charset", "utf-8")
        data = await request.body()
        if not data:
            raise HTTPException(status_code=400, detail="No data provided")

        data = data.decode(charset)
        if content_type == "text/csv":
            data = pd.read_csv(io.StringIO(data), encoding=charset)
        else:
            data = json.loads(data)
        if isinstance(data, dict) and "data" in data:  # mlflow format
            data = data["data"]
        columns = None if not columns else columns.split(",")
        ret = get_pipeline().inference(data, columns=columns)
        if content_type == "text/csv":
            if hasattr(ret, 'to_pandas_df'):
                ret = ret.to_pandas_df()
            out = io.StringIO()
            ret.to_csv(out, index=False)
            return out.getvalue()

        return process_response(ret)

    @app.get("/variables", response_model=dict)
    def variables():
        logger.info("/variables")
        return process_variables(get_pipeline().variables)

    @app.get("/description", response_model=str)
    def description():
        logger.info("/description")
        return get_pipeline().description

    @app.get("/example", response_model=List[dict])
    def example():
        logger.info("/example")
        return process_response(get_pipeline().example)

    @app.get("/ping", response_model=str)
    def ping():
        health = get_pipeline() is not None
        if not health:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        return 'pong'

    return app


def get_wsgi_application(path: str):
    from gunicorn.app.base import BaseApplication

    class WSGIApplication(BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()

        # # per worker
        def load_config(self):
            config = {
                key: value
                for key, value in self.options.items()
                if key in self.cfg.settings and value is not None
            }
            for key, value in config.items():
                self.cfg.set(key.lower(), value)
            pipeline = Pipeline.from_file(path)
            self.application.state._state[PIPELINE] = pipeline
            self.application.state._state[RAW] = pipeline.raw.copy()

        def load(self):
            return self.application

    return WSGIApplication


def sigterm_handler(nginx_pid, gunicorn_pid):
    with contextlib.suppress(OSError):
        os.kill(nginx_pid, signal.SIGQUIT)
    with contextlib.suppress(OSError):
        os.kill(gunicorn_pid, signal.SIGTERM)
    sys.exit(0)


class GoldiloxServer:
    UVICORN_WORKER = 'uvicorn.workers.UvicornH11Worker'

    def __init__(self, path, root_path='', nginx_config=None, options={}):
        self.path = path
        self.cmd_options = ' '.join(options)
        self.parameters = self._to_parameters(options)
        self.workers = self.parameters.get('workers', 1)
        self.host = self.parameters.get('host', self.parameters.get('bind').split(':')[0])
        self.port = self.parameters.get('port', self.parameters.get('bind').split(':')[1])
        self.app = get_app(path=path, root_path=root_path)
        self.nginx_config = nginx_config
        pipeline = goldilox.Pipeline.from_file(path)
        if not pipeline.validate():
            raise RuntimeError(f"Pipeline in {path} is invalid")

    def _to_parameters(self, options):
        params = {}

        def clean_key(key):
            if key.startswith('--'):
                key = key[2:]
            elif key.startswith('-'):
                key = key[1:]
            return key.replace('-', '_')

        for option in options['options']:
            splited = option.split('=')
            if len(splited) == 2:
                key, value = splited[0], splited[1]
                params[clean_key(key)] = value
            else:
                print(f"(skip) - option {option} was not understood - use key=value version please ")
        bind = params.get('bind', os.getenv('BIND', ''))
        if not bind:
            if self.is_docker():
                bind = "unix:/tmp/gunicorn.sock"
                self.cmd_options = re.sub('-b \S+|--bind=[\S]*', f"-b {bind}", self.cmd_options)
            else:
                host = params.get('host', os.getenv('HOST', '127.0.0.1'))
                port = int(params.get('port', os.getenv('PORT', 5000)))
                bind = f"{host}:{port}"
        params['bind'] = bind
        params['workers'] = params.get('workers', int(os.environ.get('WORKERS', multiprocessing.cpu_count())))
        params["worker_class"] = params.get("worker_class", GoldiloxServer.UVICORN_WORKER)
        return options

    def serve(self, nginx=False):
        if nginx:
            nginx_pid = self._serve_nginx()
            gunicorn_pid = self._serve_gunicorn()
            pids = set([gunicorn_pid, nginx_pid])
            signal.signal(signal.SIGTERM, lambda a, b: sigterm_handler(nginx_pid, gunicorn_pid))

            # If either subprocess exits, so do we.
            while True:
                pid, _ = os.wait()
                if pid in pids:
                    break

            sigterm_handler(nginx_pid, gunicorn_pid)
            print('Inference server exiting')
        return self._serve_wsgi()

    def _serve_nginx(self):
        # nginx_config_path = '/opt/program/nginx.conf'
        if self.is_docker():
            subprocess.check_call(['ln', '-sf', '/dev/stdout', '/var/log/nginx/access.log'])
            subprocess.check_call(['ln', '-sf', '/dev/stderr', '/var/log/nginx/error.log'])

        return subprocess.Popen(['nginx', '-c', self.nginx_config]).pid

    def _serve_gunicorn(self):
        return subprocess.Popen(['gunicorn', self.cmd_options, 'wsgi:app']).pid

    def _serve_wsgi(self):
        WSGIApplication = get_wsgi_application(self.path)
        options = self._to_parameters(self.options)
        return WSGIApplication(self.app, options).run()


class SimpleServer:
    def __init__(self, path, options={}):
        self.path = path
        self.options = options

    def serve(self):
        command = self._get_command()
        logger.info(f"Serving {self.name} as follow: {' '.join(command)}")
        subprocess.check_call(command)


class GunicornServer(SimpleServer):
    name = 'gunicorn'

    def _get_command(self):
        return ['gunicorn', 'wsgi:app'] + list(self.options['options'])


class MLFlowServer(SimpleServer):
    name = 'mlflow'

    def _get_command(self):
        meta = goldilox.Meta.from_file(os.path.join(self.path, 'artifacts', 'pipeline.pkl'))
        environment_param = ['--no-conda'] if meta.env_type == goldilox.config.CONSTANTS.VENV else []
        return ['mlflow', 'models', 'serve', f"-m", os.path.abspath(self.path)] + environment_param + list(
            self.options['options'])


class RayServer(SimpleServer):
    name = 'ray'

    def _get_command(self):
        return ['python', 'main.py'] + list(self.options['options'])
