import contextlib
import json
import logging
import multiprocessing
import os
import pathlib
import re
import signal
import subprocess
import sys
from typing import List, Union, Callable

from pydantic import create_model

import goldilox
from goldilox.utils import to_nulls, process_variables

PIPELINE = "pipeline"
RAW = "raw"
GUNICORN = "gunicorn"
UVICORN = "uvicorn"
PATH = "path"
logger = logging.getLogger(__name__)


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


class AppPipeline:
    pipeline: goldilox.Pipeline
    meta: goldilox.Meta


app_pipeline = AppPipeline()


def get_app(path: str):
    import io
    import pandas as pd
    from fastapi import FastAPI, HTTPException
    from starlette.requests import Request
    from goldilox.config import ALLOW_CORS, CORS_ORIGINS, ALLOW_HEADERS, ALLOW_METHODS, ALLOW_CREDENTIALS

    logger = logging.getLogger(__name__)

    meta = goldilox.Meta.from_file(path)
    fastapi_params = meta.fastapi_params or {}
    app = FastAPI(**fastapi_params)

    if ALLOW_CORS:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=CORS_ORIGINS,
            allow_credentials=ALLOW_CREDENTIALS,
            allow_methods=ALLOW_METHODS,
            allow_headers=ALLOW_HEADERS,
        )
    # A dynamic way to create a pydanic model based on the raw data
    raw = process_response(meta.raw)[0]
    Query = get_query_class(raw)

    @app.on_event('startup')
    async def load_model():
        app_pipeline.pipeline = goldilox.Pipeline.from_file(path)
        app_pipeline.meta = app_pipeline.pipeline.meta
        app_pipeline.pipeline.example  # warmup and load packages

    @app.post("/inference", response_model=List[dict])
    def inference(data: Union[List[Query], Query], columns: str = ""):
        logger.info("/inference")
        data = parse_query(data)
        if not data:
            raise HTTPException(status_code=400, detail="No data provided")
        try:
            columns = None if not columns else columns.split(",")
            ret = app_pipeline.pipeline.inference(data, columns=columns)
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
        ret = app_pipeline.pipeline.inference(data, columns=columns)
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
        return process_variables(app_pipeline.meta.variables)

    @app.get("/description", response_model=str)
    def description():
        logger.info("/description")
        return app_pipeline.meta.description

    @app.get("/example", response_model=List[dict])
    def example():
        logger.info("/example")
        return process_response(app_pipeline.pipeline.example)

    @app.get("/ping", response_model=str)
    def ping():
        health = app_pipeline.pipeline is not None
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

        def load(self):
            return self.application

    return WSGIApplication


def sigterm_handler(pids):
    for pid in pids:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGQUIT)
    sys.exit(0)


def _hack_vaex_versions():
    pass


class GoldiloxServer:
    UVICORN_WORKER = 'uvicorn.workers.UvicornH11Worker'

    def __init__(self, path: str, nginx_config: str = '', options=[]):
        self.path = path
        self.nginx_config = nginx_config or os.getenv('NGINX_CONFIG')
        self.bind = None
        self.timeout = None
        self.workers = None
        self.cmd_options = self._validate_params(options)
        self.parameters = self._to_parameters(options)
        self.app = get_app(path=path)

    @staticmethod
    def _get_workers_count():
        return int(os.getenv('WORKERS', os.getenv('MODEL_SERVER_WORKERS', multiprocessing.cpu_count())))

    def _get_bind(self):
        default = '0.0.0.0:5000' if self.is_docker else '127.0.0.1:5000'
        return os.getenv('BIND', default)

    @staticmethod
    def _get_timeout():
        return int(os.getenv('TIMEOUT', os.getenv('MODEL_SERVER_TIMEOUT', 60)))

    @staticmethod
    def _extract_params(cmd: str, default: Callable, params: List[str]) -> str:
        if not params:
            raise ValueError('No params provided')
        pattern = '|'.join([f'{p}([^ ]+)' for p in params])
        result = None
        for i in re.findall(pattern, cmd):
            for value in i:
                if value:
                    result = re.sub(pattern, '', value)
                    break
        if not result:
            result = default()
            cmd = cmd + f" {params[0]}{result}"
        return cmd, result

    def _validate_params(self, options):
        cmd = ' '.join(options)
        cmd, self.bind = self._extract_params(cmd, self._get_bind, ['-b ', '--bind='])
        cmd, self.timeout = self._extract_params(cmd, self._get_timeout, ['-t ', '--timeout='])
        cmd, self.workers = self._extract_params(cmd, self._get_workers_count, ['-w ', '--workers='])

        return cmd

    def _to_parameters(self, options):
        params = {}

        def clean_key(key):
            if key.startswith('--'):
                key = key[2:]
            elif key.startswith('-'):
                key = key[1:]
            return key.replace('-', '_')

        skip = False
        for i, option in enumerate(options):
            if skip:
                skip = False
                continue
            splited = option.split('=')
            if len(splited) == 2:
                key, value = splited[0], splited[1]
                params[clean_key(key)] = value
            elif option.startswith('-') and len(options) > i + 1:
                params[clean_key(option)] = options[i + 1]
                skip = True

        params['bind'] = self.bind
        params['workers'] = self.workers
        params['timeout'] = self.timeout
        params["worker_class"] = params.get("worker_class", GoldiloxServer.UVICORN_WORKER)
        return params

    @property
    def is_docker(self):
        path = '/proc/self/cgroup'
        return (
                os.path.exists('/.dockerenv') or
                os.path.isfile(path) and any('docker' in line for line in open(path))
        )

    def serve(self, nginx=False):

        if nginx:
            pids = set([])
            pids.add(self._serve_nginx())

            pids.add(self._serve_gunicorn(nginx=nginx))
            signal.signal(signal.SIGTERM, lambda a, b: sigterm_handler(pids))

            while True:
                pid, _ = os.wait()
                if pid in pids:
                    break
            sigterm_handler(pids)
            print('Inference server exiting')
        return self._serve_wsgi()

    def _serve_nginx(self):
        print(f"Starting nginx: {self.nginx_config}")
        if self.is_docker:
            subprocess.check_call(['ln', '-sf', '/dev/stdout', '/var/log/nginx/access.log'])
            subprocess.check_call(['ln', '-sf', '/dev/stderr', '/var/log/nginx/error.log'])
        if not self.nginx_config or not pathlib.Path(self.nginx_config).exists():
            raise RuntimeError(f"NGINX config {self.nginx_config} not found")
        print(f"Starting nginx with {self.nginx_config}")
        return subprocess.Popen(['nginx', '-c', self.nginx_config]).pid

    def _serve_gunicorn(self, nginx=False):
        cmd_options = re.sub('--nginx-conf=[\S]*', "", self.cmd_options)
        if nginx:
            if '-b ' in cmd_options or '--bind=' in cmd_options:
                cmd_options = re.sub('-b \S+|--bind=[\S]*', f"-b {'unix:/tmp/gunicorn.sock'}", cmd_options)
            else:
                cmd_options += f" -b unix:/tmp/gunicorn.sock"
        cmd = f"gunicorn {cmd_options} wsgi:app"
        print(f"Starting gunicorn with {cmd}")
        return subprocess.Popen(cmd, shell=True).pid

    def _serve_wsgi(self):
        """
        To run the server with gunicorn from within python - for customization purposes
        """
        print(f"Starting wsgi application with {self.parameters}")
        WSGIApplication = get_wsgi_application(self.path)
        return WSGIApplication(self.app, self.parameters).run()


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
