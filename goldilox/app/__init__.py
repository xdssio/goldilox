import json
import logging
from typing import List, Union

from pydantic import create_model

from goldilox import Pipeline
from goldilox.utils import to_nulls, process_variables

PIPELINE = "pipeline"
RAW = "raw"
GUNICORN = "gunicorn"
UVICORN = "uvicorn"
PATH = "path"


def parse_query(query):
    if not isinstance(query, list):
        query = [query]
    return [q.dict() for q in query if q]


def process_response(items):
    items = Pipeline.to_records(items)
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

    pipeline = Pipeline.from_file(path)
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


class Server:
    def __init__(self, path, root_path='', options={}):
        self.path = path
        self.options = options
        self.app = get_app(path=path, root_path=root_path)
        pipeline = Pipeline.from_file(path)
        if not pipeline.validate():
            raise RuntimeError(f"Pipeline in {path} is invalid")

    @staticmethod
    def _validate_worker_class(options):
        options["worker_class"] = options.get(
            "worker_class", "uvicorn.workers.UvicornH11Worker"
        )
        return options

    def serve(self, options=None):
        options = self._validate_worker_class(options or self.options)
        WSGIApplication = get_wsgi_application(self.path)
        WSGIApplication(self.app, options).run()
