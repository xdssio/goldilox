import logging
from typing import List

import numpy as np
import pandas as pd
from pydantic import create_model

from goldilox import Pipeline

PIPELINE = "pipeline"
GUNICORN = "gunicorn"
UVICORN = "uvicorn"
PATH = "path"

valida_types = {type(None), dict, list, int, float, str, bool}


def parse_query(query):
    if not isinstance(query, list):
        query = [query]
    return [q.dict() for q in query if q]


def is_list(l):
    return isinstance(l, (list, pd.core.series.Series, np.ndarray))


def to_nulls(value):
    if is_list(value):
        return [to_nulls(v) for v in value]
    elif isinstance(value, dict):
        return {to_nulls(k): to_nulls(v) for k, v in value.items()}
    elif pd.isnull(value):
        return None
    return value


def process_response(items):
    items = Pipeline.to_records(items)
    if not isinstance(items, list):
        items = [items]
    for item in items:
        for key, value in item.items():
            item[key] = to_nulls(value)
    return items


def process_variables(variables):
    return {
        key: value
        for key, value in variables.items()
        if type(value) in valida_types and not pd.isnull(value)
    }


def get_app(path):
    from fastapi import FastAPI, HTTPException

    logger = logging.getLogger(__name__)
    app = FastAPI()
    PIPELINE = "pipeline"

    pipeline = Pipeline.from_file(path)
    # A dynamic way to create a pydanic model based on the raw data
    raw = process_response(pipeline.raw)[0]
    Query = create_model(
        "Query",
        **{k: (type(v), None) for k, v in raw.items()},
        __config__=type("QueryConfig", (object,), {"schema_extra": {"example": raw}}),
    )

    def get_pipeline():
        return app.state._state.get(PIPELINE, pipeline)

    @app.post("/inference", response_model=List[dict])
    def inference(data: List[Query], columns: str = ""):
        logger.info("/inference")
        data = parse_query(data)
        if len(data) == 0:
            raise HTTPException(status_code=400, detail="No data provided")
        try:
            columns = None if not columns else columns.split(",")
            ret = get_pipeline().inference(data, columns=columns)
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=400, detail=str(e))

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

    return app


def get_wsgi_application(path):
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
            self.application.state._state[PIPELINE] = Pipeline.from_file(path)

        def load(self):
            return self.application

    return WSGIApplication


class Server:
    def __init__(self, path, options={}):
        self.path = path
        self.options = options
        self.app = get_app(path)
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
