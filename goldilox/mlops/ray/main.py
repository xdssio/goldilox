import argparse
import json
import logging
import os
import time

import ray
from fastapi import FastAPI, HTTPException
from ray import serve
from starlette.requests import Request

from goldilox import Pipeline
from goldilox.app import process_response
from goldilox.utils import process_variables

logger = logging.getLogger(__name__)
app = FastAPI()

PIPELINE = 'pipeline.pkl'
DEPLOYMENT = 'deployment'
API = 'api'
RAY_PARAMS = 'RAY_PARAMS'

deployment_params, api_params = {}, {}

meta = Pipeline.load_meta(PIPELINE)
if RAY_PARAMS in meta:
    params = meta.get(RAY_PARAMS, {})
    deployment_params = params.get(DEPLOYMENT, {})
    api_params = params.get(API, {})

if os.path.isfile('deployment_params.json'):
    with open('deployment_params.json', 'r') as infile:
        deployment_params.update(json.load(infile))
if os.path.isfile('api_params.json'):
    with open('api_params.json', 'r') as infile:
        api_params.update(json.load(infile))


@serve.deployment(**deployment_params)
@serve.ingress(app)
class PipelineDeployment:

    def __init__(self):
        self.pipeline = Pipeline.from_file(PIPELINE)

    @app.post("/predict")
    async def predict(self, request: Request) -> str:
        data = await request.json()
        ret = self.pipeline.inference(data)
        return process_response(ret)

    @app.post("/inference")
    async def inference(self, request: Request, columns: str = ""):
        logger.info("/inference")
        data = await request.json()
        if len(data) == 0:
            raise HTTPException(status_code=400, detail="No data provided")
        try:
            columns = None if not columns else columns.split(",")
            ret = self.pipeline.inference(data, columns=columns)
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=400, detail=str(
                f"Issue with inference, try runing your pipeline locally with 'pipeline.inference(data)' to see what is the problem:\n{e}"))

        return process_response(ret)

    @app.get("/variables")
    def variables(self):
        logger.info("/variables")
        return process_variables(self.pipeline.variables)

    @app.get("/description")
    def description(self):
        logger.info("/description")
        return self.pipeline.description

    @app.get("/example")
    def example(self):
        logger.info("/example")
        return process_response(self.pipeline.example)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--address', help='Ray init address', default=None)
    parser.add_argument('--detached',
                        help="Whether not the instance should be detached from this script. "
                             "If set, the instance will live on the Ray cluster until it is explicitly stopped with serve.shutdown()",
                        default=False)
    parser.add_argument('--dedicated_cpu',
                        help="Whether to reserve a CPU core for the internal Serve controller actor.  Defaults to False",
                        default=False)
    parser.add_argument('--host',
                        help='Host for HTTP servers to listen on. Defaults to "127.0.0.1". To expose Serve publicly, '
                             'you probably want to set this to "0.0.0.0"',
                        default=None)
    parser.add_argument('--port',
                        help='Port for HTTP server. Defaults to 8000',
                        default=8080)
    parser.add_argument('--num_cpus',
                        help='The number of CPU cores to reserve for each internal Serve HTTP proxy actor',
                        default=0)

    args = parser.parse_args()
    if args.address:
        ray.init(address=args.address)
    else:
        ray.init()

    http_options = {}
    if args.host:
        http_options['host'] = args.host
    if args.port:
        http_options['port'] = args.port
    serve.start(detached=args.detached,
                num_cpus=args.num_cpus,
                http_options=http_options,
                dedicated_cpu=args.dedicated_cpu)
    PipelineDeployment.deploy()
    while True:
        time.sleep(1)
