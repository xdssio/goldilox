import json
import os
from urllib.request import Request

from ray import serve

from goldilox import Pipeline

params = {}
if os.path.isfile('params.json'):
    with open('params.json', 'r') as infile:
        params = json.load(infile)


@serve.deployment(**params)
class Model:

    def __init__(self):
        self.model = Pipeline.from_file('pipeline.pkl')

    def inference(self, data) -> str:
        return self.model.inference(data)

    async def __call__(self, http_request: Request) -> str:
        data = await http_request.json()
        return self.inference(data)


model = Model.bind()
