import os

from goldilox.app import get_app

app = get_app(os.environ['PIPELINE_PATH'])
