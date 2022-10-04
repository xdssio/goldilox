import os

from goldilox.app import get_app

app = get_app(os.getenv('PIPELINE_PATH', 'pipeline.pkl'))
