import os
import pathlib
import shutil

import goldilox
from goldilox.mlops.gunicorn import export_gunicorn
from goldilox.mlops.mlflow import export_mlflow
from goldilox.mlops.ray import export_ray
from goldilox.utils import is_s3_url

MLOPS = 'mlops'
PIPELINE_FILE = 'pipeline.pkl'
ENVIRONMENT_YAML = 'environment.yml'
REQUIREMENTS_TXT = 'requirements.txt'
CONDA = 'conda'

goldilox_path = str(pathlib.Path(goldilox.__file__).parent.absolute())


def write_pipeline(pipeline, path):
    target_path = os.path.join(path, PIPELINE_FILE)
    if isinstance(pipeline, str):
        shutil.copyfile(pipeline, target_path)
    else:
        pipeline.save(target_path)
    return path


def setup_environment(pipeline, path):
    if not is_s3_url(path):
        os.makedirs(path, exist_ok=True)
    meta = goldilox.Pipeline.load_meta(pipeline) if isinstance(pipeline, str) else pipeline.meta
    filename = ENVIRONMENT_YAML if meta.get(goldilox.config.CONSTANTS.VENV_TYPE) == CONDA else REQUIREMENTS_TXT
    pathlib.Path(os.path.join(path, filename)).write_text(meta.get(goldilox.config.CONSTANTS.REQUIREMEMTS, ''))
    goldilox.mlops.write_pipeline(pipeline, path)
