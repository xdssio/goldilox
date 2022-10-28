import contextlib
import os
import pathlib
import shutil

import goldilox
from goldilox.mlops.gunicorn import export_gunicorn
from goldilox.mlops.mlflow import export_mlflow
from goldilox.mlops.ray import export_ray
from goldilox.utils import is_cloud_url

MLOPS = 'mlops'
PIPELINE_FILE = 'pipeline.pkl'
ENVIRONMENT_YAML = 'environment.yml'
REQUIREMENTS_TXT = 'requirements.txt'
CONDA = 'conda'

goldilox_path = str(pathlib.Path(goldilox.__file__).parent.absolute())


def write_pipeline(pipeline, path):
    target_path = os.path.join(path, PIPELINE_FILE)
    with contextlib.suppress(shutil.SameFileError):
        if isinstance(pipeline, str):
            shutil.copyfile(pipeline, target_path)
        else:
            pipeline.save(target_path)
        return True
    return False


def setup_environment(pipeline, path):
    if not is_cloud_url(path):
        os.makedirs(path, exist_ok=True)
    meta = goldilox.Meta.from_file(pipeline) if isinstance(pipeline, str) else pipeline.meta
    pathlib.Path(os.path.join(path, meta.environment_filename)).write_text(meta.env_file)
    return goldilox.mlops.write_pipeline(pipeline, path)
