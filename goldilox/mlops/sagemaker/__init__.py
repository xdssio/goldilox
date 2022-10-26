import os
import shutil
from typing import Union

import goldilox
import goldilox.mlops

SAGEMAKER = 'sagemaker'
from goldilox.mlops.gunicorn import export_gunicorn


def export_gunicorn(pipeline: Union[goldilox.Pipeline, str], path: str) -> str:
    export_gunicorn(pipeline, path)
    goldilox.mlops.setup_environment(pipeline, path)

    for filename in ('serve', 'train', 'training.py'):
        src_path = os.path.join(goldilox.mlops.goldilox_path, goldilox.mlops.MLOPS, SAGEMAKER, 'program', filename)
        target_path = os.path.join(path, filename)
        shutil.copyfile(src_path, target_path)

    return path
