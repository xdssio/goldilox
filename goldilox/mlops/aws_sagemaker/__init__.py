import os
import shutil
from typing import Union

import goldilox
import goldilox.mlops

SAGEMAKER = 'aws_sagemaker'
from goldilox.mlops.gunicorn import export_gunicorn


def export_sagemaker(pipeline: Union[goldilox.Pipeline, str], path: str) -> str:
    export_gunicorn(pipeline, path)
    goldilox.mlops.setup_environment(pipeline, path)

    for filename in ('train.py', 'serve'):
        src_path = os.path.join(goldilox.mlops.goldilox_path, goldilox.mlops.MLOPS, SAGEMAKER, filename)
        target_path = os.path.join(path, filename)
        print(src_path, target_path)
        shutil.copyfile(src_path, target_path)

    return path
