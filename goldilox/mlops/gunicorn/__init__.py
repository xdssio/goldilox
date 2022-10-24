import os
import shutil
from typing import Union

import goldilox
import goldilox.mlops

GUNICORN = 'gunicorn'


def export_gunicorn(pipeline: Union[goldilox.Pipeline, str], path: str) -> str:
    goldilox.mlops.setup_environment(pipeline, path)
    files = ['wsgi.py', 'gunicorn.conf.py', 'nginx.conf']

    for filename in files:
        src_path = os.path.join(goldilox.mlops.goldilox_path, goldilox.mlops.MLOPS, GUNICORN, filename)
        target_path = os.path.join(path, filename)
        shutil.copyfile(src_path, target_path)

    return path
