import os
import shutil
from typing import Union

import goldilox
import goldilox.mlops

GUNICORN = 'gunicorn'


def export_gunicorn(pipeline: Union[goldilox.Pipeline, str], path: str, nginx=False) -> str:
    try:
        import gunicorn
    except ImportError:
        raise ImportError("Please install gunicorn first")

    goldilox.mlops.setup_environment(pipeline, path)

    files = ['wsgi.py', 'gunicorn.conf.py']
    if nginx:
        files = files + ['nginx.conf', 'serve.py']
    for filename in files:
        src_path = os.path.join(goldilox.mlops.goldilox_path, goldilox.mlops.MLOPS, GUNICORN, filename)
        target_path = os.path.join(path, filename)
        shutil.copyfile(src_path, target_path)

    return path
