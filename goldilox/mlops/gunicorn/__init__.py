import os
from pathlib import Path
from typing import List

import goldilox
from goldilox.mlops.utils import copy_file
from goldilox.utils import is_s3_url, get_conda_env, get_open, get_requirements

GUNICORN = 'gunicorn'


def export_gunicorn(self, path: str, requirements: List[str] = None, clean: bool = True) -> str:
    try:
        import gunicorn
    except ImportError:
        raise ImportError("Please install gunicorn first")

    if not is_s3_url(path):
        os.makedirs(path, exist_ok=True)
    env = get_conda_env()
    open_fs = get_open(path)
    if env is not None:
        with open_fs(os.path.join(path, ' environment.yml'), 'w') as outfile:
            outfile.write(env)
    else:
        with open_fs(os.path.join(path, 'requirements.txt'), 'w') as outfile:
            outfile.write(get_requirements(requirements, clean=clean))
    self.save(os.path.join(path, 'pipeline.pkl'))
    goldilox_path = Path(goldilox.__file__)
    copy_file(goldilox_path, path, 'main.py', GUNICORN)
    copy_file(goldilox_path, path, 'gunicorn.conf.py', GUNICORN)

    return path
