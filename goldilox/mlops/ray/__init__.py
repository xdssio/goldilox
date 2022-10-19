import os
import pathlib

from typing import List

import goldilox
from goldilox.config import CONSTANTS
from goldilox.utils import is_s3_url, get_conda_env, get_open, get_requirements

RAY = 'ray'


def export_ray(self, path: str,
               requirements: List[str] = None,
               appnope: bool = False) -> str:
    try:
        import ray
        from ray import serve
    except ImportError:
        raise ImportError("Ray is not installed - please install it with `pip install ray[serve]`")
    try:
        from starlette.requests import Request
    except ImportError:
        raise ImportError("Starlette is not installed - please install it with `pip install starlette`")

    if not is_s3_url(path):
        os.makedirs(path, exist_ok=True)

    env = get_conda_env()

    def write_file(filename: str, content: str):
        full_path = os.path.join(path, filename)
        filesystem_open = get_open(full_path)
        with filesystem_open(full_path, 'w') as outfile:
            outfile.write(content)

    if requirements:
        filename = 'requirements.txt'
    else:
        filename, env = get_requirements(appnope=appnope)
    write_file(filename, env)
    filename = 'main.py'
    self.save(os.path.join(path, 'pipeline.pkl'))
    goldilox_path = str(pathlib.Path(goldilox.__file__).parent.absolute())
    local_path = os.path.join(goldilox_path, CONSTANTS.MLOPS, RAY, filename)
    file_text = pathlib.Path(local_path).read_text()
    open_fs = get_open(path)
    with open_fs(os.path.join(path, filename), 'w') as outfile:
        outfile.write(file_text)

    return path
