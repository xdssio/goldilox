import os
from pathlib import Path
from typing import List

import goldilox
from goldilox.mlops.utils import copy_file
from goldilox.utils import is_s3_url, get_conda_env, get_open, get_requirements

RAY = 'ray'


def copy_ray_file(base: str, path: str, filename: str) -> bool:
    return copy_file(base, path, filename, RAY)


def export_ray(self, path: str,
               requirements: List[str] = None,
               clean: bool = True) -> str:
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

    if env is not None:
        filename = 'environment.yml'

    else:
        filename = 'requirements.txt'
        env = get_requirements(requirements=requirements, clean=clean)
    write_file(filename, env)

    self.save(os.path.join(path, 'pipeline.pkl'))
    goldilox_path = Path(goldilox.__file__)
    copy_file(goldilox_path, path, 'main.py', RAY)

    return path
