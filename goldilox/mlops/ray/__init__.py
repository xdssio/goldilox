import json
import os
import shutil
from pathlib import Path
from typing import List

import goldilox
from goldilox.utils import is_s3_url, get_conda_env, get_open, get_requirements

MLOPS = 'mlops'
RAY = 'ray'


def copy_ray_file(base: str, path: str, filename: str) -> bool:
    shutil.copyfile(str(base.parent.absolute().joinpath(MLOPS).joinpath(RAY).joinpath(filename)),
                    os.path.join(path, filename))
    return True


def export_ray(self, path: str, requirements: List[str] = None, clean: bool = True, **kwargs) -> str:
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
    open_fs = get_open(path)
    if env is not None:
        with open_fs(os.path.join(path, ' environment.yml'), 'w') as outfile:
            outfile.write(env)
    else:
        with open_fs(os.path.join(path, 'requirements.txt'), 'w') as outfile:
            outfile.write(get_requirements(requirements, clean=clean))

    try:
        with open_fs(os.path.join(path, 'params.json'), 'w') as outfile:
            json.dump(kwargs, outfile)
    except Exception as e:
        raise RuntimeError(f"Could not write params.json: {e}")

    self.save(os.path.join(path, 'pipeline.pkl'))
    goldilox_path = Path(goldilox.__file__)
    copy_ray_file(goldilox_path, path, 'main.py')

    return path
