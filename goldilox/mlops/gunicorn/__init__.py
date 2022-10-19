import os
import pathlib

from typing import List

import goldilox
from goldilox.utils import is_s3_url, get_requirements, get_open

MLOPS = 'mlops'
GUNICORN = 'gunicorn'


def export_gunicorn(self, path: str, requirements: List[str] = None, nginx=False, appnope=False) -> str:
    try:
        import gunicorn
    except ImportError:
        raise ImportError("Please install gunicorn first")

    if not is_s3_url(path):
        os.makedirs(path, exist_ok=True)
    if requirements:
        filename = 'requirements.txt'
    else:
        filename, requirements = get_requirements(appnope=appnope)
    open_fs = get_open(path)
    with open_fs(os.path.join(path, filename), 'w') as outfile:
        outfile.write(requirements)

    self.save(os.path.join(path, 'pipeline.pkl'))
    goldilox_path = str(pathlib.Path(goldilox.__file__).parent.absolute())
    files = ['wsgi.py', 'gunicorn.conf.py']
    if nginx:
        files = files + ['nginx.conf', 'serve.py']
    for filename in files:
        local_path = os.path.join(goldilox_path, MLOPS, GUNICORN, filename)
        file_text = pathlib.Path(local_path).read_text()
        with open_fs(os.path.join(path, filename), 'w') as outfile:
            outfile.write(file_text)

    return path
