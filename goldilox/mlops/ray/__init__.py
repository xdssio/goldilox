import os
import shutil
from typing import Union

import goldilox
from goldilox.config import CONSTANTS

RAY = 'ray'


def export_ray(pipeline: Union[goldilox.Pipeline, str], path: str) -> str:
    goldilox.mlops.setup_environment(pipeline, path)
    src_path = os.path.join(goldilox.mlops.goldilox_path, CONSTANTS.MLOPS, RAY, 'main.py')
    target_path = os.path.join(path, 'main.py')
    shutil.copyfile(src_path, target_path)
    return path
