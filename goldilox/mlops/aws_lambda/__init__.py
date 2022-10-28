import os
import shutil
from typing import Union

import goldilox
import goldilox.mlops

LAMBDA = 'aws_lambda'


def export_lambda(pipeline: Union[goldilox.Pipeline, str], path: str) -> str:
    goldilox.mlops.setup_environment(pipeline, path)
    files = ('handler.py', 'entrypoint.sh', "aws-lambda-rie")

    for filename in files:
        src_path = os.path.join(goldilox.mlops.goldilox_path, goldilox.mlops.MLOPS, LAMBDA, filename)
        target_path = os.path.join(path, filename)
        shutil.copyfile(src_path, target_path)
    # shutil.copyfile(os.path.join(goldilox.mlops.goldilox_path, goldilox.mlops.MLOPS, LAMBDA, 'aws-lambda-rie'),
    #                 '/usr/bin/aws_lambda-rie')

    return path
