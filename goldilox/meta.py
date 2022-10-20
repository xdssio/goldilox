import json
import os
import subprocess
import sys
import time

import yaml

import goldilox
from goldilox.config import CONSTANTS
from goldilox.utils import read_meta_bytes, remove_signeture, unpickle


def get_python_version():
    """
    @return: current python version
    """
    return "{major}.{minor}.{micro}".format(major=sys.version_info.major,
                                            minor=sys.version_info.minor,
                                            micro=sys.version_info.micro)


def get_env_type():
    """
    @return 'conda' if running in a conda environment, 'venv' if in a virtual env, and None otherwise
    """
    if os.getenv('CONDA_DEFAULT_ENV'):
        return 'conda'
    elif os.getenv('VIRTUAL_ENV'):
        return 'venv'
    return None


class Meta:

    def __init__(self, pipeline_type,
                 raw: dict = {},
                 variables: dict = {},
                 venv_type=None, appnope=False):
        current_time = int(time.time())
        self.created = current_time
        self.updated = current_time
        self.raw = raw
        self.pipeline_type = pipeline_type
        self.variables = variables
        self.py_version = get_python_version()
        self.goldilox_version = goldilox.__version__
        self.venv_type = venv_type or get_env_type()
        environment_filename, env_file = self.get_requirements(venv_type=venv_type, appnope=appnope)
        self.environment_filename = environment_filename
        self.env_file = env_file

    def get_requirements(self, venv_type=None, appnope=False):
        """Run pip freeze  for venv and conda env export for conda
        @return requirements
        """
        venv_type = venv_type or self.venv_type
        if venv_type == CONSTANTS.CONDA:
            command = ["conda env export | cut -f 1 -d '=' "]
            env = subprocess.check_output(command, shell=True).decode()
            if appnope is False:
                env = env.replace('\n  - appnope', '')
            splited = env.split('\n')
            splited[0] = 'name: conda_env'
            splited[-2] = 'prefix: conda_env'
            return 'environment.yml', '\n'.join(splited)
        ret = subprocess.check_output([sys.executable, '-m', 'pip',
                                       'freeze']).decode()
        if appnope is False:
            import re
            ret = re.sub("appnope==(.[\d \.]*)\\n", '', ret)
        return 'requirements.txt', ret

    def __repr__(self):
        return f"Meta({json.dumps({CONSTANTS.PIPELINE_TYPE: self.pipeline_type, CONSTANTS.VENV_TYPE: self.venv_type, CONSTANTS.VERSION: self.goldilox_version, CONSTANTS.VARIABLES: self.variables, CONSTANTS.DESCRIPTION: self.description, CONSTANTS.RAW: self.raw}, indent=4)})"

    @property
    def description(self) -> str:
        return self.variables.get(CONSTANTS.DESCRIPTION, '')

    def to_dict(self):
        return {
            CONSTANTS.PIPELINE_TYPE: self.pipeline_type,
            CONSTANTS.VERSION: self.goldilox_version,
            CONSTANTS.PY_VERSION: self.py_version,
            CONSTANTS.VENV_TYPE: self.venv_type,
            CONSTANTS.REQUIREMEMTS: self.env_file,
            CONSTANTS.VARIABLES: self.variables.copy(),
            CONSTANTS.DESCRIPTION: self.description,
            CONSTANTS.RAW: self.raw,
            CONSTANTS.ENVIRONMENT_FILENAME: self.environment_filename,

        }

    @classmethod
    def from_dict(self, meta_dict):
        return Meta(pipeline_type=meta_dict.get(CONSTANTS.PIPELINE_TYPE),
                    goldilox_version=meta_dict.get(CONSTANTS.VERSION),
                    py_version=meta_dict.get(CONSTANTS.PY_VERSION),
                    env_type=meta_dict.get(CONSTANTS.VENV_TYPE, get_python_version()),
                    env_file=meta_dict.get(CONSTANTS.REQUIREMEMTS, ""),
                    variables=meta_dict.get(CONSTANTS.VARIABLES, {}),
                    raw=meta_dict.get(CONSTANTS.RAW, {}),
                    environment_filename=meta_dict.get(CONSTANTS.ENVIRONMENT_FILENAME, ""))

    @classmethod
    def from_file(cls, path):
        meta_bytes = remove_signeture(read_meta_bytes(path))
        return cls.from_dict(unpickle(meta_bytes))

    def get_conda_environment(self):
        if self.venv_type == CONSTANTS.CONDA:
            conda_env = yaml.safe_load(self.env_file)
            conda_dependencies = conda_env['dependencies']
            conda_env['dependencies'] = [f"python={self.py_version}"] + conda_dependencies
            return conda_env
        return {
            'channels': ['defaults'],
            'dependencies': [
                f"python={self.py_version}",
                {
                    'pip': self.env_file.split('\n'),
                },
            ],
            'name': 'goldilox_env'
        }
