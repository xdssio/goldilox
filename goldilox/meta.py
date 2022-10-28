import json
import os
import pathlib
import subprocess
import sys
import time

import yaml

import goldilox
from goldilox.config import CONSTANTS
from goldilox.utils import read_meta_bytes, remove_signeture, unpickle


def get_env_type():
    """
    @return 'conda' if running in a conda environment, 'venv' if in a virtual env, and None otherwise
    """
    if os.getenv('CONDA_DEFAULT_ENV'):
        return 'conda'
    elif os.getenv('VIRTUAL_ENV'):
        return 'venv'
    return None


class Environment:
    env_file: str
    env_filename: str
    py_version: str
    goldilox_version: str

    def set_requerments(self):
        pass

    def set_environemnt(self):
        pass


class Meta:

    def __init__(self, pipeline_type,
                 raw: dict = {},
                 example: dict = {},
                 variables: dict = {},
                 goldilox_version: str = goldilox.__version__,
                 py_version: str = None,
                 env_file: str = None,
                 environment_filename: str = None,
                 fastapi_params: dict = {},
                 env_type=None, appnope=False):
        current_time = int(time.time())
        self.created = current_time
        self.updated = current_time
        self.raw = raw
        self.example = example
        self.fastapi_params = fastapi_params
        self.pipeline_type = pipeline_type
        self.variables = variables
        self.py_version = py_version or self.get_python_version()
        self.goldilox_version = goldilox_version
        self.env_type = env_type or get_env_type()
        if env_file is None:
            environment_filename, env_file = self.get_requirements(venv_type=env_type, appnope=appnope)
        if environment_filename is None:
            environment_filename = 'requirements.txt' if self.env_type == CONSTANTS.VENV else 'environment.yml'
        self.environment_filename = environment_filename
        self.env_file = env_file

    def get_requirements(self, venv_type=None, appnope=False):
        """Run pip freeze  for venv and conda env export for conda
        @return requirements
        """
        venv_type = venv_type or self.env_type
        if venv_type == CONSTANTS.CONDA:

            env = subprocess.check_output(["conda env export | cut -f 1 -d '=' "], shell=True).decode()
            if not env:
                env = subprocess.check_output(["micromamba env export"], shell=True).decode()
            env = yaml.safe_load(env)
            env['dependencies'] = [f"python={self.py_version}"] + env['dependencies']
            env['name'] = 'base'
            env['prefix'] = 'base'
            return 'environment.yml', json.dumps(env)
        ret = subprocess.check_output([sys.executable, '-m', 'pip',
                                       'freeze']).decode()
        if appnope is False:
            import re
            ret = re.sub("appnope==(.[\d \.]*)\\n", '', ret)
        return 'requirements.txt', ret

    def __repr__(self):
        return f"Meta({json.dumps({CONSTANTS.PIPELINE_TYPE: self.pipeline_type, CONSTANTS.VENV_TYPE: self.env_type, CONSTANTS.VERSION: self.goldilox_version, CONSTANTS.VARIABLES: self.variables, CONSTANTS.DESCRIPTION: self.description, CONSTANTS.RAW: self.raw}, indent=4)})"

    @property
    def description(self) -> str:
        return self.variables.get(CONSTANTS.DESCRIPTION, '')

    def to_dict(self):
        return {
            CONSTANTS.PIPELINE_TYPE: self.pipeline_type,
            CONSTANTS.GOLDILOX_VERSION: self.goldilox_version,
            CONSTANTS.PY_VERSION: self.py_version,
            CONSTANTS.VENV_TYPE: self.env_type,
            CONSTANTS.REQUIREMEMTS: self.env_file,
            CONSTANTS.VARIABLES: self.variables.copy(),
            CONSTANTS.DESCRIPTION: self.description,
            CONSTANTS.RAW: self.raw,
            CONSTANTS.EXAMPLE: self.example,
            CONSTANTS.FASTAPI_PARAMS: self.fastapi_params,
            CONSTANTS.ENVIRONMENT_FILENAME: self.environment_filename,

        }

    @classmethod
    def from_dict(self, meta_dict):
        return Meta(pipeline_type=meta_dict.get(CONSTANTS.PIPELINE_TYPE),
                    goldilox_version=meta_dict.get(CONSTANTS.GOLDILOX_VERSION),
                    py_version=meta_dict.get(CONSTANTS.PY_VERSION),
                    env_type=meta_dict.get(CONSTANTS.VENV_TYPE, self.get_python_version()),
                    env_file=meta_dict.get(CONSTANTS.REQUIREMEMTS, ""),
                    variables=meta_dict.get(CONSTANTS.VARIABLES, {}),
                    raw=meta_dict.get(CONSTANTS.RAW, {}),
                    example=meta_dict.get(CONSTANTS.EXAMPLE, {}),
                    fastapi_params=meta_dict.get(CONSTANTS.FASTAPI_PARAMS, {}),
                    environment_filename=meta_dict.get(CONSTANTS.ENVIRONMENT_FILENAME, ""))

    @classmethod
    def from_file(cls, path):
        meta_bytes = remove_signeture(read_meta_bytes(path))
        return cls.from_dict(unpickle(meta_bytes))

    def get_conda_environment(self):
        if self.env_type == CONSTANTS.CONDA:
            return yaml.safe_load(self.env_file)

        return {"name": "base", "channels": ["conda-forge"], "prefix": "base",
                "dependencies": [f"python={self.py_version}", {"pip": self.env_file.split('\n')}]}

    def write_environment_file(self, output):
        if output is None:
            output = self.environment_filename
        pathlib.Path(output).write_text(self.env_file)
        return output

    @staticmethod
    def get_python_version():
        """
        @return: current python version
        """
        return "{major}.{minor}.{micro}".format(major=sys.version_info.major,
                                                minor=sys.version_info.minor,
                                                micro=sys.version_info.micro)

    @property
    def requirements(self):
        if self.env_type == CONSTANTS.CONDA:
            return yaml.safe_load(self.env_file)
        return self.env_file.split('\n')

    def set_requirements(self, requirements):
        if self.env_type == CONSTANTS.CONDA:
            conda_env = yaml.safe_load(self.env_file)
            if isinstance(requirements, dict):
                conda_env = requirements
            else:
                conda_env["dependencies"][-1]["pip"] = requirements
            self.env_file = json.dumps(conda_env)
        else:
            self.env_file = '\n'.join(requirements)
