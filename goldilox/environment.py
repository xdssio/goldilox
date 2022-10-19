import os
import subprocess
import sys

import goldilox


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


class Environment:
    filename: str = 'requirements.txt'
    env_type: str = 'venv'
    env_file: str = ''
    py_version: str = ''
    CONDA = 'conda'
    VENV = 'venv'

    def __init__(self, venv_type=None, appnope=False):
        self.py_version = get_python_version()
        self.env_type = venv_type or get_env_type()
        filename, env_file = self.get_requirements(venv_type=venv_type, appnope=appnope)
        self.filename = filename
        self.env_file = env_file

    def from_meta(self, meta):
        obj = Environment()

    def get_requirements(self, venv_type=None, appnope=False):
        """Run pip freeze  for venv and conda env export for conda
        @return requirements
        """
        venv_type = venv_type or self.env_type
        if venv_type == Environment.CONDA:
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

    def export(self, path):
        open_fs = goldilox.utils.get_open(path)
        with open_fs(os.path.join(path, self.filename), 'w') as outfile:
            outfile.write(self.env_file)

    def __repr__(self):
        return self.env_file
