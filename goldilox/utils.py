import gc
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from sys import version_info

import numpy as np
import pandas as pd

from goldilox.config import CONSTANTS

valida_types = {type(None), dict, list, int, float, str, bool}


def is_s3_url(path):
    if hasattr(path, 'dirname'):
        path = path.dirname
    return path.startswith('s3://')


logger = logging.getLogger()


def open_many(paths):
    import vaex
    dfs = [vaex.open(path) for path in paths]
    valid_eips = set()
    info_columns = set()
    for df in dfs:
        eips = df.get_column_names(regex='eip')
        for column in eips:
            if len(df[column].as_numpy().unique()) > 1:
                valid_eips.add(column)
        info_columns.update(df.get_column_names(regex='^(?!eip_).+'))
    columns = list(info_columns) + list(valid_eips)

    for column in columns:
        for df, p in zip(dfs, paths):
            if column not in df:
                logger.info(f"added missing column {column} to {p}")
                df[column] = np.full(shape=len(df), fill_value=None)
    pandas = [df[columns].to_pandas_df() for df in dfs]
    for df in pandas:
        for column in valid_eips:
            df[column] = df[column].astype(str)
    concat = pd.concat(pandas)
    concat_vaex = vaex.from_pandas(concat)
    del concat
    del dfs
    del pandas
    gc.collect()
    return concat_vaex


def read_data(path, prefix='', suffix=''):
    import vaex
    prefix = prefix or ''
    logger.info(f"read data from {path} and prefix {prefix} and suffix {suffix}")
    if os.path.isdir(path):
        files = [str(path) for path in Path(path).rglob(f"{prefix}*{suffix}")]
        logger.info(f"relevant files: {files}")
        logger.info(f"found {len(files)} files")
        if len(files) == 0:
            logger.error(f"found no data files")
            return None

        bad_files = []
        for file in files:
            try:
                if vaex.open(file).head(2):
                    pass
            except:
                logger.error(f"file {file} is broken")
                bad_files.append(file)

        paths = list(set([file for file in files if file not in bad_files]))
        logger.info(f"found {len(bad_files)} malformed files")
        logger.info(f"open {len(paths)} files")
        df = vaex.open_many(paths)
    else:
        df = vaex.open(path, shuffle=True)
    logger.info(f"data shape {df.shape}")
    return df


def is_list(l):
    return isinstance(l, (list, pd.core.series.Series, np.ndarray))


def to_nulls(value):
    if is_list(value):
        return [to_nulls(v) for v in value]
    elif isinstance(value, dict):
        return {to_nulls(k): to_nulls(v) for k, v in value.items()}
    elif hasattr(value, 'tolist'):
        return to_nulls(value.tolist())
    elif pd.isnull(value):
        return None
    return value


def process_variables(variables):
    return {
        key: to_nulls(value)
        for key, value in variables.items()
        if (type(value) in valida_types or hasattr(value, 'tolist'))
    }


def get_git_info():
    from git import Repo
    from git.cmd import Git
    return {'branch': Repo().active_branch.name,
            'remote': Git().remote(verbose=True).split('\t')[1].split(' ')[0]}


def get_goldilox_path():
    import goldilox
    return Path(goldilox.__file__)


def validate_path(path):
    """
    Make sure there is an empty dir there
    @param path: path to validate
    @return:
    """
    ret = True
    try:
        if "/" in path:
            os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
    except AttributeError as e:
        ret = False
    if os.path.isdir(path):
        shutil.rmtree(path)
    return ret


def read_bytes(path):
    open_fs = get_open(path)
    with open_fs(path, 'rb') as f:
        ret = f.read()
    return ret


def write_bytes(path, bytes_to_write):
    open_fs = get_open(path)
    with open_fs(path, "wb") as outfile:
        outfile.write(bytes_to_write)
    return path


def get_open(path):
    open_fs = open
    if is_s3_url(path):
        import s3fs
        fs = s3fs.S3FileSystem(profile=CONSTANTS.AWS_PROFILE)
        open_fs = fs.open
    return open_fs


def unpickle(b):
    try:
        import cloudpickle
        ret = cloudpickle.loads(b)
    except:
        try:
            import pickle
            ret = pickle.loads(b)
        except:
            try:
                import pickle5
                ret = pickle5.loads(b)
            except Exception as e:
                raise RuntimeError("Could not unpickle")
    return ret


def get_requirements(venv_type=None, requirements=None, clean=True):
    """Run pip freeze  for venv and conda env export for conda
    @return requirements
    """
    if requirements is not None:
        return '\n'.join(requirements)
    if venv_type is None:
        venv_type = get_env_type()
    if venv_type == 'conda':
        command = ["conda env export | cut -f 1 -d '=' "]
        env = subprocess.check_output(command, shell=True).decode()
        if clean:
            env = env.replace('\n  - appnope', '')
        splited = env.split('\n')
        splited[0] = 'name: conda_env'
        splited[-2] = 'prefix: conda_env'
        return '\n'.join(splited)
    ret = subprocess.check_output([sys.executable, '-m', 'pip',
                                   'freeze']).decode()
    if clean:
        import re
        ret = re.sub("appnope==(.[\d \.]*)\\n", '', ret)
    return ret


def get_python_version():
    """
    @return: current python version
    """
    return "{major}.{minor}.{micro}".format(major=version_info.major,
                                            minor=version_info.minor,
                                            micro=version_info.micro)


def get_env_type():
    """
    @return 'conda' if running in a conda environment, 'venv' if in a virtual env, and None otherwise
    """
    if os.getenv('CONDA_DEFAULT_ENV'):
        return 'conda'
    elif os.getenv('VIRTUAL_ENV'):
        return 'venv'
    return None


def get_conda_env(clean=True):
    """run conda env export | cut -f 1 -d '='  and clean problematic packages (for docker) like 'appnope'"""
    env = None
    if get_env_type() == 'conda':
        command = ["conda env export | cut -f 1 -d '=' "]
        env = subprocess.check_output(command, shell=True).decode()
        if clean:
            env = env.replace('\n  - appnope', '')
        splited = env.split('\n')
        splited[0] = 'name: conda_env'
        splited[-2] = 'prefix: conda_env'
        env = '\n'.join(splited)
    return env
