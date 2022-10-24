import contextlib
import gc
import json
import logging
import mmap
import os
import shutil
import sys
from pathlib import Path

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


def read_file(path):
    if path.endswith('.csv'):
        read = pd.read_csv
    elif path.endswith('.parquet'):
        read = pd.read_parquet
    elif path.endswith('.feather'):
        read = pd.read_feather
    elif path.endswith(('.hdf', '.hdf5', '.h5')):
        read = pd.read_hdf
    elif path.endswith(('.pkl', '.pickle', '.p')):
        read = pd.read_pickle
    elif path.endswith('.json'):
        try:
            with open(path, 'r') as f:
                path = json.load(f)
            read = pd.read_json
        except Exception as e:
            return None
    elif path.endswith('txt'):
        path = Path(path).read_text()
        read = pd.read_json
    else:
        return None
    with contextlib.suppress():
        return read(path)
    return None


def read_pandas_files(files):
    if len(files) == 0:
        logger.error(f"found no data files")
        return None
    bad_files, valid_files = [], []

    for path in files:
        df = read_file(path)
        if df is not None:
            valid_files.append(df)
        else:
            logger.error(f"file {path} is broken")
            bad_files.append(path)

    logger.info(f"found {len(bad_files)} malformed files")
    logger.info(f"opened {len(valid_files)} files")
    return pd.concat(valid_files, ignore_index=True)


def read_sklearn_data(path, prefix='', suffix='', shuffle=True):
    files = [str(path) for path in Path(path).rglob(f"{prefix}*{suffix}")]
    logger.info(f"relevant files: {files}")
    logger.info(f"found {len(files)} files")
    df = read_pandas_files(files)
    if df is not None and shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    if df is not None:
        logger.info(f"data shape {df.shape}")
    return df


def read_vaex_data(path, prefix='', suffix='', shuffle=True):
    import vaex
    logger.info(f"read data from {path} and prefix {prefix} and suffix {suffix}")
    if os.path.isdir(path):
        files = [str(path) for path in Path(path).rglob(f"{prefix}*{suffix}")]
        logger.info(f"relevant files: {files}")
        logger.info(f"found {len(files)} files")
        if len(files) == 0:
            logger.error(f"found no data files")
            return None

        vaex_files_paths, pandas_files, malformed_files = [], [], []

        for path in files:
            if not path.endswith(('arrow', 'csv', 'hdf5', 'parquet', 'feather')):
                df = read_file(path)
                if df is not None:
                    pandas_files.append(df)
                continue
            try:
                vaex.open(path).head(2)
                vaex_files_paths.append(path)
            except Exception as e:
                try:
                    pandas_files.append(read_file(path))
                except:
                    malformed_files.append(path)

        logger.info(
            f"found {len(malformed_files)} malformed files - {len(vaex_files_paths)} vaex files - {len(pandas_files)} pandas files")
        df = None
        if len(vaex_files_paths) > 0:
            df = vaex.open_many(vaex_files_paths)
        if pandas_files:
            from_pandas_df = vaex.from_pandas(pd.concat(pandas_files, ignore_index=True))
            if df is None:
                df = from_pandas_df
            else:
                df = vaex.concat([df, from_pandas_df])
    else:
        try:
            df = vaex.open(path, shuffle=shuffle)
        except:
            df = vaex.from_pandas(read_file(path))
            if shuffle:
                df = df.shuffle()
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


def remove_signeture(s):
    return s[len(CONSTANTS.BYTES_SIGNETURE):]


def add_signeture(s):
    return CONSTANTS.BYTES_SIGNETURE + s


def read_meta_bytes(path):
    open_fs = get_open(path)
    with open_fs(path, 'r+') as f:
        with mmap.mmap(f.fileno(), 0) as mf:
            offset = mf.find(CONSTANTS.BYTE_DELIMITER)
            if offset == -1:
                raise Exception(f"{CONSTANTS.BYTE_DELIMITER} not found")
        f.seek(offset)
    with open_fs(path, 'rb') as f:
        meta_bytes = f.read(offset)
    return meta_bytes


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
