import gc
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import vaex
from goldilox.config import DEFAULT_SUFFIX


def _is_s3_url(path):
    if hasattr(path, 'dirname'):
        path = path.dirname
    return path.startswith('s3://')


logger = logging.getLogger()


# s3_client = boto3.client('s3')


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


def read_data(path, prefix=None, suffix=DEFAULT_SUFFIX):
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
