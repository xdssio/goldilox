import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from goldilox.utils import read_sklearn_data, read_vaex_data


def write_files(dir_path, df):
    dir_path = Path(dir_path)
    df.to_csv(dir_path.joinpath('test.csv'), index=False)
    df.to_csv(dir_path.joinpath('test2.csv'), index=False)
    df.to_parquet(dir_path.joinpath('test.parquet'))
    df.to_feather(dir_path.joinpath('test.feather'))
    df.to_pickle(dir_path.joinpath('test.pkl'))
    with open(dir_path.joinpath('test.json'), 'w') as outfile:
        json.dump(df.to_json(), outfile)
    Path(dir_path.joinpath('test.txt')).write_text(df.to_json())


def test_read_sklearn_data():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': ['a', 'b', 'c']})
    directory = TemporaryDirectory()
    dir_path = directory.name
    write_files(dir_path, df)
    assert len(read_sklearn_data(dir_path)) == len(df) * 7
    assert len(read_sklearn_data(dir_path, suffix='csv')) == len(df) * 2
    assert len(read_sklearn_data(dir_path, suffix='parquet')) == len(df)
    assert read_sklearn_data(dir_path, suffix='blabla') is None


def test_read_vaex_data():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': ['a', 'b', 'c']})
    directory = TemporaryDirectory()
    dir_path = directory.name
    write_files(dir_path, df)

    assert len(read_vaex_data(dir_path)) == len(df) * 7
    assert len(read_vaex_data(dir_path, suffix='csv')) == len(df) * 2
    assert len(read_vaex_data(dir_path, suffix='parquet')) == len(df)
    assert read_vaex_data(dir_path, suffix='blabla') is None
