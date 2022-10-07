import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from goldilox.utils import read_sklearn_data, read_vaex_data


def test_read_sklearn_data():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': ['a', 'b', 'c']})
    directory = TemporaryDirectory()
    dir_path = Path(directory.name)
    df.to_csv(dir_path.joinpath('test.csv'), index=False)
    df.to_csv(dir_path.joinpath('test2.csv'), index=False)
    df.to_parquet(dir_path.joinpath('test.parquet'))
    df.to_feather(dir_path.joinpath('test.feather'))
    df.to_pickle(dir_path.joinpath('test.pkl'))
    with open(dir_path.joinpath('test.json'), 'w') as outfile:
        json.dump(df.to_json(), outfile)
    Path(dir_path.joinpath('test.txt')).write_text(df.to_json())
    assert len(read_sklearn_data(dir_path)) == len(df) * 7


def test_read_vaex_data():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': ['a', 'b', 'c']})
    directory = TemporaryDirectory()
    dir_path = Path(directory.name)
    df.to_csv(dir_path.joinpath('test.csv'), index=False)
    df.to_csv(dir_path.joinpath('test2.csv'), index=False)
    df.to_parquet(dir_path.joinpath('test.parquet'))
    df.to_feather(dir_path.joinpath('test.feather'))
    df.to_pickle(dir_path.joinpath('test.pkl'))
    with open(dir_path.joinpath('test.json'), 'w') as outfile:
        json.dump(df.to_json(), outfile)
    Path(dir_path.joinpath('test.txt')).write_text(df.to_json())
    assert len(read_vaex_data(dir_path)) == len(df) * 7
