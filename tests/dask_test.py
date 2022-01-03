import dask.dataframe as dd
import pandas as pd
import vaex
from dask.threaded import get

from goldilox import Pipeline

csv = dd.read_csv(['data/titanic.csv', 'data/titanic2.csv'])
csv.dask.to_dict()
parquet = dd.read_parquet('data/iris.parquet')
parquet['sepal_length'] = parquet['sepal_length'].astype('float')

df = dd.from_pandas(vaex.example().to_pandas_df(), npartitions=1)
raw = Pipeline.to_raw(df.head())

original_load_key = list({key: value for key, value in df.__dask_graph__().dependencies.items() if len(value) == 0})[0]
df['x1'] = df.x + 1
df = df[df['x1'] > 0]
df = df[df['x'] < 5]
df['xx'] = df['x1'] + 2
# df['dict'] = df.x.apply(lambda x: {'x': x}, meta='object')
df.head(1)

saved_graph = df.__dask_graph__()

#### Build new graph
df2 = dd.from_pandas(vaex.example().to_pandas_df().head(), npartitions=1)

illegal_prefix = ['gt-', 'lt-', 'ge-', 'le-']
new_load_key = list({key: value for key, value in df2.__dask_graph__().dependencies.items() if len(value) == 0})[0]

original_dict = saved_graph.to_dict()


def process_key(key):
    if key == (original_load_key, 0):
        return (new_load_key, 0)
    return key


def process_values(values):
    if isinstance(values, pd.DataFrame):
        # return 'VALUES'
        # if values == 'VALUES':
        return df2
    print(f"originals: {values}")
    if isinstance(values, pd.DataFrame):
        return values
    ret = []
    for v in values:
        if isinstance(v, tuple) and len(v) > 0:
            if v[0] == original_load_key:
                ret.append((new_load_key, v[1]))
            # if any([v[0].startswith(prefix) for prefix in illegal_prefix]):
            #     continue
            else:
                ret.append(v)
        else:
            ret.append(v)
    ret = tuple(ret)
    print(f"processed: {ret}")
    return ret


def to_values(values):
    if isinstance(values, pd.DataFrame):
        return df2.copy()
    return values


# apply_dict = {process_key(key): process_values(value) for key, value in original_dict.items()}
apply_dict = {key: to_values(values) for key, values in original_dict.items()}

ret = get(apply_dict, result=('add-288d6a05e920bafa5ff156619c08dbab', 0))
ret.values
