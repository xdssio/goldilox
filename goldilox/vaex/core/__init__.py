import numpy as np
import pandas as pd
import vaex
import json

@vaex.register_dataframe_accessor('gl', override=True)
class GoldiloxExternsion(object):
    def __init__(self, df):
        self.df = df

    def countna(self):
        columns = self.df.get_column_names()
        return sum([self.df[feature].countna() for feature in columns])

    @staticmethod
    def to_python(value):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        return value

    def to_pandas(self, chunk_size=None):
        if chunk_size is None:
            return pd.DataFrame.from_dict({key: list(value) for key, value in self.df.to_items()})

        def iterator():
            for _, _, chunk in self.df.to_items(chunk_size=chunk_size):
                yield pd.DataFrame.from_dict({key: list(value) for key, value in chunk})

        return iterator()

    def to_records(self, item=None, chunk_size=None):
        if isinstance(item, int):
            return {key: value[0] for key, value in self.df[item:item + 1].to_dict(array_type='python').items()}
            # return {key: self.to_python(value) for key, value in zip(self.df.get_column_names(), self.df[item])}
        if item is not None:
            raise RuntimeError(f"item can be None or an int - {type(item)} provided")
        else:
            if chunk_size is None:
                records = self.df.to_dict(array_type='python')
                keys = list(records.keys())
                return [{key:value for key,value in zip(keys,values)} for values in zip(*records.values())]

            def iterator():
                for _,_ ,chunk in self.df.to_dict(chunk_size=chunk_size, array_type='python'):
                    keys = list(chunk.keys())
                    yield [{key:value for key,value in zip(keys,values)} for values in zip(*chunk.values())]

            return iterator()

    def json_dumps(self, item=None):
        return json.dumps(self.to_records(item=item))

    def json_dump(self, fp, item=None):
        return json.dump(self.to_records(item=item), fp=fp)

