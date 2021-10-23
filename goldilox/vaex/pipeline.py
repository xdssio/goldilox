import inspect
import json
import logging
import os
import time
from collections import OrderedDict
from copy import deepcopy, copy as _copy
from glob import glob
from numbers import Number

import numpy as np
import pandas as pd
import pyarrow as pa
import s3fs
import traitlets
import vaex
from vaex.column import Column
from vaex.ml.state import HasState, serialize_pickle

from goldilox.pipeline import Pipeline
from goldilox.vaex.config import *

logging.basicConfig()
logger = logging.getLogger(__name__)

EXAMPLE = 'example'
PIPELINE_FIT = '_PIPELINE_FIT'
FUNCTIONS = 'functions'


def _is_s3_url(path):
    if hasattr(path, 'dirname'):
        path = path.dirname
    return path.startswith('s3://')


class Pipeline(HasState, Pipeline):
    pipeline_type = 'vaex'
    current_time = int(time.time())
    created = traitlets.Int(default_value=current_time, allow_none=False, help='Created time')
    updated = traitlets.Int(default_value=current_time, allow_none=False, help='Updated time')
    example = traitlets.Any(default_value=None, allow_none=True, help='An example of the transformed dataset').tag(
        **serialize_pickle)
    raw = traitlets.Any(default_value=None, allow_none=True, help='An example of the raw dataset').tag(
        **serialize_pickle)
    _original_dtypes = traitlets.Dict(default_value={},
                                      help='original columns which were not virtual expressions')
    state = traitlets.Dict(default_value=None, allow_none=True, help='The state to apply on inference')
    warnings = traitlets.Bool(default_value=True, help='Raise warnings')

    @classmethod
    def _get_original_columns(cls, df):
        return list(df.dataset._columns.keys())

    @classmethod
    def _get_original_dtypes(cls, df):
        columns = df.head(1).dataset._columns

        def data_type(data):
            if isinstance(data, np.ndarray):
                data_type = data.dtype
            elif isinstance(data, Column):
                data = data.to_arrow()
                data_type = data.type
            else:
                # when we eval constants, let arrow find it out

                if isinstance(data, Number):
                    data_type = pa.array([data]).type
                else:
                    data_type = data.type  # assuming arrow
            return data_type

        return {column: data_type(data) for column, data in columns.items()}

    def state_set(self, state):
        HasState.state_set(self, state)
        self.updated = int(time.time())
        return self

    def not_implemented(self):
        return None

    @property
    def virtual_columns(self):
        return self.state.get('virtual_columns')

    @property
    def functions(self):
        return self.state.get('functions')

    @classmethod
    def verify_vaex_dataset(cls, df):
        if not cls.is_vaex_dataset(df):
            raise ValueError('ds should be a vaex.dataset.DatasetArrays or vaex.hdf5.dataset.Hdf5MemoryMapped')
        return df.copy()

    @classmethod
    def from_dict(cls, state):
        if 'state' in state:
            ret = Pipeline()
            ret.state_set(state)
            return ret
        return Pipeline(state=state, example=None, fit_func=None)

    def sample_first(self, df):
        if len(df) == 0:
            raise RuntimeError("cannot sample from empty dataframe")
        try:
            sample = df[0:1]
            self._original_dtypes = self._get_original_dtypes(df)
            self.raw = {key: values[0] for key, values in sample.dataset._columns.items()}
            self.example = sample.to_records()[0]
            return True
        except Exception as e:
            logger.error(f"could not sample first: {e}")
        return False

    @classmethod
    def from_dataframe(cls, df, fit=None, warnings=True):
        copy = Pipeline.verify_vaex_dataset(df)
        if fit is not None:
            copy.add_function(PIPELINE_FIT, fit)
        sample = copy[0:1]
        raw = {key: values[0] for key, values in sample.dataset._columns.items()}
        example = sample.to_records()[0]
        pipeline = Pipeline(state=copy.state_get(),
                            warnings=warnings,
                            _original_dtypes=Pipeline._get_original_dtypes(df),
                            raw=raw,
                            example=example)

        return pipeline

    def copy(self):
        pipeline = Pipeline(state={})
        pipeline.state_set(deepcopy(self.state_get()))
        pipeline.updated = int(time.time())
        return pipeline

    @classmethod
    def from_file(cls, path):
        if _is_s3_url(path):
            fs = s3fs.S3FileSystem(profile=AWS_PROFILE)
            with fs.open(path, 'r') as f:
                state = cls.json_load(f.read())
        else:
            with open(path, 'r') as f:
                state = cls.json_load(f.read())
        ret = Pipeline.from_dict(state)
        # ret.reload_fit_func()
        return ret

    def json_dumps(self):
        from vaex.json import VaexJsonEncoder
        return json.dumps(_copy(self.state_get()), indent=2, cls=VaexJsonEncoder)

    @classmethod
    def json_load(cls, state):
        from vaex.json import VaexJsonDecoder
        return json.loads(state, cls=VaexJsonDecoder)

    def save(self, path):
        if self.state is None:
            raise RuntimeError("Pipeline has no state to save")

        state_to_write = self.json_dumps()
        if _is_s3_url(path):
            fs = s3fs.S3FileSystem(profile=AWS_PROFILE)
            with fs.open(path, 'w') as f:
                f.write(state_to_write)
        else:
            try:
                os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
            except AttributeError as e:
                pass
            with open(path, 'w') as outfile:
                outfile.write(state_to_write)
        # self.reload_fit_func()
        return path

    @classmethod
    def is_vaex_dataset(cls, ds):
        return isinstance(ds, vaex.dataframe.DataFrame)

    @classmethod
    def load_state(cls, state):
        instance = Pipeline()
        instance.state_set(state)
        return instance

    def __getitem__(self, index):
        virtual_columns = list(OrderedDict(self.virtual_columns).items())
        return virtual_columns[index]

    # TODO replace add_memory column - currently not working
    def add_virtual_column(self, df, name, value, first_column):
        df[name] = df.func.where(df[first_column] == value, value, value)
        return df

    def add_memmory_column(self, df, name, value, length=None):
        if length is None:
            length = len(df)
        df[name] = np.array([value] * length)
        return df

    def get_columns_to_add(self, df, columns=None):
        columns_to_add = set([])
        if any([column not in df for column in self.values]):
            example = self.transform_state(self.infer(self.example.copy()))
            if columns is None:
                columns = list(self.values.keys())

            queue = _copy(columns)
            while len(queue) > 0:
                column = queue.pop()
                variables = example[column].variables(include_virtual=True, expand_virtual=True)
                if len(variables) == 0:
                    columns_to_add.add(column)
                elif len(variables) == 1 and variables.pop() in self._original_columns:
                    columns_to_add.add(column)
                else:
                    queue.extend(variables)
        return columns_to_add

    def na_column(self, length, dtype):
        return pa.array([None] * length)

    def preprocess_transform(self, df, columns, fillna=True):
        copy = self.infer(df)
        length = len(copy)
        values = self._original_dtypes
        renamed = {x[1]: x[0] for x in self.state['renamed_columns']}
        for key, dtype in values.items():
            if key in renamed:
                key = renamed.get(key)
            if key not in copy:
                copy[key] = self.na_column(length, dtype)
        return copy

    def transform_state(self, df, keep_columns=None, state=None, set_filter=False):
        copy = df.copy()
        state = state or self.state
        if state is not None:
            if keep_columns is True:
                keep_columns = list(set(copy.get_column_names()).difference(
                    [k for k in state['column_names'] if not k.startswith('__')]))
            if keep_columns is False or (keep_columns is not None and len(keep_columns) == 0):
                keep_columns = None
            copy.state_set(state, keep_columns=keep_columns, set_filter=set_filter)
        return copy

    def transform(self, df, keep_columns=False, state=None, set_filter=True, fillna=True):
        copy = self.preprocess_transform(df, fillna=fillna)
        copy = self.transform_state(copy, keep_columns=keep_columns, state=state, set_filter=set_filter)
        return copy

    def fill_columns(self, df, columns=None, length=None):
        if columns is None:
            return df
        if length is None:
            length = len(df)
        for column in columns:
            if column not in df:
                value = self.values.get(column)
                if value is not None:
                    self.add_memmory_column(df, column, value, length)
        return df

    def inference(self, df, columns=None, set_filter=False, keep_columns=None, fillna=True, clean=False):
        if clean:
            copy = self.infer(df)
            copy.state_set(self.state, set_filter=set_filter)
            return copy
        if isinstance(columns, str):
            columns = [columns]
        copy = self.preprocess_transform(df, columns=columns, fillna=fillna)
        if columns is None and keep_columns is None:
            keep_columns = True
        ret = self.transform_state(copy, set_filter=set_filter, keep_columns=keep_columns)
        # ret = self.fill_columns(copy, columns=columns)
        if columns is not None:
            ret = ret[columns]
        return ret

    def evaluate(self):
        raise NotImplementedError('evaluate not implemented')

    def partial_fit(self, start_index=None, end_index=None):
        raise ValueError('partial_fit implemented')

    @classmethod
    def infer(cls, data, **kwargs):
        if isinstance(data, vaex.dataframe.DataFrame):
            return data.copy()
        elif isinstance(data, pd.DataFrame):
            return vaex.from_pandas(data)
        elif isinstance(data, str):
            if os.path.isfile(data):
                logger.info(f"reading file from {data}")
                return vaex.open(data, **kwargs)
            elif os.path.isdir(data):
                logger.info(f"reading files from {data}")
                files = []
                for header in VALID_VAEX_HEADERS:
                    files.extend(glob(f"{data}/{header}"))
                logger.info(f"open files: {files}")
                bad_files = set([])
                for file in files:
                    try:
                        if file.endswith('.csv'):
                            temp = vaex.open(file, nrows=1)
                        else:
                            temp = vaex.open(file)
                        temp.head()
                    except:
                        bad_files.add(file)
                files = [file for file in files if file not in bad_files]
                return vaex.open_many(files, **kwargs)
            data = json.loads(data)
        elif isinstance(data, bytes):
            data = json.loads(data)
        if isinstance(data, np.ndarray):
            columns = kwargs.get('names')
            if columns is None:
                raise RuntimeError("can't infer numpy array without 'names' as a list of columns")
            if len(columns) == data.shape[1]:
                data = data.T
            return vaex.from_dict({key: value for key, value in zip(columns, data)})
        elif isinstance(data, list):
            # try records
            return vaex.from_pandas(pd.DataFrame(data))
        elif isinstance(data, dict):
            sizes = [0 if (not hasattr(value, '__len__') or isinstance(value, str)) else len(value) for value in
                     data.values()]
            if len(sizes) == 0 or min(sizes) == 0:
                data = data.copy()
                for key, value in data.items():
                    if isinstance(value, list) or isinstance(value, np.ndarray):
                        data[key] = np.array([value])
                    else:
                        data[key] = [value]
            random_value = data[list(data.keys())[0]]
            if isinstance(random_value, dict):
                return vaex.from_pandas(pd.DataFrame(data))
            return vaex.from_arrays(**data)

        raise RuntimeError("Could not infer a vaex type")

    def set_variable(self, key, value):
        self.state[VARIABLES][key] = value
        return value

    def get_variable(self, key, default=None):
        return self.state[VARIABLES].get(key, default)

    @property
    def variables(self):
        return self.state[VARIABLES]

    def get_variable(self, variable):
        if self.state is None:
            logger.debug('state is None')
            return None
        return self.state[VARIABLES].get(variable)

    def get_columns_names(self, virtual=True, strings=True, hidden=False, regex=None):
        return self.inference(self.example).get_column_names(virtual=virtual, strings=strings, hidden=hidden,
                                                             regex=regex)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def is_valid_fit(self, f):
        return f is not None and callable(f) and len(inspect.getfullargspec(f)[0]) > 0

    def set_fit(self, f=None):
        if self.is_valid_fit(f):
            self.fit_func = f

    def get_function(self, name):
        from vaex.serialize import from_dict
        return from_dict(self.state.get(FUNCTIONS, {}).get(name), trusted=True).f

    def fit(self, df):
        copy = df.copy()
        self.verify_vaex_dataset(copy)
        fit_func = self.get_function(PIPELINE_FIT)
        if fit_func is None:
            raise RuntimeError("'fit()' was not set for this pipeline")
        trained = fit_func(copy)
        self.sample_first(trained)
        if Pipeline.is_vaex_dataset(trained):
            trained.add_function(PIPELINE_FIT, fit_func)
            self.state = trained.state_get()
        else:
            if isinstance(trained, dict):
                self.state = trained
            else:
                raise ValueError("'fit_func' should return a vaex dataset or a state, got {}".format(type(trained)))
        self.updated = int(time.time())

    def get_function_model(self, name):
        tmp = self.state['functions'][name]
        model = eval(tmp['cls'])()
        model.state_set(state=tmp['state'])
        return model
