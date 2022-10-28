from __future__ import annotations

import inspect
import json
import logging
from collections import OrderedDict
from copy import deepcopy, copy as _copy
from glob import glob
from numbers import Number
from time import time
from typing import List, Union, Callable

import cloudpickle
import numpy as np
import pandas as pd
import pyarrow as pa
import traitlets
import vaex
from vaex.ml.state import HasState, serialize_pickle

import goldilox
from goldilox.config import *
from goldilox.pipeline import Pipeline
from goldilox.utils import process_variables, read_vaex_data

logging.basicConfig()
logger = logging.getLogger(__name__)

VAEX = 'vaex'


class VaexPipeline(HasState, Pipeline):
    pipeline_type = traitlets.Unicode(default_value=VAEX)
    _original_columns = traitlets.List(
        default_value=[], help="original columns which were not virtual expressions"
    )
    state = traitlets.Dict(
        default_value=None, allow_none=True, help="The state to apply on inference"
    )
    predict_column = traitlets.Unicode(
        default_value=None, allow_none=True, help="The column to return as numpy array in predict"
    )
    meta = traitlets.Any(
        default_value=goldilox.Meta(VAEX), allow_none=False, help="The environment class"
    ).tag(**serialize_pickle)

    VARIABLES = "variables"
    FIT = "_PIPELINE_FIT"
    PREDICT = "_PIPELINE_PREDICT"
    PREDICT_PROBA = "_PIPELINE_PREDICT_PROBA"
    READ = "_PIPELINE_READ"
    FUNCTIONS = "functions"

    @classmethod
    def _get_original_columns(cls, df: vaex.dataframe.DataFrame) -> List[str]:
        return list(df.dataset._columns.keys())

    @classmethod
    def _data_type(cls, data):
        if isinstance(data, np.ndarray):
            data_type = data.dtype
        elif isinstance(data, vaex.dataset.Column):
            data = pa.array(data.tolist())
            data_type = data.type.to_pandas_dtype()
        else:
            # when we eval constants, let arrow find it out
            if isinstance(data, Number):
                data_type = pa.array([data]).type.to_pandas_dtype()
            else:
                data_type = data.type.to_pandas_dtype()  # assuming arrow
        return data_type

    @classmethod
    def _get_original_dtypes(cls, df: vaex.dataframe.DataFrame) -> dict:
        return {
            column: cls._data_type(data)
            for column, data in df.head(1).dataset._columns.items()
        }

    def state_set(self, state: dict) -> VaexPipeline:
        HasState.state_set(self, state)
        self.updated = int(time())
        return self

    def not_implemented(self):
        return None

    @property
    def virtual_columns(self) -> List[str]:
        return self.state.get("virtual_columns")

    def _example(self):
        return self.inference(self.raw).to_records(0)

    @property
    def example(self) -> dict:
        return self.meta.example

    @property
    def functions(self):
        return self.state.get("functions")

    @classmethod
    def verify_vaex_dataset(cls, df) -> vaex.dataframe.DataFrame:
        if not cls.is_vaex_dataset(df):
            raise ValueError(
                "ds should be a vaex.dataset.DatasetArrays or vaex.hdf5.dataset.Hdf5MemoryMapped"
            )
        return df.copy()

    @classmethod
    def from_dict(cls, state: dict) -> VaexPipeline:
        if "state" in state:
            ret = VaexPipeline()
            ret.state_set(state)
            return ret
        return VaexPipeline(state=state, example=None, fit_func=None)

    def sample_first(self, df: vaex.dataframe.DataFrame) -> bool:
        if len(df) == 0:
            raise RuntimeError("cannot sample from empty dataframe")
        try:
            sample = df[0:1]
            self._original_columns = self._get_original_columns(df)
            self.raw = {
                key: values.tolist()[0] for key, values in sample.dataset._columns.items()
            }
            return True
        except Exception as e:
            logger.error(f"could not sample first: {e}")
        return False

    @classmethod
    def _tolist(cls, value) -> List:
        if hasattr(value, "ar"):
            return value.ar.tolist()
        elif hasattr(value, "to_numpy"):
            try:
                value = value.to_numpy()
            except pa.lib.ArrowInvalid as e:
                logger.warning(str(e))
        return value.tolist()

    @classmethod
    def from_dataframe(cls, df: vaex.dataframe.DataFrame, fit=None, predict_column: str = None,
                       variables: dict = None) -> VaexPipeline:
        """
       Get a Pipeline out of a vaex.dataframe.DataFrame, and validate serilization and missing values.
       @param df: vaex.dataframe.DataFrame
       @param fit: method: A method which accepts a vaex dataframe and returns a vaex dataframe which run on pipeline.fit(df).
       @param variables: dict [optional]: Any variables we want to associate with the current pipeline.
              On top of the variables provided, the dataframe variables are added.
       @param predict_column: str [optional]: The predict column to use in predict case
       @return: VaexPipeline
       """
        copy = VaexPipeline.verify_vaex_dataset(df)
        if fit is not None:
            copy.add_function(VaexPipeline.FIT, fit)
        sample = copy[0:1]
        state = copy.state_get()
        renamed = {x[1]: x[0] for x in state["renamed_columns"]}
        raw = {
            renamed.get(key, key): cls._tolist(value)[0]
            for key, value in sample.dataset._columns.items()
        }
        original_columns = VaexPipeline._get_original_columns(sample)
        variables = {**(state.get(VaexPipeline.VARIABLES) or {}), **(variables or {})}

        predict_column = predict_column or df.get_column_names()[-1]
        pipeline = VaexPipeline(
            state=state,
            _original_columns=original_columns,
            meta=goldilox.Meta(VAEX, raw=raw, variables=process_variables(variables)),
            predict_column=predict_column
        )
        pipeline.meta.example = pipeline._example()
        return pipeline

    def add_function(self, name: str, func: Callable) -> VaexPipeline:
        func_expression = vaex.expression.Function(self, name, func)
        ret = name not in self.state['functions']
        self.state['functions'][name] = vaex.serialize.to_dict(func_expression.f)
        return ret

    def copy(self) -> VaexPipeline:
        pipeline = VaexPipeline(state={})
        pipeline.state_set(deepcopy(self.state_get()))
        pipeline.updated = int(time())
        return pipeline

    @classmethod
    def from_file(cls, path: str) -> VaexPipeline:
        return Pipeline.from_file(path)

    def _dumps(self) -> bytes:
        return cloudpickle.dumps(_copy(self.state_get()))

    @classmethod
    def json_load(cls, state: dict):
        from vaex.json import VaexJsonDecoder

        return json.loads(state[CONSTANTS.STATE], cls=VaexJsonDecoder)

    @classmethod
    def is_vaex_dataset(cls, ds) -> bool:
        return isinstance(ds, vaex.dataframe.DataFrame)

    @classmethod
    def load_state(cls, state) -> VaexPipeline:
        instance = VaexPipeline()
        if not isinstance(state, dict):
            state = VaexPipeline.json_load(state)
        instance.state_set(state)
        return instance

    def __getitem__(self, index):
        virtual_columns = list(OrderedDict(self.virtual_columns).items())
        return virtual_columns[index]

    # TODO replace add_memory column - currently not working
    def add_virtual_column(self, df, name, value, first_column):
        df[name] = df.func.where(df[first_column] == value, value, value)
        return df

    def add_memory_column(self, df: vaex.dataframe.DataFrame, name: str, value, length: int = None):
        if length is None:
            length = len(df)
        df[name] = np.array([value] * length)
        return df

    def na_column(self, length: int) -> pa.Array:
        return pa.array([None] * length)

    def preprocess_transform(
            self,
            df: Union[vaex.dataframe.DataFrame, pd.DataFrame],
    ) -> vaex.dataframe.DataFrame:
        copy = self.infer(df)
        length = len(copy)

        renamed = {x[1]: x[0] for x in self.state["renamed_columns"]}
        for key in self._original_columns:
            if key in renamed:
                key = renamed.get(key)
            if key not in copy:
                copy[key] = self.na_column(length)
        return copy

    def transform_state(self, df, passthrough: bool = None, state: dict = None,
                        set_filter: bool = False) -> vaex.dataframe.DataFrame:
        copy = df.copy()
        state = state or self.state
        if state is not None:
            if passthrough is True:
                passthrough = list(
                    set(copy.get_column_names()).difference(
                        [k for k in state["column_names"] if not k.startswith("__")]
                    )
                )
            if passthrough is False or (
                    passthrough is not None and len(passthrough) == 0
            ):
                passthrough = None
            copy.state_set(state, keep_columns=passthrough, set_filter=set_filter)
        return copy

    def transform(self, df: Union[vaex.dataframe.DataFrame, pd.DataFrame], passthrough: bool = False,
                  state: dict = None,
                  set_filter: bool = True) -> vaex.dataframe.DataFrame:
        copy = self.preprocess_transform(df)
        copy = self.transform_state(
            copy, passthrough=passthrough, state=state, set_filter=set_filter
        )
        return copy

    def fill_columns(self, df: vaex.dataframe.DataFrame, columns: List[str] = None, length: int = None):
        if columns is None:
            return df
        if length is None:
            length = len(df)
        for column in columns:
            if column not in df:
                value = self.values.get(column)
                if value is not None:
                    self.add_memory_column(df, column, value, length)
        return df

    def inference(
            self, df, columns: List[str] = None, passthrough: bool = True, set_filter: bool = False,
    ) -> vaex.dataframe.DataFrame:
        if isinstance(columns, str):
            columns = [columns]
        copy = self.preprocess_transform(df)
        ret = self.transform_state(
            copy, set_filter=set_filter, passthrough=passthrough
        )
        # ret = self.fill_columns(copy, columns=columns)
        if columns is not None:
            ret = ret[columns]
        return ret

    def evaluate(self):
        raise NotImplementedError("evaluate not implemented")

    def partial_fit(self, start_index=None, end_index=None):
        raise ValueError("partial_fit implemented")

    @property
    def features(self):
        return [column for column in self._original_columns if column in self.raw]

    @property
    def target(self):
        return self.predict_column

    def infer(self, data, **kwargs):
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
                        if file.endswith(".csv"):
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
            columns = kwargs.get("names", self.features)
            if len(columns) == data.shape[1]:
                data = data.T
            return vaex.from_dict({key: value for key, value in zip(columns, data)})
        elif isinstance(data, list):
            # try records
            columns = kwargs.get("names", self.features)
            data = pd.DataFrame(data)
            if len(columns) == data.shape[1]:
                data.columns = columns
            return vaex.from_pandas(data)
        elif isinstance(data, dict):
            sizes = [
                0
                if (not hasattr(value, "__len__") or isinstance(value, str))
                else len(value)
                for value in data.values()
            ]
            if len(sizes) == 0 or min(sizes) == 0:
                data = data.copy()
                for key, value in data.items():
                    if isinstance(value, (list, np.ndarray)):
                        data[key] = np.array([value])
                    else:
                        data[key] = [value]
            random_value = data[list(data.keys())[0]]
            if isinstance(random_value, dict):
                return vaex.from_pandas(pd.DataFrame(data))
            return vaex.from_arrays(**data)

        raise RuntimeError("Could not infer a vaex type")
        raise RuntimeError("Could not infer a vaex type")

    def set_variable(self, key, value):
        self.variables[key] = value
        return value

    def get_variable(self, key, default=None):
        return self.variables.get(key, default)

    def get_columns_names(self, virtual: bool = True, strings: bool = True, hidden: bool = False, regex: str = None):
        return self.inference(self.raw).get_column_names(
            virtual=virtual, strings=strings, hidden=hidden, regex=regex
        )

    def fit_transform(self, data) -> vaex.dataframe.DataFrame:
        self.fit(data)
        return self.transform(data)

    def is_valid_fit(self, f) -> bool:
        return f is not None and callable(f) and len(inspect.getfullargspec(f)[0]) > 0

    def set_fit(self, f=None) -> None:
        if self.is_valid_fit(f):
            self.fit_func = f

    def get_function(self, name: str, default: Callable = None):
        from vaex.serialize import from_dict
        func = self.state.get(VaexPipeline.FUNCTIONS, {}).get(name)
        if func:
            return from_dict(func, trusted=True).f
        return default

    def fit(self, df: Union[vaex.dataframe.DataFrame, pd.DataFrame, str]) -> VaexPipeline:
        if isinstance(df, str):
            logger.info(f"reading file from {df}")
            df = self.get_function(VaexPipeline.READ, read_vaex_data)(df)
        copy = df.copy()
        self.verify_vaex_dataset(copy)
        self.sample_first(copy)
        fit_func = self.get_function(VaexPipeline.FIT)
        if fit_func is None:
            raise RuntimeError("'fit()' was not set for this pipeline")
        self.set_raw(copy.to_records(0))
        trained = fit_func(copy)
        if VaexPipeline.is_vaex_dataset(trained):
            trained.add_function(VaexPipeline.FIT, fit_func)
            self.state = trained.state_get()
        else:
            if isinstance(trained, dict):
                self.state = trained
            else:
                raise ValueError(
                    f"'fit_func' should return a vaex dataset or a state, got {type(trained)} instead"
                )
        self.variables.update(self.state.get(CONSTANTS.VARIABLES, {}))
        self.updated = int(time())
        self.meta.example = self._example()
        return self

    @staticmethod
    def _predict(pipeline, df: Union[vaex.dataframe.DataFrame, pd.DataFrame]):
        ret = pipeline.inference(df, columns=pipeline.predict_column)[pipeline.predict_column]
        if hasattr(ret, 'to_numpy'):
            return ret.to_numpy()
        return ret.tolist()

    def predict(self, df: Union[vaex.dataframe.DataFrame, pd.DataFrame]):
        predict_function = self.get_function(VaexPipeline.PREDICT)
        if predict_function:
            return predict_function(self, df)
        return self._predict(self, df)

    def predict_proba(self, df: Union[vaex.dataframe.DataFrame, pd.DataFrame]):
        predict_proba_function = self.get_function(VaexPipeline.PREDICT_PROBA)
        if predict_proba_function:
            return predict_proba_function(self, df)
        raise NotImplementedError(
            "'predict_proba' is not implemented - please provide a function and add it with 'pipeline.add_function(pipeline.PIPELINE_PREDICT_PROBA, func)'")

    def get_function_model(self, name: str):
        tmp = self.state["functions"][name]
        model = eval(tmp["cls"])()
        model.state_set(state=tmp["state"])
        return model
