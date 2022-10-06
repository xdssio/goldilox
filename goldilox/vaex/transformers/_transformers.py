import logging
import re
from contextlib import suppress
from uuid import uuid4

import vaex
from traitlets import traitlets
from vaex.ml.state import HasState

logger = logging.getLogger()


class TransformerBase(HasState):
    output_columns = traitlets.List(traitlets.Unicode(), allow_none=True, help='List of the columns which were set')
    features = traitlets.List(traitlets.Unicode(), allow_none=True, help='List of the features as input')

    def fit(self, df, **kwargs):
        super().fit(df)
        return self

    def transform(self, df, **kwargs):
        return super().transform(df)

    def fit_transform(self, df, **kwargs):
        self.fit(df, **kwargs)
        return self.transform(df, **kwargs)

    @classmethod
    def load_state(cls, state):
        instance = cls()
        instance.state_set(state)
        return instance

    def preprocess_transform(self, df):
        copy = df.copy()
        return copy


@vaex.serialize.register
class Imputer(TransformerBase):
    strategy = traitlets.Dict(help='Strategy for handling missing values')
    prefix = traitlets.Unicode(default_value='', help="")
    values = traitlets.Dict(help='The value to fill in each feature')
    warnings = traitlets.Bool(default_value=True, help="Raise warnings when can not fillna a column")
    MEAN = 'MEAN'
    MODE = 'MODE'
    NEW_VALUE = 'NEW_VALUE'
    VALUE = 'VALUE'
    COMMON = 'COMMON'

    @staticmethod
    def mean(ar):
        return ar.mean()

    @staticmethod
    def mode(ar):
        return ar.df.percentile_approx(ar)

    @staticmethod
    def common(ar):
        return ar.value_counts(dropna=True).index[0]

    @staticmethod
    def new_value(ar):
        if ar.dtype.is_string:
            return f"NA_{uuid4()}"
        elif ar.dtype.is_numeric:
            return int(ar.max()) + 1
        raise ValueError(f"Can not create a new value for {ar.expression} of {ar.dtype} dtype")

    def get_value(self, ar):
        # TODO run in one go
        dtype = ar.dtype
        name = ar.expression
        feature_strategy = None

        try:
            # TODO fix whe NotImplementedError: large_string is fixed
            dtype_name = dtype.name
        except:
            dtype_name = None
        try:
            shape = ar.shape
        except:
            shape = (1,)
        if name in self.strategy:  # by name
            feature_strategy = self.strategy.get(name)
        elif 1 < len(shape):  # nd arrays
            feature_strategy = None
        # by dtype
        elif dtype_name is not None and dtype_name in self.strategy:
            feature_strategy = self.strategy.get(dtype_name.name)
        elif dtype.is_integer and int in self.strategy:
            feature_strategy = self.strategy.get(int)
        elif dtype.is_float and float in self.strategy:
            feature_strategy = self.strategy.get(float)
        elif dtype == bool and bool in self.strategy:
            feature_strategy = self.strategy.get(bool)
        elif dtype == str and str in self.strategy:
            feature_strategy = self.strategy.get(str)
        if feature_strategy is None and len(shape) == 1:
            feature_strategy = self.get_default_value(dtype)
        if callable(feature_strategy):
            feature_strategy = feature_strategy(ar)

        value = feature_strategy
        if dtype.is_string:
            if not isinstance(value, str):
                value = str(value)
            if feature_strategy == self.COMMON:
                value = self.common(ar)
            elif feature_strategy == self.NEW_VALUE:
                value = self.new_value(ar)
        elif dtype.is_numeric and len(shape) == 1:
            if isinstance(feature_strategy, str):
                if feature_strategy == self.MEAN:
                    value = float(ar.mean())
                elif feature_strategy == self.MODE:
                    value = float(self.mode(ar))
                elif feature_strategy == self.COMMON:
                    value = float(self.common(ar))
                elif feature_strategy == self.NEW_VALUE:
                    value = float(self.new_value(ar))

            if not isinstance(value, (int, float)):
                raise RuntimeError(f"value {value} cannot be used for {name} of type {dtype}")

        if value is None and self.warnings:
            logger.warning(f"value {value} was 'None' for {name} of type {dtype}")
            # raise RuntimeError(f"value {value} was 'None' for {name} of type {dtype}")
        return value

    def preprocess_fit(self, df, **kwargs):
        self.set_features(df)
        return df

    def set_features(self, df):
        if self.features is None or len(self.features) == 0:
            columns = df.get_column_names()
            self.features = [column for column in columns if
                             df[column].dtype.is_primitive or df[column].dtype.is_string]
        return self.features

    def fit(self, df, **kwargs):
        copy = df.copy()
        self.preprocess_fit(df)
        self.output_columns = []
        for feature in self.features:
            self.values[feature] = self.get_value(copy[feature])
            if self.values.get(feature) is not None and feature in self.features:
                name = self.prefix + feature
                self.output_columns.append(name)
        return self

    def add_column(self, df, name, value, first_column):
        df[name] = df.func.where(df[first_column] == value, value, value)
        return df

    def preprocess_transform(self, df):
        copy = df.copy()
        first_column = copy.get_column_names()[0]
        for column, value in self.values.items():
            if column not in copy:
                copy = self.add_column(copy, str(column), value, first_column)
        return copy

    def transform(self, df, **kwargs):
        copy = self.preprocess_transform(df)
        for column, value in self.values.items():
            name = self.prefix + column
            if value is None:
                if self.warnings:
                    logger.warning(f" could not fillna for {name} ")
                continue
            copy[name] = copy[column].fillna(value)
        return copy

    def get_default_value(self, dtype):
        if dtype.is_string:
            return ''
        if dtype.is_numeric:
            return 0
        with suppress():
            if dtype.name == 'bool':
                return False
        if dtype == object:
            return {}
        if self.warnings:
            logger.warning(f"dtype {dtype} has no default value")
        return None

        # raise ValueError(f"dtype {dtype} has no default value")

    def __repr__(self):
        return f"Imputer({self.strategy})"

    @staticmethod
    def _to_type(s):
        return eval(re.sub('[^0-9a-zA-Z]+', '', s.replace('class', '')))

    @staticmethod
    def _is_type(s):
        return isinstance(s, str) and s.startswith('<class')

    def state_get(self):
        state = TransformerBase.state_get(self)
        for key, value in state['strategy'].items():
            if isinstance(key, type):
                state['strategy'][str(key)] = state['strategy'].pop(key)
        return state

    def state_set(self, state, trusted=True):
        if 'strategy' in state:
            for key, value in state['strategy'].items():
                if self._is_type(key):
                    state['strategy'][self._to_type(key)] = state['strategy'].pop(key)
        TransformerBase.state_set(self, state)
