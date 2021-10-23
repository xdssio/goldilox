from time import time

import numpy as np
import pandas as pd
import traitlets
from vaex.ml.state import HasState

from goldilox import Pipeline

DEFAULT_OUTPUT_COLUMN = 'prediction'


class Pipeline(HasState, Pipeline):
    pipeline_type = 'sklearn'
    current_time = int(time())
    created = traitlets.Int(default_value=current_time, allow_none=False, help='Created time')
    updated = traitlets.Int(default_value=current_time, allow_none=False, help='Updated time')
    warnings = traitlets.Bool(default_value=True, help='Raise warnings')
    example = traitlets.Dict(default_value=None, allow_none=True, help='An example of the transformed dataset')
    pipeline = traitlets.Any(allow_none=False, help='A sklearn pipeline')
    features = traitlets.List(allow_none=None, help="A list of features")
    target = traitlets.Unicode(allow_none=True, help="A column to learn on fit")
    output_column = traitlets.Unicode(default_value=DEFAULT_OUTPUT_COLUMN, help="A column to learn on fit")

    @property
    def raw(self):
        return self.example

    @classmethod
    def from_sklearn(cls, pipeline, X=None, y=None, example=None, columns=None, output_column=DEFAULT_OUTPUT_COLUMN):
        features = columns
        target = None
        if isinstance(X, list):
            features = X
        elif isinstance(X, pd.DataFrame):
            features = list(X.columns)
            if example is None:
                example = X.iloc[0].to_dict()
        elif isinstance(X, np.ndarray):
            if columns is None:
                raise RuntimeError("When 'X' is a numpy array, you must provide 'columns'")
            elif example is None:
                example = {column: value for column, value in zip(columns, X[0])}
        if isinstance(y, pd.Series):
            target = y.name
        elif isinstance(y, str):
            target = y
        return Pipeline(pipeline=pipeline, features=features, target=target, example=example,
                        output_column=output_column)

    def infer(self, df):
        if isinstance(df, pd.DataFrame):
            return df.copy()
        if isinstance(df, np.ndarray):
            return pd.DataFrame(df, columns=self.features)
        if isinstance(df, dict):
            df = [df.copy()]
        if isinstance(df, list):
            return pd.DataFrame(df)

        raise RuntimeError(f"could not infer type:{type(df)}")


    def fit(self, df, **kwargs):
        self.pipeline = self.pipeline.fit(df[self.features], df[self.target])
        return self

    def inference(self, df, columns=None, **kwargs):
        copy = self.infer(df)
        features = self.features or df.columns
        if hasattr(self.pipeline, 'predict'):
            copy[self.output_column] = self.pipeline.predict(copy[features])
        else:
            copy = self.pipeline.transform(copy[features])
        return copy

    def preprocess(self, df):
        pass
