from __future__ import annotations

import logging
from contextlib import suppress
from typing import List, Any

import cloudpickle
import numpy as np
import pandas as pd
import traitlets
from sklearn.base import TransformerMixin

import goldilox
from goldilox.utils import read_sklearn_data

DEFAULT_OUTPUT_COLUMN = "prediction"
logger = logging.getLogger()
SKLEARN = 'sklearn'


class SklearnPipeline(traitlets.HasTraits, goldilox.Pipeline, TransformerMixin):
    pipeline_type = traitlets.Unicode(default_value=SKLEARN)
    pipeline = traitlets.Any(allow_none=False, help="A sklearn pipeline")
    features = traitlets.List(allow_none=True, help="A list of features")
    target = traitlets.Unicode(allow_none=True, help="A column to learn on fit")
    output_columns = traitlets.List(allow_none=True, help="The output column names")
    fit_params = traitlets.Dict(allow_none=True, default_value={}, help="params to use on fit time")
    meta = traitlets.Any(default_value=goldilox.Meta(SKLEARN), allow_none=False, help="The meta class")

    @classmethod
    def from_sklearn(
            cls,
            pipeline,
            raw: dict = None,
            features: List[str] = None,
            target: str = None,
            output_columns: List[str] = None,
            variables: dict = None,
            fit_params: dict = None,
    ) -> SklearnPipeline:
        """
        :param sklearn.preprocessing.pipeline.Pipeline pipeline: The sklearn pipeline
        :param raw: dict [optional]: An example of data which will be queried in production (only the features)
                - If X is provided, would be the first row.
        :param features: list [optional]: A list of columns - if X is provided, will take it's columns
        :param target: str [optional]: The name of the target column - Used for retraining
        :param output_columns: List[str] [optional]: For sklearn estimator which predict a numpy.ndarray, name the output columns.
        :param variables: dict [optional]: Variables to associate with the pipeline - fit_params automatically are added
        :return: SkleranPipeline object
        """
        if isinstance(features, pd.core.indexes.base.Index):
            features = list(features)
        elif isinstance(raw, dict) and features is None:
            features = list(raw.keys())
        elif raw is None and isinstance(features, list):
            raw = {key: 0 for key in features}
        if (
                hasattr(pipeline, "__sklearn_is_fitted__")
                and pipeline.__sklearn_is_fitted__()
                and raw is None
                and features is None
        ):
            raise RuntimeError(
                "For a fitted pipeline, please provide either the 'features' or 'sample'"
            )
        if variables is None:
            variables = {}
        if fit_params:
            variables.update(fit_params)
        if hasattr(pipeline, 'predict') and not hasattr(pipeline, 'transform') and (
                output_columns is None or len(output_columns) == 0):
            output_columns = [DEFAULT_OUTPUT_COLUMN]
        pipeline = SklearnPipeline(
            pipeline=pipeline,
            features=features,
            target=target,
            meta=goldilox.Meta(SKLEARN, raw=raw, variables=variables),
            output_columns=output_columns,
            fit_params=fit_params,
            variables=variables
        )
        pipeline.meta.example = pipeline._example()
        return pipeline

    def _example(self):
        if not self._is_sklearn_fitted(self):
            return self.raw
        return self.inference(self.raw).to_dict(orient='records')[0]

    @property
    def example(self) -> dict:
        return self.meta.example

    @property
    def fitted(self) -> bool:
        """Returns True is the pipeline/model is fitted"""
        return self.pipeline is not None and self.pipeline.__sklearn_is_fitted__()

    def infer(self, df: Any) -> pd.DataFrame:
        """Turn many inputs into a dataframes"""
        if isinstance(df, pd.DataFrame):
            return df.copy()
        if isinstance(df, pd.Series):
            return pd.DataFrame({df.name: df})
        if isinstance(df, np.ndarray):
            if self.features and len(self.features) == df.shape[1]:
                return pd.DataFrame(df, columns=self.features)
            elif self.output_columns and len(self.output_columns) == df.shape[1]:
                return pd.DataFrame(df, columns=self.output_columns)
            else:
                return pd.DataFrame(df)
        if isinstance(df, dict):
            df = [df.copy()]
        if isinstance(df, list):
            ret = pd.DataFrame(df)
            if self.features is not None and \
                    isinstance(ret.columns, pd.Index) and \
                    len(self.features) == len(
                ret.columns):
                ret.columns = self.features
            return ret
        with suppress():
            import vaex
            if isinstance(df, vaex.dataframe.DataFrame):
                return df.to_pandas_df()

        raise RuntimeError(f"could not infer type:{type(df)}")

    def _dumps(self) -> bytes:
        return cloudpickle.dumps(self)

    @classmethod
    def loads(cls, state) -> dict:
        if isinstance(state, bytes):
            state = cloudpickle.loads(state)
        return state[goldilox.config.CONSTANTS.STATE]

    @classmethod
    def from_file(cls, path: str) -> SklearnPipeline:
        return goldilox.Pipeline.from_file(path)

    def _to_pandas(self, X, y=None) -> tuple:
        try:  # noqa: FURB107
            import vaex
            if isinstance(X, vaex.dataframe.DataFrame):
                X = X.to_pandas_df()
            elif isinstance(X, pd.Series):
                X = pd.DataFrame({X.name: X})
            elif isinstance(X, vaex.expression.Expression):
                name = X.expression
                X = X.to_pandas_series()
                X.name = name
            if isinstance(y, vaex.expression.Expression):
                name = y.expression
                y = y.to_pandas_series()
                y.name = name
                self.target = name
        except:
            pass
        if isinstance(X, np.ndarray):
            X = self.infer(X)
        if y is None:
            y = X[self.target] if self.target else None
        elif isinstance(y, pd.Series):
            self.target = y.name
        elif isinstance(y, str):
            self.target = y
            y = X[y]
        X = self._set_features(X)
        return X, y

    def _set_features(self, X):
        if isinstance(X, pd.DataFrame):
            if self.features is None:
                self.features = list(X.columns)
        if self.features:
            if self.target in self.features:
                self.features.remove(self.target)
            if len(self.features) > 1:
                X = X[self.features]
            elif len(self.features) == 1:
                X = X[self.features[0]]
        if self.output_columns is None:
            self.output_columns = self.features
        return X

    def fit(self, df, y=None, validate: bool = True, check_na: bool = True) -> SklearnPipeline:
        if isinstance(df, str):
            df = read_sklearn_data(df)
        X, y = self._to_pandas(df, y)
        self.set_raw(self.to_raw(X))
        params = self.fit_params or {}
        self.pipeline = self.pipeline.fit(X=X, y=y, **params)
        if validate:
            self.validate(check_na=check_na)
        return self

    def transform(self, df, **kwargs) -> pd.DataFrame:
        """
        Transform the data based on the the pipeline.
        @param df: [DataFrame] data to transform
        @param kwargs: Transform
        @return:
        """
        copy = self.infer(df)
        features = self.features or copy.columns
        copy = self.pipeline.transform(copy[features])
        if isinstance(copy, np.ndarray) and copy.shape[1] == len(features):
            copy = pd.DataFrame(copy, columns=features)
        return copy

    def predict(self, df) -> np.ndarray:
        copy = self.infer(df)
        features = self.features or copy.columns
        if features is None:
            raise RuntimeError("Model is not trained yet")

        if len(features) == 1:
            features = features[0]
        X = copy[features] if features else copy
        return self.pipeline.predict(X)

    def inference(self, df, columns: List[str] = None, passthrough: bool = True, **kwargs) -> pd.DataFrame:
        """
        Returns the transformed data.
        Always tries to return a dataframe.
        If self.pipeline implements predict, returns the predictions as a new column (self.output_columns[0]).
        Else, returns self.pipeline.transform(df)
        @param df:
        @param columns: Only this columns are returns from the data frame.
        @param passthrough: If True, extra columns in the data passthrough.
        @param kwargs:
        @return:
        """
        copy = self.infer(df)
        features = self.features or copy.columns
        if features is None:
            raise RuntimeError("Model is not trained yet")
        passthrough_data = None
        if len(features) == 1:  # for text transformers and likewise
            features = features[0]
        X = copy[features] if features else copy
        if hasattr(self.pipeline, "predict") and self.output_columns is not None and len(self.output_columns) == 1:
            copy[self.output_columns[0]] = self.pipeline.predict(X)
        else:
            if passthrough:
                passthrough_columns = [column for column in copy.columns if column not in features]
                if columns is not None and len(columns) > 0:
                    passthrough_columns = [column for column in passthrough_columns if column in columns]
                if passthrough_columns:
                    passthrough_data = copy[passthrough_columns]
            copy = self.pipeline.transform(X)
            if isinstance(copy, np.ndarray) and self.output_columns and copy.shape[1] == len(self.output_columns):
                copy = pd.DataFrame(copy, columns=self.output_columns)
        if passthrough_data is not None and passthrough_data.shape[0] == copy.shape[0]:
            if isinstance(copy, pd.DataFrame):
                for column in passthrough_columns:
                    copy[column] = passthrough_data[column].values
            elif isinstance(copy, np.ndarray):
                copy = np.column_stack((copy, passthrough_data.values))
        if isinstance(copy, pd.DataFrame) and columns is not None and len(columns) > 0:
            copy = copy[[column for column in columns if column in copy]]
        return copy
