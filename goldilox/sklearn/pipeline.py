from time import time

import numpy as np
import pandas as pd
import traitlets

from goldilox import Pipeline

DEFAULT_OUTPUT_COLUMN = 'prediction'


class SklearnPipeline(traitlets.HasTraits, Pipeline):
    pipeline_type = 'sklearn'
    current_time = int(time())
    created = traitlets.Int(default_value=current_time, allow_none=False, help='Created time')
    updated = traitlets.Int(default_value=current_time, allow_none=False, help='Updated time')
    warnings = traitlets.Bool(default_value=True, help='Raise warnings')
    sample = traitlets.Any(default_value=None, allow_none=True, help='An example of the transformed dataset')
    pipeline = traitlets.Any(allow_none=False, help='A sklearn pipeline')
    features = traitlets.List(allow_none=True, help="A list of features")
    target = traitlets.Unicode(allow_none=True, help="A column to learn on fit")
    output_column = traitlets.Unicode(default_value=DEFAULT_OUTPUT_COLUMN, help="A column to learn on fit")

    @property
    def example(self):
        return self.inference(self.sample)

    @classmethod
    def from_sklearn(cls, pipeline, sample=None, features=None, target=None, output_column=DEFAULT_OUTPUT_COLUMN):
        """
        :param sklearn.preprocessing.pipeline.Pipeline pipeline: The skleran pipeline
        :param sample: dict [optional]: An example of data which will be queried in production (only the features)
                - If X is provided, would be the first row.
        :param features: list [optional]: A list of columns - if X is provided, will take it's columns
        :param target: str [optional]: The name of the target column - Used for retraining
        :param output_column: str [optional]: For sklearn estimator with 'predict' predictions a name.
        :return: SkleranPipeline object
        """
        if isinstance(features, pd.core.indexes.base.Index):
            features = list(features)
        elif isinstance(sample, dict) and features is None:
            features = list(sample.keys())
        elif sample is None and isinstance(features, list):
            sample = {key: 0 for key in features}
        if pipeline.__sklearn_is_fitted__() and sample is None and features is None:
            raise RuntimeError("For a fitted pipeline, please provide either the 'features' or 'sample'")

        return SklearnPipeline(pipeline=pipeline, features=features, target=target, sample=sample,
                               output_column=output_column)

    @property
    def fitted(self):
        return self.pipeline is not None and self.pipeline.__sklearn_is_fitted__()

    def infer(self, df):
        if isinstance(df, pd.DataFrame):
            return df.copy()
        if isinstance(df, np.ndarray):
            return pd.DataFrame(df, columns=self.features)
        if isinstance(df, dict):
            df = [df.copy()]
        if isinstance(df, list):
            return pd.DataFrame(df, columns=self.features)
        raise RuntimeError(f"could not infer type:{type(df)}")

    def fit(self, df, y=None, **kwargs):
        if y is None:
            y = df[self.target] if self.target else None
        elif isinstance(y, pd.Series):
            self.target = y.name
        elif isinstance(y, str):
            self.target = y
            y = df[y]

        if isinstance(df, pd.DataFrame):
            if self.features is None:
                self.features = list(df.columns)
                if self.target in self.features:
                    self.features.remove(self.target)
            X = df[self.features]
            self.sample = self._sample_df(X)
        if isinstance(df, np.ndarray):
            self.sample = list(df[0])
            X = df
        self.pipeline = self.pipeline.fit(X=X, y=y)
        return self

    @staticmethod
    def _sample_df(df):
        return df.iloc[0].to_dict()

    def inference(self, df, columns=None, **kwargs):
        copy = self.infer(df)
        features = self.features or copy.columns

        if hasattr(self.pipeline, 'predict'):
            copy[self.output_column] = self.pipeline.predict(copy[features])
        else:
            copy = self.pipeline.transform(copy[features])
            if isinstance(copy, np.ndarray) and copy.shape[1] == len(features):
                copy = pd.DataFrame(copy, columns=features)
        return copy

    def preprocess(self, df):
        pass

    def validate(self):
        try:
            self.inference(self.example)
        except Exception as e:
            return e
        return True
