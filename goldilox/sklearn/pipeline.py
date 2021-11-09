from time import time

import cloudpickle
import numpy as np
import pandas as pd
import traitlets

import goldilox
from goldilox import Pipeline
from goldilox.config import AWS_PROFILE, STATE, PIPELINE_TYPE, VERSION
from goldilox.utils import _is_s3_url

DEFAULT_OUTPUT_COLUMN = 'prediction'
TRAITS = '_trait_values'


class SklearnPipeline(traitlets.HasTraits, Pipeline):
    pipeline_type = 'sklearn'
    current_time = int(time())
    created = traitlets.Int(default_value=current_time, allow_none=False, help='Created time')
    updated = traitlets.Int(default_value=current_time, allow_none=False, help='Updated time')
    raw = traitlets.Any(default_value=None, allow_none=True, help='An example of the transformed dataset')
    pipeline = traitlets.Any(allow_none=False, help='A sklearn pipeline')
    features = traitlets.List(allow_none=True, help="A list of features")
    target = traitlets.Unicode(allow_none=True, help="A column to learn on fit")
    output_column = traitlets.Unicode(default_value=DEFAULT_OUTPUT_COLUMN, help="A column to learn on fit")
    fit_params = traitlets.Dict(allow_none=True, default_value={}, help="params to use on fit time")
    description = traitlets.Unicode(default_value='', help='Any notes to associate with a pipeline instance')
    variables = traitlets.Dict(default_value={}, help='Any variables to associate with a pipeline instance')

    @property
    def example(self):
        return self.inference(self.raw).to_dict(orient='records')[0]

    @classmethod
    def from_sklearn(cls, pipeline, raw=None, features=None, target=None, output_column=DEFAULT_OUTPUT_COLUMN,
                     variables=None, fit_params=None, description=''):
        """
        :param sklearn.preprocessing.pipeline.Pipeline pipeline: The skleran pipeline
        :param raw: dict [optional]: An example of data which will be queried in production (only the features)
                - If X is provided, would be the first row.
        :param features: list [optional]: A list of columns - if X is provided, will take it's columns
        :param target: str [optional]: The name of the target column - Used for retraining
        :param output_column: str [optional]: For sklearn estimator with 'predict' predictions a name.
        :param variables: dict [optional]: Variables to associate with the pipeline - fit_params automatically are added
        :param description: str [optional]: A pipeline description and notes in text
        :return: SkleranPipeline object
        """
        if isinstance(features, pd.core.indexes.base.Index):
            features = list(features)
        elif isinstance(raw, dict) and features is None:
            features = list(raw.keys())
        elif raw is None and isinstance(features, list):
            raw = {key: 0 for key in features}
        if hasattr(pipeline,
                   '__sklearn_is_fitted__') and pipeline.__sklearn_is_fitted__() and raw is None and features is None:
            raise RuntimeError("For a fitted pipeline, please provide either the 'features' or 'sample'")
        if variables is None:
            variables = {}
        if fit_params:
            variables.update(fit_params)
        return SklearnPipeline(pipeline=pipeline, features=features, target=target, sample=raw,
                               output_column=output_column, fit_params=fit_params, variables=variables,
                               description=description)

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
        try:
            import vaex
            if isinstance(df, vaex.dataframe.DataFrame):
                return df.to_pandas_df()
        except:
            pass
        raise RuntimeError(f"could not infer type:{type(df)}")

    def json_get(self):
        state = {STATE: cloudpickle.dumps(self),
                 PIPELINE_TYPE: self.pipeline_type,
                 VERSION: goldilox.__version__}

        return state

    @classmethod
    def loads(cls, state):
        if STATE in state:
            state = state[STATE]
        return cloudpickle.loads(state)

    def _to_pandas(self, X, y=None):
        try:
            import vaex
            if isinstance(X, vaex.dataframe.DataFrame):
                X = X.to_pandas_df()
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
        if y is None:
            y = X[self.target] if self.target else None
        elif isinstance(y, pd.Series):
            self.target = y.name
        elif isinstance(y, str):
            self.target = y
            y = X[y]
        return X, y

    def _get_features(self, X):
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
        return X

    def fit(self, df, y=None):
        X, y = self._to_pandas(df, y)
        X = self._get_features(X)
        self.raw = self._sample(X)
        params = self.fit_params or {}
        self.pipeline = self.pipeline.fit(X=X, y=y, **params)
        return self

    def transform(self, df, **kwargs):
        copy = self.infer(df)
        features = self.features or copy.columns
        copy = self.pipeline.transform(copy[features])
        if isinstance(copy, np.ndarray) and copy.shape[1] == len(features):
            copy = pd.DataFrame(copy, columns=features)
        return copy

    def inference(self, df, columns=None, **kwargs):
        copy = self.infer(df)
        features = self.features or copy.columns
        if len(features) == 1:  # for text transformers and likewise
            features = features[0]
        if hasattr(self.pipeline, 'predict'):
            copy[self.output_column] = self.pipeline.predict(copy[features])
        else:
            copy = self.pipeline.transform(copy[features])
            if isinstance(copy, np.ndarray) and copy.shape[1] == len(features):
                copy = pd.DataFrame(copy, columns=features)
        return copy

    def preprocess(self, df):
        pass

    def _validate_na(self, df):
        ret = True
        for column in df:
            tmp = df.copy()
            tmp[column] = None
            try:
                self.inference(tmp)
            except:
                ret = False
                print(f"Pipeline doesn't handle na for {column}")
        return ret


