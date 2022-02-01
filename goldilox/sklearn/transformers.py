import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from goldilox.sklearn.pipeline import SklearnPipeline, Pipeline


class Imputer(SklearnPipeline):
    IMPUTER = "imputer"

    def __init__(self, features=None, target=None, strategy='mean', fill_value=None):
        """

        @param features: The columns to impute
        @param strategy: can be "mean", "median", "most_frequent", "constant"
        @param fill_value: If strategy is constant, apply this value.
        """
        self.features = features
        self.target = target
        self.strategy = strategy
        self.fill_value = fill_value

        if strategy == 'constant' and fill_value is None:
            raise RuntimeError("If 'strategy' sets to 'constant- please set a fill_value")
        self.model = None

    def _create_imputer(self):
        return Pipeline.from_sklearn(ColumnTransformer([
            (Imputer.IMPUTER, SimpleImputer(strategy=self.strategy, fill_value=self.fill_value),
             self.features)], remainder='passthrough'), features=self.features)

    def _get_features(self, X):
        features = self.features
        if isinstance(features, str):
            features = [features]
        if features is None or (isinstance(features, list) and len(features) == 0):
            if isinstance(X, pd.DataFrame):
                features = list(X.columns)
            elif isinstance(X, np.ndarray):
                features = list(range(X.shape[1]))
            else:
                try:
                    import vaex.dataframe
                    if isinstance(X, vaex.dataframe):
                        features = X.get_column_names()
                except ModuleNotFoundError as e:
                    raise RuntimeError("Please provide features ot the imputer")
        return features

    def fit(self, X, y=None):
        X, y = self._to_pandas(X, y)
        self.model = self._create_imputer()
        self.model.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self.model.transform(X)
