import numpy as np
import pandas as pd
import pytest
import vaex
from goldilox import Pipeline
from hdbscan import HDBSCAN, approximate_predict
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.datasets import make_blobs

n_features = 10
features = [f"f{i}" for i in range(n_features)]


@pytest.fixture()
def data():
    blobs, labels = make_blobs(n_samples=2000, n_features=n_features)
    df = pd.DataFrame(blobs, columns=features)
    df['target'] = labels
    return df


def test_hdbscan_vaex(data):
    df = vaex.from_pandas(data)
    model = HDBSCAN(prediction_data=True)
    model.fit(df[features], df['target'])

    @vaex.register_function()
    def hdbscan(*columns):
        data = np.array(columns).T
        labels, _ = approximate_predict(model, data)
        return labels

    df.add_function('hdbscan', hdbscan)
    df['label'] = df.func.hdbscan(*features)

    pipeline = Pipeline.from_vaex(df)
    assert 'label' in pipeline.inference(pipeline.raw)


def test_hdbscan_sklearn(data):
    df = data.copy()
    features = [f"f{i}" for i in range(n_features)]
    target = "target"
    X = df[features]
    y = df[target]

    class HDBSCANTransformer(TransformerMixin, BaseEstimator):

        def __init__(self, prediction_column='label', **kwargs):
            kwargs['prediction_data'] = True
            self.model = HDBSCAN(**kwargs)
            self.prediction_column = prediction_column

        def fit(self, X, y):
            self.model.fit(X, y)
            return self

        def predict(self, X):
            labels, strength = approximate_predict(self.model, X)
            return labels

        def transform(self, X):
            X[self.prediction_column] = self.predict(X)
            return X

    self = model = HDBSCANTransformer().fit(X, y)
    pipeline = Pipeline.from_sklearn(model).fit(X, y)
    assert pipeline.output_column in pipeline.inference(X)
    assert pipeline.raw == Pipeline.to_raw(X)
    assert pipeline.features == features
    assert pipeline.target == target
