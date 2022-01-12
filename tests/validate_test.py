import pytest
import sklearn.pipeline
from sklearn.base import TransformerMixin, BaseEstimator

from goldilox import Pipeline
from goldilox.datasets import load_iris


@pytest.fixture()
def data():
    return load_iris()


@pytest.mark.skip("todo")
def test_validate_sklearn(data):
    df, features, target = load_iris()
    X, y = df[features], df[target]

    class DropNaTransformer(TransformerMixin, BaseEstimator):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X.dropna()

    sk_pipeline = sklearn.pipeline.Pipeline([('dropna', DropNaTransformer())]).fit(X)

    with pytest.raises(Exception) as e_info:
        Pipeline.from_sklearn(sk_pipeline).fit(X, y)
