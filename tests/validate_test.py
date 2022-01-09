import pandas as pd
import pytest
import sklearn.pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.datasets import load_iris

from goldilox import Pipeline

columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
           'petal width (cm)']
target = 'target'


@pytest.fixture()
def data():
    iris = load_iris()
    features = iris.feature_names
    df = pd.DataFrame(iris.data, columns=features)
    df['target'] = iris.target
    df['sepal length (cm)'][0] = None
    return df


def test_validate_sklearn(data):
    df = data.copy()

    class DropNaTransformer(TransformerMixin, BaseEstimator):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X.dropna()

    sk_pipeline = sklearn.pipeline.Pipeline([('dropna', DropNaTransformer())]).fit(
        df[columns])

    with pytest.raises(Exception) as e_info:
        Pipeline.from_sklearn(sk_pipeline).fit(df[columns], df[target])


