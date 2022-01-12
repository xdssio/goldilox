import pytest
import sklearn.pipeline
import vaex
from sklearn.metrics import accuracy_score
from vaex.ml.xgboost import XGBoostModel
from xgboost.sklearn import XGBClassifier

from goldilox import Pipeline
from goldilox.datasets import load_iris
from goldilox.sklearn.pipeline import Pipeline as SklearnPipeline
from goldilox.vaex.pipeline import VaexPipeline as VaexPipeline
from tests.test_utils import validate_persistence


@pytest.fixture()
def iris():
    # iris = load_iris()
    return load_iris()


def test_vaex_xgboost(iris):
    df, features, target = load_iris()
    df = vaex.from_pandas(df)
    train, test = df.ml.train_test_split(test_size=0.2, verbose=False)
    train['petal_ratio'] = train['petal_length'] / train['petal_width']

    booster = XGBoostModel(features=features + ['petal_ratio'],
                           target=target,
                           prediction_name='prediction',
                           num_boost_round=10, params={'verbosity': 0,
                                                       'objective': 'multi:softmax',
                                                       'num_class': 3})
    booster.fit(train)
    train = booster.transform(train)

    pipeline = VaexPipeline.from_dataframe(train)
    pipeline.set_variable('accuracy',
                          accuracy_score(pipeline.inference(test[features])['prediction'].values, test[target].values))

    sample = test.to_records(0)

    assert pipeline.inference(test).head(1).shape == (1, 7)
    assert pipeline.inference(sample).head(1).shape == (1, 7)
    assert pipeline.inference({'sepal_length': 5.9, 'petal_length': 4.2}).head(1).shape == (1, 7)
    assert pipeline.inference({'sepal_length': 5.9, 'petal_length': 4.2}, columns='prediction').head(1).shape == (1, 1)
    assert pipeline.inference({'sepal_length': 5.9, 'petal_length': 4.2, 'Y': 'new column'}).head(1).shape == (1, 8)


def test_xgboost_vaex_fit(iris):
    def fit(df):
        from vaex.ml.xgboost import XGBoostModel
        from sklearn.metrics import accuracy_score
        train, test = df.ml.train_test_split(test_size=0.2, verbose=False)

        features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
        target = 'target'

        booster = XGBoostModel(features=features,
                               target=target,
                               prediction_name='prediction',
                               num_boost_round=10, params={'verbosity': 0,
                                                           'objective': 'multi:softmax',
                                                           'num_class': 3})
        booster.fit(train)
        train = booster.transform(train)
        pipeline = VaexPipeline.from_dataframe(train)
        accuracy = accuracy_score(pipeline.inference(test[features])['prediction'].values,
                                  test[target].values)
        booster = XGBoostModel(features=features,
                               target=target,
                               prediction_name='predictions',
                               num_boost_round=10, params={'verbosity': 0,
                                                           'objective': 'multi:softmax',
                                                           'num_class': 3})
        booster.fit(df)
        df = booster.transform(df)
        df.variables['accuracy'] = accuracy
        return df

    df, features, target = iris
    df = vaex.from_pandas(df)
    pipeline = VaexPipeline.from_dataframe(df, fit=fit)
    sample = Pipeline.to_raw(df[features])
    assert pipeline.inference(sample).shape == df.head(1).shape
    data = df.to_records(0)
    pipeline.fit(df)
    pipeline = validate_persistence(pipeline)

    assert pipeline.inference(data).shape == (1, 6)
    assert pipeline.get_variable('accuracy')
    assert pipeline.raw == data
    assert list(pipeline.example.keys()) == features + [target] + ['predictions']


def test_xgboost_sklearn(iris):
    df, features, target = load_iris()
    X, y = df[features], df[target]
    sk_pipeline = sklearn.pipeline.Pipeline([('classifier', XGBClassifier(n_estimators=10, verbosity=0))])
    self = pipeline = SklearnPipeline.from_sklearn(sk_pipeline).fit(X, y)

    assert pipeline.inference(X).head(10).shape == (10, 5)
    assert pipeline.inference(X.values[:10]).shape == (10, 5)
    assert pipeline.inference(self.raw).shape == (1, 5)

    pipeline.fit(df)
    pipeline = validate_persistence(pipeline)
    assert pipeline.inference(X).head(10).shape == (10, 5)
    assert pipeline.inference(X.values[:10]).shape == (10, 5)
    assert pipeline.inference(self.raw).shape == (1, 5)

    # with a trained sklearn pipeline
    sample = Pipeline.to_raw(X)
    self = pipeline = SklearnPipeline.from_sklearn(sk_pipeline, raw=sample).fit(X, y)
    assert pipeline.inference(X).head(10).shape == (10, 5)
    assert pipeline.inference(X.values[:10]).shape == (10, 5)
    assert pipeline.inference(self.raw).shape == (1, 5)
