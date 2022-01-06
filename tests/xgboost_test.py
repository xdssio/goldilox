import pytest
import sklearn.pipeline
from sklearn.metrics import accuracy_score
from vaex.ml.datasets import load_iris_1e5
from vaex.ml.xgboost import XGBoostModel
from xgboost.sklearn import XGBClassifier

from goldilox.sklearn.pipeline import Pipeline as SklearnPipeline
from goldilox.vaex.pipeline import VaexPipeline as VaexPipeline
from tests.test_utils import validate_persistence


@pytest.fixture()
def iris():
    # iris = load_iris_1e5()
    return load_iris_1e5()


def test_vaex_xgboost(iris):
    train, test = iris.ml.train_test_split(test_size=0.2, verbose=False)
    features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    target = 'class_'
    train['X'] = train['petal_length'] / train['petal_width']

    booster = XGBoostModel(features=features,
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
        target = 'class_'

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

    df = iris.copy()
    pipeline = VaexPipeline.from_dataframe(df, fit=fit)
    data = df.to_records(0)
    data.pop('class_')
    assert pipeline.inference(data).shape == df.head(1).shape
    data = df.to_records(0)
    pipeline.fit(df)
    pipeline = validate_persistence(pipeline)

    assert pipeline.inference(data).shape == (1, 6)
    assert pipeline.get_variable('accuracy')
    assert pipeline.raw == data
    assert list(pipeline.example.keys()) == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class_',
                                             'predictions']


def test_xgboost_sklearn(iris):
    df = iris.copy()
    features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    target = 'class_'
    sk_pipeline = sklearn.pipeline.Pipeline([('classifier', XGBClassifier(n_estimators=10, verbosity=0))])
    X = df[features]
    y = df[target]
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
    sample = X.head(1).to_records()[0]
    self = pipeline = SklearnPipeline.from_sklearn(sk_pipeline, raw=sample).fit(X, y)
    assert pipeline.inference(X).head(10).shape == (10, 5)
    assert pipeline.inference(X.values[:10]).shape == (10, 5)
    assert pipeline.inference(self.raw).shape == (1, 5)
