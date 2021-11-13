import vaex

from goldilox.sklearn.pipeline import Pipeline as SklearnPipeline
from goldilox.vaex.pipeline import VaexPipeline as VaexPipeline
from vaex.ml.datasets import load_iris_1e5
import pytest

from tests.test_utils import validate_persistance


@pytest.fixture()
def iris():
    # iris = load_iris_1e5()
    return load_iris_1e5()


def test_lightgbm_vaex(iris, tmpdir):
    import vaex
    import numpy as np
    from vaex.ml.lightgbm import LightGBMModel
    from sklearn.metrics import accuracy_score

    train, test = iris.ml.train_test_split(test_size=0.2, verbose=False)
    features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    target = 'class_'
    train['X'] = train['petal_length'] / train['petal_width']

    booster = LightGBMModel(features=features,
                            target=target,
                            prediction_name='predictions',
                            num_boost_round=500, params={'verbose': -1,
                                                         'objective': 'multiclass',
                                                         'num_class': 3})
    booster.fit(train)
    train = booster.transform(train)

    @vaex.register_function()
    def argmax(ar, axis=1):
        return np.argmax(ar, axis=axis)

    train.add_function('argmax', argmax)
    train['prediction'] = train['predictions'].argmax()
    pipeline = VaexPipeline.from_dataframe(train)
    pipeline.set_variable('accuracy',
                          accuracy_score(pipeline.inference(test[features])['prediction'].values, test[target].values))

    data = test.to_records(0)
    pipeline = validate_persistance(pipeline)
    assert pipeline.inference(test).head(1).shape == (1, 8)
    assert pipeline.inference(data).head(1).shape == (1, 8)
    assert pipeline.inference({'sepal_length': 5.9, 'petal_length': 4.2}).head(1).shape == (1, 8)
    assert pipeline.inference({'sepal_length': 5.9, 'petal_length': 4.2}, columns='prediction').head(1).shape == (1, 1)
    assert pipeline.inference({'sepal_length': 5.9, 'petal_length': 4.2, 'Y': 'new column'}).head(1).shape == (1, 9)


def test_lightgbm_vaex_fit(iris, tmpdir):

    def fit(df):
        import vaex
        import numpy as np
        from vaex.ml.lightgbm import LightGBMModel
        from sklearn.metrics import accuracy_score
        train, test = df.ml.train_test_split(test_size=0.2, verbose=False)

        features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
        target = 'class_'

        booster = LightGBMModel(features=features,
                                target=target,
                                prediction_name='predictions',
                                num_boost_round=500, params={'verbose': -1,
                                                             'objective': 'multiclass',
                                                             'num_class': 3})
        booster.fit(df)

        @vaex.register_function()
        def argmax(ar, axis=1):
            return np.argmax(ar, axis=axis)

        train = booster.transform(df)
        train.add_function('argmax', argmax)
        train['prediction'] = train['predictions'].argmax()

        pipeline = VaexPipeline.from_dataframe(train)
        accuracy = accuracy_score(pipeline.inference(test[features])['prediction'].values,
                                  test[target].values)
        booster = LightGBMModel(features=features,
                                target=target,
                                prediction_name='predictions',
                                num_boost_round=500, params={'verbose': -1,
                                                             'objective': 'multiclass',
                                                             'num_class': 3})
        booster.fit(df)
        df = booster.transform(df)
        df.add_function('argmax', argmax)
        df['prediction'] = df['predictions'].argmax()
        df.variables['accuracy'] = accuracy
        return df

    df = iris.copy()
    pipeline = VaexPipeline.from_dataframe(df, fit=fit)
    data = df.to_records(0)
    assert pipeline.inference(data).shape == df.head(1).shape
    pipeline.fit(df)
    pipeline = validate_persistance(pipeline)
    assert pipeline.inference(data).shape == (1, 7)
    assert pipeline.get_variable('accuracy')
    assert pipeline.raw == data
    assert list(pipeline.example.keys()) == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class_',
                                             'predictions', 'prediction']


def test_lightgbm_sklearn(iris, tmpdir):
    from lightgbm.sklearn import LGBMClassifier
    import sklearn.pipeline

    df = iris.copy()
    features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    target = 'class_'
    sk_pipeline = sklearn.pipeline.Pipeline([('classifier', LGBMClassifier())])
    X = df[features]
    y = df[target]
    self = pipeline = SklearnPipeline.from_sklearn(sk_pipeline).fit(X, y)

    assert pipeline.inference(X).head(10).shape == (10, 5)
    assert pipeline.inference(X.values[:10]).shape == (10, 5)
    assert pipeline.inference(self.raw).shape == (1, 5)
    pipeline.fit(df)
    path = str(tmpdir) + '/model.pkl'
    pipeline.save(path)
    pipeline = SklearnPipeline.from_file(path)
    assert pipeline.inference(X).head(10).shape == (10, 5)
    assert pipeline.inference(X.values[:10]).shape == (10, 5)
    assert pipeline.inference(self.raw).shape == (1, 5)

    # with a trained sklearn pipeline
    sample = X.head(1).to_records()[0]
    self = pipeline = SklearnPipeline.from_sklearn(sk_pipeline, raw=sample).fit(X, y)
    pipeline.save(path)
    pipeline = SklearnPipeline.from_file(path)
    assert pipeline.inference(X).head(10).shape == (10, 5)
    assert pipeline.inference(X.values[:10]).shape == (10, 5)
    assert pipeline.inference(self.raw).shape == (1, 5)

