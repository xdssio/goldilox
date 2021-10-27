import numpy as np
import vaex
from vaex.ml.sklearn import Predictor

from goldilox import Pipeline
from vaex.ml.datasets import load_iris_1e5
import pytest
import sklearn.pipeline
@pytest.fixture()
def df():
    # df = load_iris_1e5()
    return load_iris_1e5()

def test_flaml_vaex(df):
    features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    target = 'class_'

    from flaml import AutoML
    model = AutoML()
    automl_settings = {
        "time_budget": 10,  # in seconds
        "metric": 'accuracy',
        "task": 'classification'
    }
    model.fit(df[features].values, y_train=df[target].values,
              **automl_settings)

    @vaex.register_function(on_expression=False)
    def automl(*columns):
        data = np.array(columns).T
        return model.predict(data)

    df.add_function('automl', automl)
    df['predictions'] = df.func.automl(*tuple([df[col] for col in features]))

    pipeline = Pipeline.from_vaex(df)
    assert pipeline.inference(pipeline.sample).shape == (1, 6)


def test_flaml_sklearn(df):
    features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    target = 'class_'

    from flaml import AutoML

    automl_settings = {
        "automl__time_budget": 10,  # in seconds
        "automl__metric": 'accuracy',
        "automl__task": 'classification'
    }
    sk_pipeline = sklearn.pipeline.Pipeline([('automl', AutoML())])
    pipeline = Pipeline.from_sklearn(sk_pipeline, features=features, target=target).fit(df, **automl_settings)

    assert pipeline.inference(pipeline.sample).shape == (1, 5)