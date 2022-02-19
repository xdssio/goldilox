import numpy as np
import pytest
import sklearn.pipeline
import vaex

from goldilox import Pipeline
from goldilox.datasets import load_iris
from tests.test_utils import validate_persistence


@pytest.fixture()
def iris():
    return load_iris()


def test_flaml_vaex(iris, tmpdir):
    from flaml import AutoML
    df, features, target = iris
    df = vaex.from_pandas(df)
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
    pipeline = validate_persistence(pipeline)
    assert pipeline.inference(pipeline.raw).shape == (1, 6)


def test_flaml_sklearn(iris):
    from flaml import AutoML
    df, features, target = iris
    automl_settings = {
        "automl__time_budget": 10,  # in seconds
        "automl__metric": 'accuracy',
        "automl__task": 'classification'
    }
    pipeline = Pipeline.from_sklearn(sklearn.pipeline.Pipeline([('automl', AutoML())]),
                                     features=features, target=target, fit_params=automl_settings).fit(df)

    assert pipeline.inference(pipeline.raw).shape == (1, 5)
