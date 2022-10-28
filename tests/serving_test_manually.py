import tempfile

import numpy as np
import pytest
import requests
import sklearn.pipeline
import vaex
from sklearn.linear_model import LogisticRegression
from vaex.ml.lightgbm import LightGBMModel
from vaex.ml.sklearn import Predictor

import goldilox.sklearn.transformers
from goldilox import Pipeline
from goldilox.datasets import load_iris


def _pipeline():
    df, features, target = load_iris()
    df = vaex.from_pandas(df)
    names = {0: "setosa", 1: "versicolor", 2: "virginica"}
    df.variables['names'] = names
    model = Predictor(features=features,
                      target=target,
                      prediction_name='linear_regression',
                      model=sklearn.pipeline.Pipeline([('imputer', goldilox.sklearn.transformers.Imputer()),
                                                       ('model', LogisticRegression(max_iter=500))])
                      )
    model.fit(df)
    df = model.transform(df)

    booster = LightGBMModel(features=features,
                            target=target,
                            prediction_name='probabilities',
                            num_boost_round=10,
                            params={'verbosity': -1,
                                    'objective': 'multiclass',
                                    'num_class': 3})
    booster.fit(df)
    df = booster.transform(df)

    @vaex.register_function()
    def argmax(ar, axis=1):
        return np.argmax(ar, axis=axis)

    df.add_function('argmax', argmax)
    df['prediction'] = df['probabilities'].argmax()

    df["label"] = df["prediction"].map(names)
    pipeline = Pipeline.from_vaex(df, predict_column='label')

    pipeline.set_variable('num_class', 3)
    pipeline.set_variable('objective', 'multiclass')
    pipeline.set_variable('description', 'description')

    pipeline.meta.set_requirements(['sklearn', 'lightgbm', 'vaex-ml', 'vaex-core', 'goldilox'])
    pipeline.raw.pop('target')
    return pipeline


@pytest.mark.skip("running locally")
def manually_test_docker():
    pipeline = _pipeline()
    import json
    json.dumps(pipeline.raw)
    pipeline.save('pipeline.pkl')
    pipeline.predict(pipeline.raw)
    pipeline.export_mlflow('tests/mlops/mlflow')

    # docker with nginx
    print(requests.get('http://127.0.0.1:8080/ping', json=[pipeline.raw]).json())
    print(requests.post('http://127.0.0.1:8080/invocations', json=[pipeline.raw]).json())
    print(requests.post('http://127.0.0.1:8080/inference', json=[pipeline.raw]).json())
    print(requests.get('http://127.0.0.1:8080/example').text)
    """curl -XPOST -H "Content-Type: application/json"  http://127.0.0.1:8080/inference -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}' """

    # mlflow
    print(requests.post('http://127.0.0.1:8080/invocations', json=[pipeline.raw]).json())

    # ray
    print(requests.post('http://127.0.0.1:8080/PipelineDeployment/inference', json=pipeline.raw).json())


@pytest.mark.skip("running locally")
def test_cloud_persistence():
    pipeline = _pipeline()
    # pipeline.save('pipeline.pkl')
    local_path = str(tempfile.TemporaryDirectory().name) + '/pipeline.pkl'
    assert len(Pipeline.from_file(pipeline.save(local_path)).inference(pipeline.raw)) == 1

    cloud_path = 's3://pipelines.goldilox/example_pipeline/pipeline0.0.20.pkl'
    assert len(Pipeline.from_file(pipeline.save(cloud_path)).inference(pipeline.raw)) == 1
    Pipeline.from_file(cloud_path).variables
