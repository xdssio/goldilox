import io
import json
from tempfile import TemporaryDirectory
import pandas as pd
import pytest
import vaex
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from vaex.ml.sklearn import Predictor

import goldilox
import goldilox.app
from goldilox.datasets import load_iris


def _client():
    df, features, target = load_iris()
    df = vaex.from_pandas(df)
    df.variables['variables'] = {'test': 'test'}
    df.variables['a'] = 1
    model = Predictor(features=features, target=target, model=LogisticRegression(max_iter=1000))
    model.fit(df)
    df = model.transform(df)
    pipeline = goldilox.Pipeline.from_vaex(df)
    pipeline.set_variable('b', 2)
    pipeline.set_variable('description', 'description')
    model_path = str(TemporaryDirectory().name) + '/pipeline.pkl'
    pipeline.save(model_path)
    return TestClient(goldilox.app.get_app(model_path))


def _raw():
    return {'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2}


@pytest.fixture()
def raw():
    return _raw()


@pytest.fixture()
def client():
    return _client()


def test_app_variables(client):
    response = client.get('/variables').json()
    assert response['a'] == 1
    assert response['b'] == 2
    assert response['description'] == 'description'
    assert response['variables']['test'] == 'test'


def test_app_description(client):
    assert client.get('/description').json() == 'description'


def test_app_inference(client, raw):
    predictions = client.post('/inference?columns=prediction,target', json=[raw, raw]).json()
    for prediction in predictions:
        assert prediction['prediction'] == 0
        assert prediction['target'] is None


def test_app_invocations(client, raw):
    client = _client()
    raw = _raw()

    predictions = client.post('/invocations', json=[raw, raw]).json()
    for prediction in predictions:
        assert prediction['prediction'] == 0
        assert prediction['target'] is None
    df = pd.DataFrame([raw, raw])

    predictions = client.post('/invocations', headers={'Content-Type': 'text/csv'},
                              files=[("parameter", ("file1.txt", io.BytesIO(
                                  bytes(df.to_csv(line_terminator='\r\n', index=False), encoding='utf-8'))))]).json()


def test_app_ping(client):
    client = _client()
    assert client.get('/ping').json() == 'pong'


def test_app_example(client):
    response = client.get('/example').json()[0]
    assert response['sepal_length'] == 5.1
    assert response['sepal_width'] == 3.5
    assert response['petal_length'] == 1.4
    assert response['petal_width'] == 0.2
    assert response['prediction'] == 0
    assert response['target'] == 0
