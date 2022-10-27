import io
from tempfile import TemporaryDirectory

import pandas as pd
import vaex
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from vaex.ml.sklearn import Predictor

import goldilox
import goldilox.app
from goldilox.datasets import load_iris

app_pipeline = goldilox.app.AppPipeline()


def _pipeline():
    df, features, target = load_iris()
    df = vaex.from_pandas(df)
    df.variables['variables'] = {'test': 'test'}
    df.variables['a'] = 1
    model = Predictor(features=features, target=target, model=LogisticRegression(max_iter=1000))
    model.fit(df)
    df = model.transform(df)
    pipeline = goldilox.Pipeline.from_vaex(df)
    pipeline.raw.pop('target')
    pipeline.set_variable('b', 2)
    pipeline.set_variable('description', 'description')
    return pipeline


pipeline = _pipeline()
model_path = str(TemporaryDirectory().name) + '/pipeline.pkl'
pipeline.save(model_path)

app = goldilox.app.get_app(model_path)
raw = {'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2}


@app.on_event('startup')
async def load_model():
    app_pipeline.pipeline = pipeline
    app_pipeline.meta = app_pipeline.pipeline.meta
    app_pipeline.pipeline.example  # warmup and load packages


def test_app_variables():
    with TestClient(app) as client:
        response = client.get('/variables').json()
    assert response['a'] == 1
    assert response['b'] == 2
    assert response['description'] == 'description'
    assert response['variables']['test'] == 'test'


def test_app_description():
    with TestClient(app) as client:
        assert client.get('/description').json() == 'description'


def test_app_inference():
    with TestClient(app) as client:
        predictions = client.post('/inference?columns=prediction,target', json=[raw, raw]).json()
    for prediction in predictions:
        assert prediction['prediction'] == 0
        assert prediction['target'] is None

    with TestClient(app) as client:
        prediction = client.post('/inference?columns=prediction,target', json=raw).json()[0]
        assert prediction['prediction'] == 0
        assert prediction['target'] is None


def test_app_invocations():
    # standard
    payload = [raw, raw]
    with TestClient(app) as client:
        predictions = client.post('/invocations', json=payload).json()
    for prediction in predictions:
        assert prediction['prediction'] == 0
        assert prediction['target'] is None

    # mlflow format
    payload = {"data": [list(raw.values()), list(raw.values())]}
    with TestClient(app) as client:
        predictions = client.post('/invocations?columns=prediction', json=payload).json()
    assert predictions[0]['prediction'] == 0
    df = pd.DataFrame([raw, raw])
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=True)

    # csv format
    with TestClient(app) as client:
        response = client.post('/invocations', headers={'Content-Type': 'text/csv'}, data=buffer.getvalue()).json()
    predictions = pd.read_csv(io.StringIO(response))
    assert predictions.shape[1] == df.shape[1] + 2
    with TestClient(app) as client:
        response = client.post('/invocations?columns=target,prediction', headers={'Content-Type': 'text/csv'},
                               data=buffer.getvalue()).json()
    predictions = pd.read_csv(io.StringIO(response))
    assert predictions.shape[1] == 2


def test_app_ping():
    with TestClient(app) as client:
        assert client.get('/ping').json() == 'pong'


def test_app_example():
    with TestClient(app) as client:
        response = client.get('/example').json()[0]
    assert response['sepal_length'] == 5.1
    assert response['sepal_width'] == 3.5
    assert response['petal_length'] == 1.4
    assert response['petal_width'] == 0.2
    assert response['prediction'] == 0


def test_server_validate_params():
    def default():
        return 'default'

    assert goldilox.app.GoldiloxServer._extract_params('python -m goldilox.app --bind=0.0.0.0:5000', default,
                                                       ['-b ', '--bind='])[
               1] == '0.0.0.0:5000'
    assert \
        goldilox.app.GoldiloxServer._extract_params('python -m goldilox.app -b 0.0.0.0:5000', default,
                                                    ['-b ', '--bind='])[
            1] == '0.0.0.0:5000'
    assert goldilox.app.GoldiloxServer._extract_params('python -m goldilox.app', default, ['-b ', '--bind='])[
               1] == 'default'

    assert goldilox.app.GoldiloxServer._extract_params('python -m goldilox.app -t 60', default, ['-t ', '--timeout='])[
               1] == '60'
    assert \
        goldilox.app.GoldiloxServer._extract_params('python -m goldilox.app --timeout=60', default,
                                                    ['-t ', '--timeout='])[
            1] == '60'
    assert goldilox.app.GoldiloxServer._extract_params('python -m goldilox.app ', default, ['t ', '--timeout='])[
               1] == 'default'

    assert goldilox.app.GoldiloxServer._extract_params('python -m goldilox.app -w 1', default, ['-w ', '--workers='])[
               1] == '1'
    assert \
        goldilox.app.GoldiloxServer._extract_params('python -m goldilox.app --workers=1', default,
                                                    ['-w ', '--workers='])[
            1] == '1'
    assert goldilox.app.GoldiloxServer._extract_params('python -m goldilox.app ', default, ['-w ', '--workers='])[
               1] == 'default'

    pipeline = _pipeline()
    path = str(TemporaryDirectory().name) + '/pipeline.pkl'
    pipeline.save(path)
    server = goldilox.app.GoldiloxServer(path, options=['python -m goldilox.app --workers=1 -b 0.0.0.0:5001 -t 60'])
    assert server.workers == '1'
    assert server.bind == '0.0.0.0:5001'
    assert server.timeout == '60'
