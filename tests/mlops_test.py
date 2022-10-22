from glob import glob
from tempfile import TemporaryDirectory

import numpy as np
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
    df.variables['variables'] = {'test': 'test'}
    df.variables['a'] = 1
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
                            prediction_name='lgbm',
                            num_boost_round=500,
                            params={'verbosity': -1,
                                    'objective': 'multiclass',
                                    'num_class': 3})
    booster.fit(df)
    df = booster.transform(df)

    @vaex.register_function()
    def argmax(ar, axis=1):
        return np.argmax(ar, axis=axis)

    df.add_function('argmax', argmax)
    df['prediction'] = df['lgbm'].argmax()
    pipeline = Pipeline.from_vaex(df, predict_column='lgbm')
    pipeline.set_variable('b', 2)
    pipeline.set_variable('description', 'description')
    return pipeline


def manually_test_docker():
    pipeline = _pipeline()

    # docker with nginx
    print(requests.get('http://127.0.0.1:8080/ping', json=[pipeline.raw]).json())
    print(requests.post('http://127.0.0.1:8080/invocations', json=[pipeline.raw]).json())
    print(requests.post('http://127.0.0.1:8080/inference', json=[pipeline.raw]).json())
    print(requests.get('http://127.0.0.1:8080/example').text)

    # docker without nginx
    print(requests.get('http://127.0.0.1:5000/docs').text)
    print(requests.get('http://127.0.0.1:5000/ping').text)
    print(requests.post('http://127.0.0.1:5000/invocations', json=[pipeline.raw]).json())
    print(requests.post('http://127.0.0.1:5000/inference', json=[pipeline.raw]).json())

    # mlflow
    print(requests.post('http://127.0.0.1:5000/invocations', json=[pipeline.raw]).json())
    print(requests.post('http://127.0.0.1:5000/ping', json=[pipeline.raw]).json())

    # ray
    print(requests.post('http://127.0.0.1:5000/PipelineDeployment/inference', json=pipeline.raw).json())


def test_export_gunicorn():
    pipeline = _pipeline()

    assert len(pipeline.variables) == 4
    assert pipeline.description == 'description'

    path = str(TemporaryDirectory().name) + '/pipeline'

    pipeline.export_gunicorn(path)
    files_str = ' '.join(glob(path + "/*"))
    assert "pipeline/requirements.txt" in files_str
    assert "pipeline/pipeline.pkl" in files_str
    assert "pipeline/gunicorn.conf.py" in files_str
    assert "pipeline/wsgi.py" in files_str
    assert not "pipeline/nginx.conf" in files_str
    pipeline.export_gunicorn(path, nginx=True)
    files_str = ' '.join(glob(path + "/*"))
    assert "pipeline/nginx.conf" in files_str


def test_mlflow():
    df, features, target = load_iris()
    df = vaex.from_pandas(df)

    path = str(TemporaryDirectory().name) + '/pipeline'
    pipeline.export_mlflow(path)

    """mlflow models serve -m <path> --no-conda"""
    import mlflow.pyfunc
    model = mlflow.pyfunc.load_model(path)
    assert len(model.predict(pipeline.to_pandas(pipeline.raw))) == 1
    import requests
    print(requests.post('http://127.0.0.1:5000/invocations', json=[pipeline.raw]).json())


def test_export_ray():
    pipeline = _pipeline()
    path = str(TemporaryDirectory().name) + '/pipeline'
    pipeline.export_ray(path)
    files_str = ' '.join(glob(path + "/*"))
    assert "pipeline/requirements.txt" in files_str
    assert "pipeline/pipeline.pkl" in files_str
    assert "pipeline/main.py" in files_str
