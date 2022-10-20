from glob import glob
from tempfile import TemporaryDirectory

import numpy as np
import vaex
from sklearn.linear_model import LogisticRegression
from vaex.ml.sklearn import Predictor

from goldilox import Pipeline
from goldilox.datasets import load_iris


def test_export_gunicorn():
    df, features, target = load_iris()
    df = vaex.from_pandas(df)
    df.variables['variables'] = {'test': 'test'}
    df.variables['a'] = 1
    model = Predictor(features=features, target=target, model=LogisticRegression(max_iter=1000))
    model.fit(df)
    df = model.transform(df)
    pipeline = Pipeline.from_vaex(df)
    pipeline.set_variable('b', 2)
    pipeline.set_variable('description', 'description')
    assert len(pipeline.variables) == 4
    assert pipeline.description

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
    assert "pipeline/serve.py" in files_str
    assert "pipeline/nginx.conf" in files_str
    # pipeline.export_gunicorn('tests/mlops/gunicorn')
    # pipeline.save('./pipeline.pkl')


def test_mlflow():
    df, features, target = load_iris()
    df = vaex.from_pandas(df)

    from vaex.ml.lightgbm import LightGBMModel

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

    path = str(TemporaryDirectory().name) + '/pipeline'
    pipeline.export_mlflow(path)

    """mlflow models serve -m <path> --no-conda"""
    import mlflow.pyfunc
    model = mlflow.pyfunc.load_model(path)
    assert len(model.predict(pipeline.to_pandas(pipeline.raw))) == 1
    import requests
    print(requests.post('http://127.0.0.1:5000/invocations', json=[pipeline.raw]).json())


def test_export_ray():
    df, features, target = load_iris()
    df = vaex.from_pandas(df)
    df['predictions'] = df['target'] + 0

    pipeline = Pipeline.from_vaex(df)

    path = str(TemporaryDirectory().name) + '/pipeline'

    pipeline.export_ray(path)
    files_str = ' '.join(glob(path + "/*"))
    assert "pipeline/requirements.txt" in files_str
    assert "pipeline/pipeline.pkl" in files_str
    assert "pipeline/main.py" in files_str
    # import requests
    # requests.post('http://127.0.0.1:5000/PipelineDeployment/inference', json=pipeline.raw).json()
