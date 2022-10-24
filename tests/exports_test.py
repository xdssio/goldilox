from glob import glob
from tempfile import TemporaryDirectory

import numpy as np
import vaex
from lightgbm.sklearn import LGBMClassifier
from vaex.ml.sklearn import Predictor

from goldilox import Pipeline
from goldilox.datasets import load_iris


def test_export_sageamker():
    df, features, target = load_iris()
    df = vaex.from_pandas(df)

    def fit(df):
        lgb = Predictor(model=LGBMClassifier(verbose=-1, objective='multiclass', num_class=3), features=features,
                        target=target, prediction_name='prediction')
        lgb.fit(df)
        df = lgb.transform(df)
        return df

    pipeline = Pipeline.from_vaex(df, fit=fit)

    pipeline.save('tests/mlops/sagemaker/pipeline.pkl')


def test_export_gunicorn():
    df, features, target = load_iris()
    df = vaex.from_pandas(df)

    pipeline = Pipeline.from_vaex(df)
    path = str(TemporaryDirectory().name) + '/pipeline'
    pipeline.export_gunicorn(path)
    files = [f.replace(path + '/', '') for f in glob(path + "/*")]

    assert "requirements.txt" in files
    assert "pipeline.pkl" in files
    assert "gunicorn.conf.py" in files
    assert "wsgi.py" in files


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
    pipeline = Pipeline.from_vaex(df)

    path = str(TemporaryDirectory().name) + '/pipeline'
    pipeline.export_mlflow(path)

    """mlflow models serve -m <path> --no-conda"""
    import mlflow.pyfunc
    model = mlflow.pyfunc.load_model(path)
    assert len(model.predict(pipeline.to_pandas(pipeline.raw))) == 1


def test_export_ray():
    df, features, target = load_iris()
    df = vaex.from_pandas(df)
    df['predictions'] = df['target'] + 0

    pipeline = Pipeline.from_vaex(df)

    path = str(TemporaryDirectory().name) + '/pipeline'

    pipeline.export_ray(path)
    files = [f.replace(path + '/', '') for f in glob(path + "/*")]
    assert "requirements.txt" in files
    assert "pipeline.pkl" in files
    assert "main.py" in files
