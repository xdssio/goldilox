from tempfile import TemporaryDirectory

import vaex
from sklearn.linear_model import LogisticRegression
from vaex.ml.sklearn import Predictor

import goldilox.app
from goldilox import Pipeline
from goldilox.datasets import load_iris


def test_serving():
    df, features, target = load_iris()
    df = vaex.from_pandas(df)
    df.variables['variables'] = {'test': 'test'}
    model = Predictor(features=features, target=target, model=LogisticRegression(max_iter=1000))
    model.fit(df)
    df = model.transform(df)
    pipeline = Pipeline.from_vaex(df)

    path = str(TemporaryDirectory().name) + '/pipeline.pkl'
    pipeline.save(path)
    s = goldilox.app.GoldiloxServer(path=path)
    pipeline.export_mlflow('tests/mlops/mlflow')
