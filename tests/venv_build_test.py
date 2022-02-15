from tempfile import TemporaryDirectory

import pytest


@pytest.mark.skip("run manually")
def build_test_venv():
    import sklearn.pipeline
    from lightgbm.sklearn import LGBMClassifier

    from goldilox import Pipeline
    from goldilox.datasets import load_iris
    from goldilox.sklearn.transformers import Imputer
    df, features, target = load_iris()
    sk_pipeline = sklearn.pipeline.Pipeline([('imputer', Imputer(features=features)), ("classifier", LGBMClassifier())])
    X = df[features]
    y = df[target]
    pipeline = Pipeline.from_sklearn(
        sk_pipeline, description="Lightgbm with sklearn"
    ).fit(X, y)
    # pipeline.save("venv_pipeline.pkl")

    path = str(TemporaryDirectory().name) + '/venv_pipeline.pkl'
    pipeline.save(path)
    from goldilox.app.cli import build
    build(path, name="goldilox-venv-test", image=None, platform='linux/amd64')
