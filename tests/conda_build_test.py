import pytest


@pytest.mark.skip("run manually from venvs")
def build_test_conda():
    import sklearn.pipeline
    from lightgbm.sklearn import LGBMClassifier

    from goldilox import Pipeline
    from goldilox.datasets import load_iris

    df, features, target = load_iris()
    sk_pipeline = sklearn.pipeline.Pipeline([("classifier", LGBMClassifier())])
    X = df[features]
    y = df[target]
    self = pipeline = Pipeline.from_sklearn(
        sk_pipeline, description="Lightgbm with sklearn"
    ).fit(X, y)
    path = 'conda_pipeline.pkl'
    pipeline.save(path)

    """
    glx environment conda_pipeline.pkl 
    delete:
    - backports
    - backports.functools_lru_cache
    - appnope
   
    glx build conda_pipeline.pkl --platform=linux/amd64
    
    """
