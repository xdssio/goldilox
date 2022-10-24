import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline

from goldilox import Pipeline
from goldilox.datasets import load_iris


def _pipeline():
    df, features, target = load_iris()
    pipeline = SklearnPipeline([('classifier', LogisticRegression(max_iter=500))])
    pipeline = Pipeline.from_sklearn(pipeline=pipeline, features=features, target=target).fit(df)
    pipeline.meta.set_requirements(['scikit-learn'])
    return pipeline


@pytest.mark.skip("running manually")
def test_docker_factory():
    pipeline = _pipeline()
    path = 'pipeline.pkl'
    pipeline.save(path)
    from goldilox.app.docker import DockerFactory
    factory = DockerFactory(path=path, name='test')
    print(' '.join(factory._get_build_command()))
    platform = 'linux/arm64'
    factory.build(platform='linux/arm64')

    """
    docker build -f=/Users/yonatanalexander/development/xdss/goldilox/goldilox/app/Dockerfile -t=test --build-arg PYTHON_VERSION=3.10.8 --build-arg PYTHON_IMAGE=python:3.10.8-slim-bullseye --build-arg GOLDILOX_VERSION=0.0.16 --target venv-image --build-arg PIPELINE_FILE=pipeline.pkl .
    """


pipeline = _pipeline()
path = 'pipeline.pkl'
pipeline.save(path)
