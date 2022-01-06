import sklearn.pipeline
from sklearn.linear_model import LogisticRegression
from vaex.ml.datasets import load_iris
from vaex.ml.sklearn import Predictor

from goldilox import Pipeline
from goldilox.sklearn.pipeline import SklearnPipeline
from goldilox.vaex.pipeline import VaexPipeline
# from tempfile import TemporaryDirectory; tmpdir = TemporaryDirectory().name  # TODO remove
from tests.test_utils import validate_persistence


def test_skleran_save_load(tmpdir):
    iris = load_iris().to_pandas_df()
    features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    target = 'class_'
    X = iris[features]
    y = iris[target]
    pipeline = SklearnPipeline.from_sklearn(sklearn.pipeline.Pipeline([('regression', LogisticRegression())])).fit(X, y)
    pipeline = validate_persistence(pipeline)
    assert pipeline.inference(pipeline.raw).shape == (1, 5)


def test_vaex_save_load(tmpdir):
    df = load_iris()
    features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    target = 'class_'
    model = Predictor(model=LogisticRegression(), features=features, target=target)
    model.fit(df)
    df = model.transform(df)
    pipeline = VaexPipeline.from_vaex(df)
    # from tempfile import TemporaryDirectory; tmpdir = TemporaryDirectory().name
    path = str(tmpdir) + '/model.pkl'
    pipeline.save(path)
    pipeline = VaexPipeline.from_file(path)
    assert pipeline.inference(pipeline.raw).shape == (1, 6)


def test_goldilox_save_load(tmpdir):
    df = load_iris()
    features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    target = 'class_'
    model = Predictor(model=LogisticRegression(), features=features, target=target)
    model.fit(df)
    df = model.transform(df)
    pipeline = VaexPipeline.from_vaex(df)
    pipeline = validate_persistence(pipeline)
    assert pipeline.pipeline_type == 'vaex'
    pipeline.inference(pipeline.raw).shape == (1, 6)

    df = df.to_pandas_df()
    X = df[features]
    y = df[target]
    pipeline = SklearnPipeline.from_sklearn(sklearn.pipeline.Pipeline([('regression', LogisticRegression())])).fit(X, y)
    path = 'tests/models/sk.pkl'
    pipeline.save(path)
    pipeline = Pipeline.from_file(path)
    assert pipeline.pipeline_type == 'sklearn'
    assert pipeline.inference(pipeline.raw).shape == (1, 5)
